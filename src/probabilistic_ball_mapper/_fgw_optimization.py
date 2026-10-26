"""Research utilities for squared-loss FGW with positive-semidefinite similarities.

For PBM soft-overlap, K=D^-1/2 Q D^-1/2 + diag(1-diag(...)) is PSD.
The exact objective on feasible transport plans is concave:
   F(T) = (1-alpha)<M,T> + alpha/s^2*(c1+c2-2<K1 T K2,T>).
Thus the minimum on each Frank-Wolfe segment is at an endpoint. With
a minimizing linear OT oracle, taking the full step is always nonincreasing.
This does NOT imply globally optimal convergence.
"""
from __future__ import annotations

from itertools import combinations
from math import comb
from time import perf_counter

import numpy as np


def _as_dense(K):
    return np.asarray(K.toarray() if hasattr(K, "toarray") else K, dtype=float)


def objective_parts(T, M, K1, K2, a, b, alpha=0.5, relation_scale=1.0):
    """FGW objective terms for feasible T, including the true squared relation scale."""
    K1, K2 = _as_dense(K1), _as_dense(K2)
    c1 = float(a @ ((K1*K1) @ a))
    c2 = float(b @ ((K2*K2) @ b))
    cross = float(np.sum(T * (K1 @ T @ K2)))
    feat = float(np.sum(M*T))
    structural = (c1+c2-2*cross) / (relation_scale*relation_scale)
    # Cancellation may produce tiny negative roundoff, never affect comparisons.
    return feat, max(0.0, structural)


def _objective_from_parts(feat, structural, alpha):
    return (1.0-alpha)*feat+alpha*structural


def rectangle_vertex_exchange(T, gradient, K1, K2, alpha, relation_scale=1.0,
                              *, max_candidates=1024, rng=None, min_gain=1e-13):
    """Search valid 2x2 mass-preserving cycle moves, accepting only improvements.

    A nonzero pair (i,j),(k,l), i!=k, j!=l, admits
      D=(e_i-e_k)(e_j-e_l)^T,  t=-min(T[i,j],T[k,l]).
    This replaces mass on the diagonal with off-diagonal entries and keeps
    BOTH transport marginals exactly. The quadratic decrease is evaluated
    exactly using the PSD K factors, not a heuristic gradient-only test.
    Sampling only a subset of pairs is an explicitly LOCAL heuristic.
    """
    if max_candidates < 0:
        raise ValueError("max_candidates must be nonnegative")
    T = np.asarray(T, dtype=float)
    # Do not densify K for local exchanges: preserve sparse summary storage.
    diag1 = np.asarray(K1.diagonal()).ravel()
    diag2 = np.asarray(K2.diagonal()).ravel()
    entries = np.argwhere(T > 1e-14)
    total = len(entries)*(len(entries)-1)//2
    if total == 0 or max_candidates == 0:
        return T, 0.0, 0
    if rng is None:
        rng = np.random.default_rng(0)
    if total <= max_candidates:
        pairs = combinations(range(len(entries)), 2)
    else:
        # Random pair proposals. Duplicates are harmless and bounded.
        pairs = ((int(i),int(j)) for i,j in rng.integers(
            len(entries), size=(max_candidates*3,2)) if i != j)
    best_gain = 0.0
    best = None
    examined = 0
    for u, v in pairs:
        i,j = entries[u]
        k,l = entries[v]
        if i == k or j == l:
            continue
        t = -min(float(T[i,j]), float(T[k,l]))
        if t >= -1e-14:
            continue
        # delta F = t * <grad F,D> - 2 alpha t²/s² (u^T K1 u)(v^T K2 v).
        derivative = (gradient[i,j]-gradient[i,l]
                      -gradient[k,j]+gradient[k,l])
        quad = ((diag1[i]+diag1[k]-2*float(K1[i,k]))
                *(diag2[j]+diag2[l]-2*float(K2[j,l])))
        delta = t*derivative - (2*alpha/(relation_scale**2))*t*t*quad
        examined += 1
        if delta < -max(min_gain,1e-13) and -delta > best_gain:
            best_gain = -float(delta)
            best = (int(i),int(k),int(j),int(l),float(t))
        if examined >= max_candidates:
            break
    if best is None:
        return T, 0.0, examined
    i,k,j,l,t = best
    U = T.copy()
    U[i,j] += t
    U[k,l] += t
    U[i,l] -= t
    U[k,j] -= t
    if np.min(U) < -1e-12:
        raise FloatingPointError("Vertex exchange violated nonnegativity")
    # Negative roundoff is clipped only below machine-precision tolerance.
    U[U < 0] = 0
    return U, best_gain, examined


def enumerate_transport_vertices(a, b, max_vertices=30000, tolerance=1e-10):
    """Exhaustive numerical vertex enumeration for *tiny* transport polytopes.

    All basic feasible solutions are supported by at most m+n-1 edges.
    We enumerate all candidate bases of this size (including degenerate
    vertices through their extended bases), checking both marginals.
    A finite float-tolerance numerical global reference, NOT a formal
    rational-arithmetic proof certificate.
    """
    a, b = np.asarray(a,dtype=float), np.asarray(b,dtype=float)
    if (a.ndim!=1 or b.ndim!=1 or min(a.size,b.size)==0
        or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b))
        or np.any(a<=0) or np.any(b<=0)
        or abs(a.sum()-b.sum())>tolerance):
        raise ValueError("Positive compatible transport marginals required")
    m,n=len(a),len(b)
    if comb(m*n, m+n-1)>max_vertices:
        raise ValueError("Polytope too large for exhaustive enumeration")
    # Independent m row constraints plus first n-1 column constraints.
    A=np.zeros((m+n-1,m*n),dtype=float)
    for i in range(m):
        A[i,i*n:(i+1)*n] = 1
    for j in range(n-1):
        A[m+j,j::n] = 1
    rhs = np.r_[a,b[:-1]]
    for indices in combinations(range(m*n), m+n-1):
        B=A[:,indices]
        if np.linalg.matrix_rank(B)<m+n-1:
            continue
        x=np.linalg.solve(B,rhs)
        if np.min(x)<-tolerance:
            continue
        T=np.zeros((m,n),dtype=float)
        T.ravel()[list(indices)]=np.maximum(x,0)
        if (np.max(np.abs(T.sum(axis=1)-a))>tolerance
            or np.max(np.abs(T.sum(axis=0)-b))>tolerance):
            continue
        yield T


def globally_optimal_small_plan(M,K1,K2,a,b,alpha=0.5,
                                relation_scale=1.0,max_bases=30000):
    """Numerically exhaust all transport-polytope vertices, for small m,n."""
    best=np.inf
    plan=None
    count=0
    for T in enumerate_transport_vertices(a,b,max_vertices=max_bases):
        feat,struct=objective_parts(T,M,K1,K2,a,b,alpha,relation_scale)
        value=_objective_from_parts(feat,struct,alpha)
        count+=1
        if value<best:
            best=value
            plan=T.copy()
    if plan is None:
        raise RuntimeError("No feasible transport vertices found")
    return best,plan,{"feasible_bases":count,"numerical_tolerance":1e-10}


def generate_initial_plans(a,b,n_restarts,random_state,transport_oracle):
    """Deterministic product plus reproducible random linear-OT extreme points."""
    if n_restarts<1:
        raise ValueError("n_restarts must be >=1")
    yield np.outer(a,b)
    rng=np.random.default_rng(random_state)
    for _ in range(1,n_restarts):
        # Random cost generates genuine transport-polytope vertices.
        C=rng.standard_normal((len(a),len(b)))
        yield transport_oracle(a,b,C)


def optimize_concave_fgw(M,K1,K2,a,b,alpha,relation_scale,
                         *,transport_oracle,max_iter=1000,tol_rel=1e-9,
                         tol_abs=1e-9,n_restarts=1,random_state=0,
                         pivot_steps=0,pivot_candidates=1024,
                         time_budget=None):
    """Multi-start full-step Frank-Wolfe; best feasible solution is retained.

    The FW gap is a local stationarity measure, *not* a global optimality
    certificate for concave minimization. The time budget is soft: an ongoing
    linear OT oracle call cannot be interrupted.
    """
    if n_restarts<1 or pivot_steps<0 or pivot_candidates<0:
        raise ValueError("Invalid restart/pivot count")
    if time_budget is not None and (not np.isfinite(time_budget) or time_budget<=0):
        raise ValueError("time_budget must be positive")
    if not (0<alpha<1) or not relation_scale>0:
        raise ValueError("Invalid alpha or relation scale")
    started=perf_counter()
    rng=np.random.default_rng(random_state)
    factor=1.0/(relation_scale*relation_scale)
    c1=float(a@(K1.multiply(K1)@a)) if hasattr(K1,"multiply") else float(a@((K1*K1)@a))
    c2=float(b@(K2.multiply(K2)@b)) if hasattr(K2,"multiply") else float(b@((K2*K2)@b))
    oracle_sec=0.0
    contraction_sec=0.0
    n_oracle=0
    n_contraction=0

    def contract(T):
        nonlocal contraction_sec,n_contraction
        tick=perf_counter()
        P=np.asarray((K1@T)@K2,dtype=float)
        contraction_sec+=perf_counter()-tick
        n_contraction+=1
        return P

    def oracle(C):
        nonlocal oracle_sec,n_oracle
        tick=perf_counter()
        ans=np.asarray(transport_oracle(a,b,C),dtype=float)
        oracle_sec+=perf_counter()-tick
        n_oracle+=1
        return ans

    def evaluate(T):
        prod=contract(T)
        feat=float(np.sum(T*M))
        struct=max(0.0, factor*(c1+c2-2*float(np.sum(T*prod))))
        return _objective_from_parts(feat,struct,alpha),feat,struct,prod

    best=np.inf
    best_T=None
    best_info=None
    histories=[]
    completed=0
    init_gen=generate_initial_plans(a,b,n_restarts,random_state,transport_oracle=lambda _a,_b,C:oracle(C))
    for restart, initial in enumerate(init_gen):
        if restart and time_budget is not None and perf_counter()-started>=time_budget:
            break
        T=initial.copy()
        if np.min(T)<-1e-12 or not np.allclose(T.sum(axis=1),a,atol=1e-8) or not np.allclose(T.sum(axis=0),b,atol=1e-8):
            raise ValueError("Invalid initial transport plan")
        converged=False
        iterations=0
        pivots=0
        gain_total=0.0
        for it in range(max_iter):
            if time_budget is not None and it>0 and perf_counter()-started>=time_budget:
                break
            val,feat,struct,P=evaluate(T)
            grad=(1-alpha)*M-4*alpha*factor*P
            S=oracle(grad)
            gap=max(0.0,float(np.sum(grad*(T-S))))
            threshold=max(tol_abs,tol_rel*max(1.0,abs(val)))
            iterations=it+1
            if gap<=threshold:
                converged=True
                break
            # Strict full-step descent follows from K1,K2 PSD.
            # Avoid constructing the quadratic line-search contraction.
            T=S

        if pivot_steps:
            for _ in range(pivot_steps):
                _,_,_,P=evaluate(T)
                grad=(1-alpha)*M-4*alpha*factor*P
                candidate,gain,_=rectangle_vertex_exchange(
                    T,grad,K1,K2,alpha,relation_scale,
                    max_candidates=pivot_candidates,rng=rng)
                if gain<=0:
                    break
                T=candidate
                gain_total+=gain
                pivots+=1
        # Recompute end-of-run gap at returned plan, including after pivots.
        val,feat,struct,P=evaluate(T)
        grad=(1-alpha)*M-4*alpha*factor*P
        S=oracle(grad)
        gap=max(0.0,float(np.sum(grad*(T-S))))
        converged=gap<=max(tol_abs,tol_rel*max(1.0,abs(val)))
        report={"restart":restart,"objective":float(val),"iterations":iterations,
                "converged":bool(converged),"final_fw_gap":float(gap),
                "pivot_improvements":pivots,"pivot_gain":float(gain_total),
                "feature_part":feat,"structural_part":struct}
        histories.append(report)
        completed+=1
        if val<best:
            best=float(val)
            best_T=T.copy()
            best_info=report
        if time_budget is not None and perf_counter()-started>=time_budget:
            break
    if best_T is None:
        raise RuntimeError("No restarts completed")
    metrics=dict(best_info)
    metrics.update({
        "restart_history":histories,"restarts_completed":completed,
        "oracle_seconds":oracle_sec,"contraction_seconds":contraction_sec,
        "oracle_calls":n_oracle,"contraction_calls":n_contraction,
        "elapsed_seconds":perf_counter()-started,
        "optimization_conclusion":"stationary point only; no global guarantee",
    })
    return best_T,metrics


def exact_rational_2x2_certificate(M, K1, K2, a, b, alpha="1/2", scale=1):
    """Globally certify a 2x2 squared-loss FGW minimum using exact fractions.

    All inputs MUST be exact rationals (int, Fraction, or string fraction).
    Floating-point inputs are rejected, since converting an inexact decimal
    float to Fraction would not certify the intended mathematical problem.

    Every 2x2 coupling is determined by t=T[0,0] in the closed interval
    [max(0, a0-b1), min(a0,b0)]. The objective is concave for PSD K and
    therefore takes its minimum at an endpoint. We also test positive
    semidefiniteness exactly, via diagonal signs and determinant.
    """
    from fractions import Fraction

    def q(x):
        if isinstance(x, (float, np.floating)):
            raise TypeError("Use exact integer, Fraction, or string rational inputs")
        return Fraction(x)

    def vec(v):
        if len(v)!=2:
            raise ValueError("Only 2-point marginals are supported")
        return [q(x) for x in v]

    def mat(v):
        if len(v)!=2 or any(len(r)!=2 for r in v):
            raise ValueError("Only 2x2 matrices are supported")
        return [[q(x) for x in row] for row in v]

    a,b=vec(a),vec(b)
    M,K1,K2=mat(M),mat(K1),mat(K2)
    alpha,scale=q(alpha),q(scale)
    if (any(x<=0 for x in a+b) or sum(a)!=sum(b) or sum(a)!=1
        or not 0<alpha<1 or scale<=0):
        raise ValueError("Require positive probability marginals, alpha and scale")
    for K in (K1,K2):
        if (K[0][1]!=K[1][0] or min(K[0][0],K[1][1])<0
            or K[0][0]*K[1][1]-K[0][1]*K[1][0]<0):
            raise ValueError("Rational similarity matrices must be symmetric PSD")

    lo=max(Fraction(0),a[0]-b[1])
    hi=min(a[0],b[0])
    if lo>hi:
        raise RuntimeError("Invalid probability marginals")

    def plan(t):
        return [[t,a[0]-t],[b[0]-t,b[1]-a[0]+t]]

    def objective(T):
        f=sum(M[i][j]*T[i][j] for i in range(2) for j in range(2))
        s=sum((K1[i][k]-K2[j][l])**2*T[i][j]*T[k][l]
              for i in range(2) for j in range(2)
              for k in range(2) for l in range(2))
        return (1-alpha)*f+alpha*s/(scale*scale)

    candidates=[(objective(plan(t)),plan(t)) for t in sorted({lo,hi})]
    opt,T=min(candidates,key=lambda pair:pair[0])
    return opt,T,{
        "endpoint_objectives": tuple(v for v,_ in candidates),
        "feasible_t_interval": (lo,hi),
        "certificate": "exact rational endpoint enumeration + PSD concavity",
    }
