"""Matched-start and matched-time PBM-FGW research benchmark.

Commands:
  python benchmarks/bench_fgw_optimization.py --n 250 --landmarks 12 --restarts 4
  python benchmarks/bench_fgw_optimization.py --n 80 --landmarks 5 --restarts 2

POT and sparse FW receive the same random-vertex initialization sequence.
Each method is evaluated with the SAME sparse-similarity objective, regardless
of its internally reported objective. The matched-time experiment uses a soft
budget (uninterruptible linear OT oracle calls can overshoot).
"""
import argparse
import json
from time import perf_counter
import tracemalloc

import numpy as np

from probabilistic_ball_mapper import (
    MembershipConfig, fit_from_landmarks, sparse_similarity,
)
from probabilistic_ball_mapper._fgw_optimization import (
    generate_initial_plans, objective_parts, optimize_concave_fgw,
)
from probabilistic_ball_mapper._sparse_fgw import _transport_oracle


def objective(T,M,K1,K2,p,q,alpha,scale):
    f,s=objective_parts(T,M,K1,K2,p,q,alpha,scale)
    return (1-alpha)*f+alpha*s


def measure(call):
    tracemalloc.start()
    t0=perf_counter()
    x=call()
    elapsed=perf_counter()-t0
    _,peak=tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return x,elapsed,peak


def main():
    import ot
    p=argparse.ArgumentParser()
    p.add_argument("--n",type=int,default=250)
    p.add_argument("--landmarks",type=int,default=12)
    p.add_argument("--restarts",type=int,default=4)
    p.add_argument("--max-iter",type=int,default=150)
    p.add_argument("--seed",type=int,default=45)
    p.add_argument("--alpha",type=float,default=0.5)
    p.add_argument("--scale",type=float,default=1.0)
    args=p.parse_args()
    if args.n<args.landmarks or args.landmarks<2 or args.restarts<1:
        p.error("Require n>=landmarks>=2 and restarts>=1")
    rng=np.random.default_rng(args.seed)
    X=rng.normal(size=(args.n,2))
    Y=X+0.2*rng.normal(size=X.shape)
    from sklearn.metrics import pairwise_distances
    # Ensure strict open-ball coverage for the fixed-landmark benchmark.
    radius=max(
        pairwise_distances(X,X[:args.landmarks]+0.05).min(axis=1).max(),
        pairwise_distances(Y,Y[:args.landmarks]+0.05).min(axis=1).max(),
    )*1.01 + 1e-9
    cfg=MembershipConfig(eps=radius, membership="gaussian")
    a=fit_from_landmarks(X,X[:args.landmarks]+0.05,config=cfg,storage="sparse")
    b=fit_from_landmarks(Y,Y[:args.landmarks]+0.05,config=cfg,storage="sparse")
    x=a.vertex_mass.copy()
    y=b.vertex_mass.copy()
    ia=np.flatnonzero(x>0)
    ib=np.flatnonzero(y>0)
    x=x[ia]/x[ia].sum()
    y=y[ib]/y[ib].sum()
    K1=sparse_similarity(a,ia)
    K2=sparse_similarity(b,ib)
    A=a.landmark_points[ia]
    B=b.landmark_points[ib]
    from sklearn.metrics import pairwise_distances
    M=(pairwise_distances(A,B)/cfg.eps)**2
    C1=(1-K1.toarray())/args.scale
    C2=(1-K2.toarray())/args.scale
    np.fill_diagonal(C1,0)
    np.fill_diagonal(C2,0)

    def run_pot():
        results=[]
        rng_local=np.random.default_rng(args.seed)
        for restart in range(args.restarts):
            G0=(np.outer(x,y) if restart==0 else
                _transport_oracle(x,y,rng_local.standard_normal((len(x),len(y)))))
            t0=perf_counter()
            T,log=ot.gromov.fused_gromov_wasserstein(
                M,C1,C2,p=x,q=y,G0=G0,loss_fun="square_loss",
                alpha=args.alpha,armijo=False,symmetric=True,
                max_iter=args.max_iter,tol_rel=1e-9,tol_abs=1e-9,
                log=True)
            results.append({
                "restart":restart,"objective":float(objective(
                    T,M,K1,K2,x,y,args.alpha,args.scale)),
                "seconds":perf_counter()-t0,
                "marginal_error":float(max(
                    np.max(abs(T.sum(axis=1)-x)),
                    np.max(abs(T.sum(axis=0)-y))))})
        return results

    (pot_runs,pot_sec,pot_peak)=measure(run_pot)

    def run_sparse():
        return optimize_concave_fgw(
            M,K1,K2,x,y,args.alpha,args.scale,
            transport_oracle=_transport_oracle,max_iter=args.max_iter,
            n_restarts=args.restarts,random_state=args.seed,
        )
    ((T,diags),sparse_sec,sparse_peak)=measure(run_sparse)
    def matched_time():
        return optimize_concave_fgw(
            M,K1,K2,x,y,args.alpha,args.scale,
            transport_oracle=_transport_oracle,max_iter=args.max_iter,
            n_restarts=10000,random_state=args.seed,
            time_budget=pot_sec,
        )
    ((Ttime,time_diag),time_sec,time_peak)=measure(matched_time)
    output={
        "dataset":{"n":args.n,"landmarks_a":len(x),"landmarks_b":len(y),
                   "alpha":args.alpha,"scale":args.scale,"seed":args.seed},
        "equal_restart":{
            "restarts":args.restarts,
            "pot_best_objective":min(r["objective"] for r in pot_runs),
            "sparse_best_objective":diags["objective"],
            "pot_seconds":pot_sec,"sparse_seconds":sparse_sec,
            "pot_tracemalloc_peak_bytes":pot_peak,
            "sparse_tracemalloc_peak_bytes":sparse_peak,
            "pot_each":pot_runs,"sparse_each":diags["restart_history"],
            "sparse_oracle_seconds":diags["oracle_seconds"],
            "sparse_contraction_seconds":diags["contraction_seconds"],
            "sparse_oracle_calls":diags["oracle_calls"],
            "sparse_contraction_calls":diags["contraction_calls"],
        },
        "equal_time":{
            "target_seconds":pot_sec,
            "sparse_seconds_actual":time_sec,
            "sparse_best_objective":time_diag["objective"],
            "sparse_restarts_completed":time_diag["restarts_completed"],
            "sparse_peak_bytes":time_peak,
        },
    }
    print(json.dumps(output,indent=2))
    assert np.isfinite(diags["objective"])
    assert np.isfinite(time_diag["objective"])
    assert np.isfinite(min(r["objective"] for r in pot_runs))


if __name__=="__main__":
    main()
