"""Squared-loss fused GW using exact sparse PBM soft-overlap contractions.

The transport plan remains dense O(m*n); sparse similarities K1 and K2 avoid
materializing within-graph dissimilarities C=1-K.
The conditional-gradient method finds a stationary point of nonconvex FGW,
not a certified global optimum.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.metrics import pairwise_distances

from ._sparse import sparse_similarity


def _transport_oracle(a, b, cost):
    try:
        import ot
        return np.asarray(ot.emd(a, b, cost, numItermax=100000), dtype=np.float64)
    except ImportError:
        # Test-friendly fallback; practical only for small matrices.
        from scipy.optimize import linprog
        m, n = cost.shape
        if m*n > 400:
            raise ImportError("Install POT for sparse FGW on larger graphs")
        rr = np.repeat(np.arange(m), n)
        cc = np.tile(np.arange(n), m)
        ids = np.arange(m*n)
        A = sparse.coo_matrix((np.ones(m*n*2),
                               (np.concatenate([rr, m+cc]),
                                np.concatenate([ids, ids]))),
                               shape=(m+n,m*n)).tocsr()
        result = linprog(cost.ravel(), A_eq=A,
                         b_eq=np.r_[a,b], bounds=(0,None), method="highs")
        if not result.success:
            raise RuntimeError(result.message)
        return result.x.reshape(m,n)


def sparse_fgw_distance(pbm_a, pbm_b, config, return_plan=False):
    """FGW via sparse similarities and exact-line-search conditional gradient."""
    if config.relation != "soft_overlap":
        raise ValueError("sparse_fw currently supports only relation='soft_overlap'")
    if config.lambda_geo == 0 and config.lambda_degree == 0:
        raise ValueError("At least one node cost must be positive")

    mass_a = np.asarray(pbm_a.vertex_mass, dtype=np.float64)
    mass_b = np.asarray(pbm_b.vertex_mass, dtype=np.float64)
    ia = np.flatnonzero(mass_a > config.mass_tol)
    ib = np.flatnonzero(mass_b > config.mass_tol)
    if not len(ia) or not len(ib):
        raise ValueError("No vertices have mass above tolerance")
    a = mass_a[ia].copy()
    b = mass_b[ib].copy()
    a /= a.sum()
    b /= b.sum()
    K1 = sparse_similarity(pbm_a, ia)
    K2 = sparse_similarity(pbm_b, ib)

    if config.geo_scale == "epsilon":
        if config.lambda_geo:
            if pbm_a.eps != pbm_b.eps:
                raise ValueError("geo_scale='epsilon' requires the same epsilon")
            geo_scale = pbm_a.eps
        else:
            # Identical to the dense POT path: unused geometric scale.
            geo_scale = 1.0
    else:
        geo_scale = float(config.geo_scale)

    M = np.zeros((len(ia),len(ib)), dtype=np.float64)
    if config.lambda_geo:
        if config.metric is None:
            if (pbm_a.metric != pbm_b.metric or
                dict(pbm_a.metric_kwargs or {}) != dict(pbm_b.metric_kwargs or {})):
                raise ValueError("PBMs have different metrics: specify a comparison metric")
            metric = pbm_a.metric
            kwargs = dict(pbm_a.metric_kwargs or {})
        else:
            metric = config.metric
            kwargs = dict(config.metric_kwargs or {})
        D = pairwise_distances(pbm_a.landmark_points[ia],
                               pbm_b.landmark_points[ib], metric=metric, **kwargs)
        M += config.lambda_geo * (D / geo_scale)**2
    if config.lambda_degree:
        def degree(pbm, indices):
            A = pbm.adjacency_matrix
            degrees = (np.asarray(A.sum(axis=1)).ravel() if sparse.issparse(A)
                       else np.asarray(A).sum(axis=1))
            return (degrees / max(pbm.n_vertices-1,1))[indices]
        M += config.lambda_degree * (degree(pbm_a,ia)[:,None] -
                                     degree(pbm_b,ib)[None,:])**2

    alpha = config.alpha
    relation_scale = 1.0 if config.relation_scale is None else float(config.relation_scale)
    structural_factor = 1.0 / (relation_scale * relation_scale)
    c1 = float(a @ (K1.multiply(K1) @ a))
    c2 = float(b @ (K2.multiply(K2) @ b))

    def contraction(T):
        return np.asarray((K1 @ T) @ K2, dtype=np.float64)

    c1 *= structural_factor
    c2 *= structural_factor

    def eval_terms(T, prod=None):
        prod = contraction(T) if prod is None else prod
        feat = float(np.sum(M*T))
        structure = max(0.0, c1+c2-2.0*structural_factor*float(np.sum(prod*T)))
        return feat, structure

    from ._fgw_optimization import optimize_concave_fgw

    T, diagnostics = optimize_concave_fgw(
        M, K1, K2, a, b, alpha, relation_scale,
        transport_oracle=_transport_oracle,
        max_iter=config.max_iter, tol_rel=config.tol_rel,
        tol_abs=config.tol_abs, n_restarts=config.n_restarts,
        random_state=config.random_state,
        pivot_steps=config.pivot_steps,
        pivot_candidates=config.pivot_candidates,
        time_budget=config.time_budget,
    )
    feat = diagnostics["feature_part"]
    struct = diagnostics["structural_part"]
    objective = diagnostics["objective"]
    converged = diagnostics["converged"]
    iterations = diagnostics["iterations"]
    gap = diagnostics["final_fw_gap"]
    info = {
        "objective": objective,
        "alpha": float(alpha),
        "relation": "soft_overlap",
        "lambda_geo": float(config.lambda_geo),
        "lambda_degree": float(config.lambda_degree),
        "geo_scale": float(geo_scale),
        "relation_scale": relation_scale,
        "feature_part": feat,
        "structural_part": struct,
        "weighted_feature_part": (1-alpha)*feat,
        "weighted_structural_part": alpha*struct,
        "feature_cost_matrix": M,
        "source_similarity_matrix": K1,
        "target_similarity_matrix": K2,
        "source_vertex_indices": ia,
        "target_vertex_indices": ib,
        "source_weights": a,
        "target_weights": b,
        "iterations": iterations,
        "converged": converged,
        "final_fw_gap": float(gap),
        "solver": "sparse_fw",
        "restarts_completed": diagnostics["restarts_completed"],
        "restart_history": diagnostics["restart_history"],
        "pivot_improvements": diagnostics["pivot_improvements"],
        "pivot_gain": diagnostics["pivot_gain"],
        "oracle_seconds": diagnostics["oracle_seconds"],
        "contraction_seconds": diagnostics["contraction_seconds"],
        "oracle_calls": diagnostics["oracle_calls"],
        "contraction_calls": diagnostics["contraction_calls"],
        "elapsed_seconds": diagnostics["elapsed_seconds"],
        "optimization_conclusion": diagnostics["optimization_conclusion"],
    }
    if return_plan:
        info["transport_plan"] = T.copy()
    return (float(np.sqrt(objective)) if config.take_sqrt else float(objective)), info
