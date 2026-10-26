"""Concave PBM-FGW theorem, multi-start and global-small-case regression tests."""
import numpy as np
import pytest

from probabilistic_ball_mapper import (
    FGWConfig, MembershipConfig, fit_from_landmarks, fgw_distance,
    sparse_similarity,
)
from probabilistic_ball_mapper._fgw_optimization import (
    enumerate_transport_vertices, globally_optimal_small_plan,
    objective_parts, rectangle_vertex_exchange,
)


def _tiny(seed=12):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(75, 2))
    Y = X + 0.4 * rng.normal(size=X.shape)
    cfg = MembershipConfig(eps=2.5, membership="gaussian")
    a = fit_from_landmarks(X, X[:5] + 0.05, config=cfg, storage="sparse")
    b = fit_from_landmarks(Y, Y[:5] + 0.05, config=cfg, storage="sparse")
    return a, b


def test_sparse_similarity_psd_and_exact_full_step():
    a, b = _tiny()
    K1 = sparse_similarity(a).toarray()
    K2 = sparse_similarity(b).toarray()
    assert np.linalg.eigvalsh(K1).min() > -1e-10
    assert np.linalg.eigvalsh(K2).min() > -1e-10
    p = a.vertex_mass
    q = b.vertex_mass
    rng = np.random.default_rng(15)
    M = rng.uniform(size=(len(p), len(q)))
    from probabilistic_ball_mapper._sparse_fgw import _transport_oracle
    T = np.outer(p, q)
    alpha = 0.6
    scale = 2.0
    P = K1 @ T @ K2
    G = (1-alpha)*M - 4*alpha/(scale*scale)*P
    S = _transport_oracle(p, q, G)
    values = []
    for t in np.linspace(0, 1, 21):
        V = (1-t)*T + t*S
        feat, struct = objective_parts(V, M, K1, K2, p, q, alpha, scale)
        values.append((1-alpha)*feat+alpha*struct)
    # A concave objective on a segment takes its minimum at an endpoint;
    # the linear oracle guarantees the t=1 endpoint is no worse than t=0.
    assert values[-1] <= values[0]+1e-10
    assert np.min(values) >= values[-1]-1e-10


def test_multistart_reproducible_and_no_worse_than_single():
    a, b = _tiny(12)
    c1 = FGWConfig(solver="sparse_fw", n_restarts=1,
                   random_state=4, max_iter=75)
    c4 = FGWConfig(solver="sparse_fw", n_restarts=4,
                   random_state=4, max_iter=75)
    d1, r1 = fgw_distance(a,b,config=c1,return_plan=True)
    d4, r4 = fgw_distance(a,b,config=c4,return_plan=True)
    d4again, r4again = fgw_distance(a,b,config=c4,return_plan=True)
    assert d4 <= d1+1e-10
    assert d4again == pytest.approx(d4,abs=1e-10)
    assert r4["restarts_completed"] == 4
    np.testing.assert_allclose(r4["transport_plan"],r4again["transport_plan"],atol=1e-9)
    np.testing.assert_allclose(r4["transport_plan"].sum(axis=1),a.vertex_mass,atol=1e-9)
    np.testing.assert_allclose(r4["transport_plan"].sum(axis=0),b.vertex_mass,atol=1e-9)
    assert len(r4["restart_history"]) == 4
    assert r4["oracle_seconds"] >= 0
    assert r4["contraction_seconds"] >= 0
    assert r4["oracle_calls"] >= 4
    assert r4["final_fw_gap"] >= 0


def test_rectangle_vertex_exchange_preserves_marginals_and_decreases():
    # A deliberately unfavorable diagonal coupling with a linear preference
    # for exchanging both diagonals for off-diagonals.
    T = 0.5*np.eye(2)
    K = np.eye(2)
    M = np.array([[3.0,0.0],[0.0,3.0]])
    alpha = 0.5
    gradient = (1-alpha)*M-4*alpha*(K@T@K)
    U,gain,n = rectangle_vertex_exchange(
        T,gradient,K,K,alpha,max_candidates=10)
    assert n > 0 and gain > 0
    np.testing.assert_allclose(U.sum(axis=0),[0.5,0.5])
    np.testing.assert_allclose(U.sum(axis=1),[0.5,0.5])
    def F(t):
        feat,struct = objective_parts(
            t,M,K,K,np.array([0.5,0.5]),np.array([0.5,0.5]),alpha)
        return (1-alpha)*feat+alpha*struct
    assert F(U) < F(T)-1e-6
    assert F(T)-F(U) == pytest.approx(gain,abs=1e-10)


def test_2x2_exhaustive_global_reference():
    a=np.array([0.6,0.4])
    b=np.array([0.45,0.55])
    M=np.array([[0.0,0.2],[0.9,0.1]])
    K1=np.array([[1.0,0.3],[0.3,1.0]])
    K2=np.array([[1.0,0.5],[0.5,1.0]])
    vertices=list(enumerate_transport_vertices(a,b))
    assert len(vertices)==2
    opt,T,diagnostics=globally_optimal_small_plan(
        M,K1,K2,a,b,alpha=0.4)
    assert diagnostics["feasible_bases"]==2
    assert np.allclose(T.sum(axis=1),a)
    assert np.allclose(T.sum(axis=0),b)
    for v in vertices:
        f,g=objective_parts(v,M,K1,K2,a,b,alpha=0.4)
        assert opt <= 0.6*f+0.4*g+1e-12


def test_3x3_vertex_enumeration_against_interior_and_exact_transport():
    rng=np.random.default_rng(111)
    a=np.array([0.2,0.5,0.3])
    b=np.array([0.3,0.45,0.25])
    M=rng.random((3,3))
    Z=rng.random((3,3))
    K=Z@Z.T
    K/=np.max(np.linalg.eigvalsh(K))
    opt,T,meta=globally_optimal_small_plan(
        M,K,K,a,b,alpha=0.5)
    assert meta["feasible_bases"] >= 1
    f,g=objective_parts(np.outer(a,b),M,K,K,a,b,alpha=0.5)
    assert opt <= 0.5*f+0.5*g+1e-10
    assert opt >= 0


def test_small_polytope_guard():
    with pytest.raises(ValueError,match="too large"):
        list(enumerate_transport_vertices(np.ones(5)/5,np.ones(5)/5))


def test_config_validation_and_budget():
    for kwargs in ({"n_restarts":0},{"pivot_steps":-1},
                   {"time_budget":-1},{"solver":"pot","n_restarts":2}):
        with pytest.raises(ValueError):
            FGWConfig(**kwargs)
    a,b=_tiny(90)
    c=FGWConfig(solver="sparse_fw",n_restarts=16,time_budget=0.2,
                random_state=3,max_iter=15,pivot_steps=1)
    distance,info=fgw_distance(a,b,config=c,return_plan=True)
    assert np.isfinite(distance)
    assert 1<=info["restarts_completed"]<=16
    assert np.isfinite(info["final_fw_gap"])
    assert info["elapsed_seconds"]>0
