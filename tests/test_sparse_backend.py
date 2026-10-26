"""Sparse PBM regression tests against the unchanged dense mathematical baseline."""
import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import eigsh

from probabilistic_ball_mapper import (
    FGWConfig, MembershipConfig, fit_from_landmarks, fgw_distance,
    sparse_similarity, spectral_operator,
)


@pytest.mark.parametrize("kernel", [
    "uniform", "gaussian", "compact_gaussian", "triangular",
    "epanechnikov", "quartic", "inverse_distance", "inverse_square",
])
def test_sparse_dense_equality_all_kernels(kernel):
    rng = np.random.default_rng(32)
    X = rng.normal(size=(80, 2))
    C = X[:12].copy()
    weights = rng.uniform(0.2, 3, size=len(X))
    cfg = MembershipConfig(eps=4.0, membership=kernel)
    dense = fit_from_landmarks(X, C, config=cfg, sample_weight=weights)
    tiny = fit_from_landmarks(X, C, config=cfg, sample_weight=weights,
                              storage="sparse")
    assert sparse.isspmatrix_csr(tiny.W)
    np.testing.assert_allclose(tiny.W.toarray(), dense.W, atol=1e-12)
    np.testing.assert_allclose(tiny.vertex_mass, dense.vertex_mass, atol=1e-12)
    np.testing.assert_allclose(
        tiny.overlap_matrix.toarray(), dense.overlap_matrix, atol=1e-12)
    np.testing.assert_array_equal(
        tiny.adjacency_matrix.toarray(), dense.adjacency_matrix)
    assert all(np.array_equal(a, b) for a, b in zip(tiny.cover, dense.cover))
    np.testing.assert_allclose(tiny.to_dense().W, dense.W)


def test_strict_boundary_with_extreme_sigma():
    X = np.array([[0.0], [1.0]])
    C = X.copy()
    cfg = MembershipConfig(eps=1.0, membership="gaussian", sigma=1e-10)
    pbm = fit_from_landmarks(X, C, config=cfg, storage="sparse")
    np.testing.assert_array_equal(pbm.W.toarray(), np.eye(2))
    assert pbm.graph.number_of_edges() == 0
    assert [v.tolist() for v in pbm.cover] == [[0], [1]]


def test_uncovered_fallback_does_not_modify_geometric_cover():
    X = np.array([[0.0], [5.0]])
    cfg = MembershipConfig(
        eps=0.5, membership="uniform", out_of_cover_policy="nearest")
    pbm = fit_from_landmarks(
        X, np.array([[0.0]]), config=cfg, storage="sparse")
    assert pbm.outside_cover_indices.tolist() == [1]
    assert pbm.cover[0].tolist() == [0]
    np.testing.assert_array_equal(pbm.W.toarray(), np.ones((2,1)))


def test_sparse_spectral_operator_matches_dense_matrix():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(150, 2))
    pbm = fit_from_landmarks(
        X, X[:30], config=MembershipConfig(eps=1.2), storage="sparse")
    L, idx = spectral_operator(pbm)
    nu = pbm.vertex_mass[idx]
    Q = pbm.overlap_matrix.toarray()[np.ix_(idx, idx)]
    exact = Q / np.sqrt(np.outer(nu, nu))
    Z = rng.normal(size=(len(idx), 4))
    np.testing.assert_allclose(L.matmat(Z), exact @ Z, atol=1e-12)
    values = eigsh(L, k=3, which="LA", return_eigenvectors=False)
    np.testing.assert_allclose(
        np.sort(values), np.linalg.eigvalsh(exact)[-3:], atol=1e-9)


@pytest.mark.parametrize("relation_scale", [1.0, 2.0])
def test_sparse_fgw_exact_objective_and_plan(relation_scale):
    rng = np.random.default_rng(12)
    X = rng.normal(size=(40, 2))
    Y = X + 0.2 * rng.normal(size=X.shape)
    cfg = MembershipConfig(eps=2.0)
    a = fit_from_landmarks(X, X[:5], config=cfg, storage="sparse")
    b = fit_from_landmarks(Y, Y[:6], config=cfg, storage="sparse")
    config = FGWConfig(
        alpha=0.5, solver="sparse_fw", relation="soft_overlap",
        relation_scale=relation_scale, max_iter=60)
    dist, info = fgw_distance(a, b, config=config, return_plan=True)
    T = info["transport_plan"]
    np.testing.assert_allclose(T.sum(axis=1), a.vertex_mass, atol=1e-7)
    np.testing.assert_allclose(T.sum(axis=0), b.vertex_mass, atol=1e-7)
    K1 = sparse_similarity(a).toarray()
    K2 = sparse_similarity(b).toarray()
    C1 = (1-K1) / relation_scale
    C2 = (1-K2) / relation_scale
    structural = (
        np.sum(C1**2 * np.outer(T.sum(axis=1), T.sum(axis=1)))
        + np.sum(C2**2 * np.outer(T.sum(axis=0), T.sum(axis=0)))
        - 2*np.sum(C1 * (T @ C2 @ T.T))
    )
    assert info["structural_part"] == pytest.approx(structural, abs=1e-9)
    assert dist**2 == pytest.approx(
        0.5 * structural + 0.5 * np.sum(T * info["feature_cost_matrix"]),
        abs=1e-9)
    assert np.isfinite(info["final_fw_gap"])


def test_pot_compatibility_requires_explicit_dense_conversion():
    X = np.array([[0.0], [0.5], [1.0]])
    pbm = fit_from_landmarks(
        X, X[:2], config=MembershipConfig(eps=1.5), storage="sparse")
    with pytest.raises(ValueError, match="solver='sparse_fw'"):
        fgw_distance(pbm, pbm)
    assert np.isfinite(fgw_distance(pbm.to_dense(), pbm.to_dense())[0])
