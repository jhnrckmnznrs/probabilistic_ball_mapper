"""Exact sparse localized probabilistic Ball Mapper summaries.

The geometric cover is the strict incidence d(x,c)<eps, and is NEVER
inferred from nonzero floating-point weights (which can underflow).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


def _matrix(x: Any, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) == 0 or not np.isfinite(x).all():
        raise ValueError(f"{name} must be a nonempty finite 2D array")
    return x


def _radius_incidence(X: np.ndarray, C: np.ndarray, eps: float,
                      metric: str, metric_kwargs: dict, batch_size: int = 2048):
    """Return CSR distances on strict open balls and a fitted neighbor index.

    Unsupported sklearn neighbor metrics fall back to chunked exact distances.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    index = None
    try:
        index = NearestNeighbors(metric=metric, metric_params=metric_kwargs,
                                 algorithm="auto").fit(C)
        distances, neighbors = index.radius_neighbors(
            X, radius=eps, return_distance=True, sort_results=True)
        indptr = np.zeros(X.shape[0] + 1, dtype=np.int64)
        counts = np.fromiter((len(j) for j in neighbors), dtype=np.int64,
                             count=X.shape[0])
        indptr[1:] = np.cumsum(counts)
        cols = (np.concatenate(neighbors).astype(np.int64) if indptr[-1]
                else np.array([], dtype=np.int64))
        vals = (np.concatenate(distances).astype(np.float64) if indptr[-1]
                else np.array([], dtype=np.float64))
        # Neighbor queries include the boundary. Discard exact d == eps.
        valid = vals < eps
        if not valid.all():
            rows = np.repeat(np.arange(len(X)), counts)[valid]
            return sparse.csr_matrix((vals[valid], (rows, cols[valid])),
                                     shape=(len(X), len(C))), index
        return sparse.csr_matrix((vals, cols, indptr),
                                 shape=(len(X), len(C))), index
    except (ValueError, TypeError, NotImplementedError):
        # Bounded memory even for metrics not accelerated by NearestNeighbors.
        rr, cc, vv = [], [], []
        for start in range(0, len(X), batch_size):
            D = pairwise_distances(X[start:start + batch_size], C,
                                   metric=metric, **metric_kwargs)
            r, c = np.nonzero(D < eps)
            rr.append(r + start)
            cc.append(c)
            vv.append(D[r, c])
        if not any(len(v) for v in vv):
            return sparse.csr_matrix((len(X), len(C)), dtype=np.float64), None
        return sparse.csr_matrix((np.concatenate(vv), (np.concatenate(rr),
                                 np.concatenate(cc))),
                                 shape=(len(X), len(C))), None


def _sparse_memberships(distances, config, X, C, neighbor_index=None,
                        batch_size=2048):
    n, m = distances.shape
    indptr = distances.indptr
    counts = np.diff(indptr)
    rows = np.repeat(np.arange(n), counts)
    d = distances.data
    inside_rows = np.flatnonzero(counts)
    uncovered = np.flatnonzero(counts == 0)
    k = config.membership
    eps = config.eps
    if k == "uniform":
        weights = np.ones_like(d)
    elif k in ("gaussian", "compact_gaussian"):
        sigma = config.effective_sigma
        smallest = np.minimum.reduceat(d, indptr[inside_rows])
        reference = np.zeros(n, dtype=np.float64)
        reference[inside_rows] = smallest
        dmin = reference[rows]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            score = -((d - dmin) * (d + dmin)) / (2.0 * sigma * sigma)
        score[d == dmin] = 0.0
        score[~np.isfinite(score)] = -np.inf
        if k == "compact_gaussian" and config.taper_power > 0:
            taper = np.maximum(1.0 - (d / eps) ** 2, 0.0)
            with np.errstate(divide="ignore"):
                score += config.taper_power * np.log(taper)
        max_score = np.maximum.reduceat(score, indptr[inside_rows])
        reference[inside_rows] = max_score
        if not np.isfinite(max_score).all():
            raise FloatingPointError("Degenerate Gaussian score")
        weights = np.exp(score - reference[rows])
    elif k == "triangular":
        weights = np.maximum(1.0 - d / eps, 0.0)
    elif k == "epanechnikov":
        weights = np.maximum(1.0 - (d / eps) ** 2, 0.0)
    elif k == "quartic":
        weights = np.maximum(1.0 - (d / eps) ** 2, 0.0) ** 2
    elif k == "inverse_distance":
        weights = 1.0 / (d + config.inverse_offset)
    elif k == "inverse_square":
        weights = 1.0 / (d*d + config.inverse_offset)
    else:
        raise ValueError(f"Unknown membership: {k}")

    totals = np.bincount(rows, weights=weights, minlength=n)
    if np.any(totals[inside_rows] <= 0):
        raise FloatingPointError("Membership weights underflowed")
    weights /= totals[rows]
    W = sparse.csr_matrix((weights, distances.indices.copy(), indptr.copy()),
                          shape=(n, m))
    if len(uncovered):
        if config.out_of_cover_policy == "error":
            raise RuntimeError(f"{len(uncovered)} points are outside the fixed cover. "
                               f"First indices: {uncovered[:10].tolist()}")
        if config.out_of_cover_policy != "nearest":
            raise ValueError("Unknown out-of-cover policy")
        if neighbor_index is not None:
            _, nn = neighbor_index.kneighbors(X[uncovered], n_neighbors=1)
            nn = nn[:, 0]
        else:
            nn = np.empty(len(uncovered), dtype=np.int64)
            for start in range(0, len(uncovered), batch_size):
                batch = uncovered[start:start + batch_size]
                D = pairwise_distances(X[batch], C, metric=config.metric,
                                       **dict(config.metric_kwargs or {}))
                nn[start:start + len(batch)] = D.argmin(axis=1)
        W = W + sparse.csr_matrix((np.ones(len(uncovered)),
                                  (uncovered, nn)), shape=(n, m))
    W.eliminate_zeros()
    return W, uncovered


@dataclass(frozen=True)
class SparseProbabilisticBallMapper:
    landmarks: list[int] | None
    landmark_points: np.ndarray
    cover: list[np.ndarray]
    graph: nx.Graph
    W: sparse.csr_matrix
    vertex_mass: np.ndarray
    overlap_matrix: sparse.csr_matrix
    adjacency_matrix: sparse.csr_matrix
    sample_weight: np.ndarray
    config: Any
    uncovered_indices: np.ndarray

    @property
    def n_points(self): return self.W.shape[0]

    @property
    def n_vertices(self): return self.W.shape[1]

    @property
    def eps(self): return self.config.eps

    @property
    def metric(self): return self.config.metric

    @property
    def metric_kwargs(self): return self.config.metric_kwargs

    @property
    def membership(self): return self.config.membership

    @property
    def sigma(self): return self.config.effective_sigma

    @property
    def outside_cover_indices(self): return self.uncovered_indices

    def to_dense(self):
        """Explicit compatibility conversion; may require O(n*m+m*m) memory."""
        from ._core import ProbabilisticBallMapper
        return ProbabilisticBallMapper(
            landmarks=self.landmarks, landmark_points=self.landmark_points,
            cover=self.cover, graph=self.graph, W=self.W.toarray(),
            vertex_mass=self.vertex_mass, overlap_matrix=self.overlap_matrix.toarray(),
            adjacency_matrix=self.adjacency_matrix.toarray(),
            sample_weight=self.sample_weight, config=self.config)


def summarize_sparse_memberships(W, inside, landmarks, landmark_points,
                                 config, sample_weight=None, uncovered=None):
    """Exact sparse PBM moment, overlap, and geometric graph calculations."""
    W = sparse.csr_matrix(W, dtype=np.float64)
    inside = sparse.csr_matrix(inside, dtype=bool)
    if W.shape != inside.shape:
        raise ValueError("Membership and incidence shapes differ")
    n, m = W.shape
    if n == 0 or m == 0 or not np.isfinite(W.data).all() or (W.data < 0).any():
        raise ValueError("Invalid memberships")
    if not np.allclose(np.asarray(W.sum(axis=1)).ravel(), 1.0, atol=1e-10):
        raise ValueError("Rows of memberships must sum to one")
    if sample_weight is None:
        p = np.full(n, 1.0/n)
    else:
        p = np.asarray(sample_weight, dtype=np.float64)
        if (p.shape != (n,) or not np.isfinite(p).all() or
                (p < 0).any() or p.sum() <= 0):
            raise ValueError("Invalid sample weights")
        p = p / p.sum()
    mass = np.asarray(W.T @ p).ravel()
    Q = (W.T @ sparse.diags(p) @ W).tocsr()
    A = (inside.astype(np.float64).T @ inside.astype(np.float64)).tocsr()
    A.data[:] = 1.0
    A.setdiag(0)
    A.eliminate_zeros()
    A.sort_indices()
    graph = nx.from_scipy_sparse_array(A)
    columns = inside.tocsc()
    cover = [columns.indices[columns.indptr[j]:columns.indptr[j+1]].copy()
             for j in range(m)]
    if uncovered is None:
        uncovered = np.flatnonzero(np.diff(inside.indptr) == 0)
    return SparseProbabilisticBallMapper(
        landmarks=None if landmarks is None else list(map(int, landmarks)),
        landmark_points=landmark_points, cover=cover, graph=graph,
        W=W, vertex_mass=mass, overlap_matrix=Q, adjacency_matrix=A,
        sample_weight=p, config=config,
        uncovered_indices=np.asarray(uncovered, dtype=np.int64))


def fit_pbm_sparse_from_landmarks(X, landmark_points, config, sample_weight=None,
                                  *, batch_size=2048):
    X = _matrix(X, "X")
    C = _matrix(landmark_points, "landmark_points")
    if X.shape[1] != C.shape[1]:
        raise ValueError("Point and landmark dimensions differ")
    D, index = _radius_incidence(X, C, config.eps, config.metric,
                                  dict(config.metric_kwargs or {}), batch_size)
    incidence = D.copy()
    incidence.data[:] = 1
    W, uncovered = _sparse_memberships(D, config, X, C, index, batch_size)
    return summarize_sparse_memberships(W, incidence, None, C, config,
                                        sample_weight, uncovered)


def fit_pbm_sparse(X, config, landmark_config=None, sample_weight=None,
                   *, batch_size=2048):
    from ._core import LandmarkConfig, _load_fast_ballmapper
    X = _matrix(X, "X")
    lc = landmark_config or LandmarkConfig()
    choose, fps = _load_fast_ballmapper()
    strict_eps = float(np.nextafter(config.eps, -np.inf))
    if strict_eps <= 0:
        raise ValueError("eps is too small for strict-radius landmark search")
    kwargs = dict(X=X, eps=strict_eps, method=lc.method,
                  metric=config.metric, leaf_size=lc.leaf_size,
                  metric_kwargs=config.metric_kwargs)
    if lc.selection == "fps":
        ids, _ = fps(**kwargs, start_index=lc.start_index)
    else:
        ids, _ = choose(**kwargs)
    ids = np.asarray(ids, dtype=np.int64)
    if (ids.ndim != 1 or len(ids) == 0 or np.any(ids < 0) or
            np.any(ids >= len(X)) or len(np.unique(ids)) != len(ids)):
        raise RuntimeError("Invalid landmarks returned by fast-ballmapper")
    obj = fit_pbm_sparse_from_landmarks(X, X[ids], config, sample_weight,
                                        batch_size=batch_size)
    from dataclasses import replace
    return replace(obj, landmarks=list(map(int, ids)))


def spectral_operator(pbm, *, tol=0.0):
    """Symmetric reversible Markov operator without materializing the overlap.

    Returns (LinearOperator, active_vertex_indices). The eigenvalues lie in
    [0, 1] up to numerical error; the maximal eigenvalue is 1.
    """
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative")
    mass = np.asarray(pbm.vertex_mass, dtype=np.float64)
    active = np.flatnonzero(mass > tol)
    if not len(active):
        raise ValueError("No positive-mass landmarks")
    W = pbm.W[:, active]
    p = pbm.sample_weight
    invsqrt = 1.0 / np.sqrt(mass[active])

    def matvec(x):
        x = np.asarray(x, dtype=np.float64)
        return invsqrt * np.asarray(
            W.T @ (p * np.asarray(W @ (invsqrt*x)).ravel())).ravel()

    def matmat(X):
        X = np.asarray(X, dtype=np.float64)
        return invsqrt[:, None] * np.asarray(
            W.T @ (p[:, None] * (W @ (invsqrt[:, None]*X))))
    return LinearOperator((len(active), len(active)), matvec=matvec,
                          rmatvec=matvec, matmat=matmat, dtype=np.float64), active


def sparse_similarity(pbm, active=None):
    """K=1-C for PBM soft overlap; K has diagonal 1 and sparse off-diagonal."""
    mass = np.asarray(pbm.vertex_mass)
    if active is None:
        active = np.flatnonzero(mass > 0)
    active = np.asarray(active, dtype=np.int64)
    if np.any(mass[active] <= 0):
        raise ValueError("Similarity requires positive vertex masses")
    Q = sparse.csr_matrix(pbm.overlap_matrix)[active, :][:, active].tocsr()
    inv = 1.0 / np.sqrt(mass[active])
    K = (sparse.diags(inv) @ Q @ sparse.diags(inv)).tocsr()
    K.data = np.clip(K.data, 0.0, 1.0)
    K.setdiag(1.0)
    K.eliminate_zeros()
    return K
