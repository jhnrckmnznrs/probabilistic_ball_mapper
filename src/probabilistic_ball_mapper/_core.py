"""
Probabilistic Ball Mapper built on top of fast-ballmapper.

This package uses fast-ballmapper for ordinary Ball Mapper landmark selection and
cover/graph construction, then adds probabilistic point-to-vertex memberships
and fused Gromov--Wasserstein (FGW) comparison with POT.

Typical use
-----------
    import numpy as np
    from probabilistic_ball_mapper import (
        FGWConfig,
        MembershipConfig,
        fit,
        fgw_distance,
    )

    X_a = np.random.normal(size=(500, 2))
    X_b = np.random.normal(loc=0.3, size=(500, 2))

    membership = MembershipConfig(eps=0.6, membership="gaussian")
    pbm_a = fit(X_a, config=membership)
    pbm_b = fit(X_b, config=membership)

    distance, info = fgw_distance(
        pbm_a,
        pbm_b,
        config=FGWConfig(alpha=0.5, relation="soft_overlap"),
        return_plan=True,
    )
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import pairwise_distances

try:
    import networkx as nx
except Exception as exc:
    raise ImportError(
        "networkx is required. Install fast-ballmapper or networkx."
    ) from exc

Array = NDArray[np.float64]


def _load_fast_ballmapper() -> tuple[Any, Any]:
    """Import fast-ballmapper only when landmark selection is requested."""

    try:
        from fast_ballmapper import compute_landmarks, compute_landmarks_fps
    except Exception as exc:  # pragma: no cover - exercised only without dependency
        raise ImportError(
            "fit() requires fast-ballmapper for automatic landmark selection. "
            "Install it with `pip install fast-ballmapper`, or use "
            "fit_from_landmarks() with precomputed landmark coordinates."
        ) from exc
    return compute_landmarks, compute_landmarks_fps


def _build_mapper_graph(cover: Sequence[np.ndarray]) -> nx.Graph:
    """Build the Ball Mapper intersection graph from cover index lists."""

    graph = nx.Graph()
    graph.add_nodes_from(range(len(cover)))

    point_to_vertices: dict[int, list[int]] = {}
    for vertex, members in enumerate(cover):
        for point_index in np.asarray(members, dtype=np.int64):
            point_to_vertices.setdefault(int(point_index), []).append(vertex)

    for vertices in point_to_vertices.values():
        if len(vertices) > 1:
            graph.add_edges_from(combinations(sorted(set(vertices)), 2))

    return graph


Membership = Literal[
    "uniform",
    "gaussian",
    "compact_gaussian",
    "triangular",
    "epanechnikov",
    "quartic",
    "inverse_distance",
    "inverse_square",
]
OutOfCoverPolicy = Literal["error", "nearest"]
Relation = Literal["soft_overlap", "shortest_path", "adjacency", "landmark_distance"]


@dataclass(frozen=True)
class MembershipConfig:
    """Configuration for probabilistic memberships on epsilon balls.

    Parameters
    ----------
    eps:
        Ball radius used for the Ball Mapper cover. Must be positive.
    membership:
        Radial weighting rule used inside each epsilon ball.
    sigma:
        Gaussian bandwidth. If omitted for Gaussian-style memberships, uses
        ``0.5 * eps``.
    metric:
        Distance metric passed to ``sklearn.metrics.pairwise_distances`` and,
        for landmark selection, to fast-ballmapper.
    metric_kwargs:
        Optional metric-specific keyword arguments.
    taper_power:
        Taper exponent used by ``compact_gaussian``.
    inverse_offset:
        Small positive offset used by inverse-distance memberships.
    out_of_cover_policy:
        ``"nearest"`` assigns uncovered points to their nearest landmark;
        ``"error"`` raises instead.
    """

    eps: float
    membership: Membership = "gaussian"
    sigma: float | None = None
    metric: str = "euclidean"
    metric_kwargs: Mapping[str, Any] | None = None
    taper_power: float = 2.0
    inverse_offset: float = 1e-12
    out_of_cover_policy: OutOfCoverPolicy = "nearest"

    def __post_init__(self) -> None:
        if self.eps <= 0:
            raise ValueError("eps must be positive.")
        valid_memberships = set(Membership.__args__)
        if self.membership not in valid_memberships:
            raise ValueError(
                "membership must be one of "
                + ", ".join(sorted(valid_memberships))
                + "."
            )
        if self.sigma is not None and self.sigma <= 0:
            raise ValueError("sigma must be positive when provided.")
        if self.taper_power < 0:
            raise ValueError("taper_power cannot be negative.")
        if self.inverse_offset <= 0:
            raise ValueError("inverse_offset must be positive.")
        if self.out_of_cover_policy not in {"error", "nearest"}:
            raise ValueError("out_of_cover_policy must be 'error' or 'nearest'.")

    @property
    def effective_sigma(self) -> float | None:
        if self.membership in {"gaussian", "compact_gaussian"}:
            return 0.5 * self.eps if self.sigma is None else float(self.sigma)
        return None


@dataclass(frozen=True)
class ProbabilisticBallMapper:
    """Static probabilistic Ball Mapper summary."""

    landmarks: list[int] | None
    landmark_points: Array
    cover: list[NDArray[np.int_]]
    graph: nx.Graph

    W: Array
    vertex_mass: Array
    overlap_matrix: Array
    adjacency_matrix: Array

    config: MembershipConfig

    @property
    def n_vertices(self) -> int:
        return int(self.W.shape[1])

    @property
    def n_points(self) -> int:
        return int(self.W.shape[0])

    @property
    def eps(self) -> float:
        return self.config.eps

    @property
    def metric(self) -> str:
        return self.config.metric

    @property
    def membership(self) -> Membership:
        return self.config.membership

    @property
    def sigma(self) -> float | None:
        return self.config.effective_sigma

    @property
    def metric_kwargs(self) -> Mapping[str, Any] | None:
        return self.config.metric_kwargs


def _as_2d_float(X: NDArray[Any], name: str = "X") -> Array:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array.")
    if X.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one point.")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{name} contains non-finite values.")
    return X


def distance_to_landmarks(
    X: NDArray[Any],
    landmark_points: NDArray[Any],
    metric: str = "euclidean",
    metric_kwargs: Mapping[str, Any] | None = None,
) -> Array:
    """Distances from rows of X to landmark points.

    Uses sklearn.metrics.pairwise_distances so the same metric can usually be
    used as in fast-ballmapper's BallTree backend.  For FAISS, use ``euclidean``
    or ``cosine``.
    """

    X = _as_2d_float(X, "X")
    C = _as_2d_float(landmark_points, "landmark_points")
    if X.shape[1] != C.shape[1]:
        raise ValueError("X and landmark_points must have the same feature dimension.")
    metric_kwargs = dict(metric_kwargs or {})
    D = pairwise_distances(X, C, metric=metric, **metric_kwargs)
    D = np.asarray(D, dtype=np.float64)
    if not np.all(np.isfinite(D)):
        raise ValueError("The metric produced non-finite distances.")
    return np.maximum(D, 0.0)


def _radial_membership_weights(
    D: Array,
    inside: np.ndarray,
    eps: float,
    membership: Membership,
    sigma: float | None,
    taper_power: float,
    inverse_offset: float,
) -> Array:
    """Unnormalized subordinate radial weights from distances to landmarks."""

    if membership == "uniform":
        return inside.astype(np.float64)

    if membership in {"gaussian", "compact_gaussian"}:
        if sigma is None or sigma <= 0:
            raise ValueError("sigma must be positive for Gaussian memberships.")
        weights = np.exp(-(D * D) / (2.0 * sigma * sigma)) * inside
        if membership == "compact_gaussian":
            taper = np.clip(1.0 - (D * D) / (eps * eps), 0.0, 1.0)
            weights *= taper**taper_power
        return weights.astype(np.float64)

    scaled = np.clip(D / eps, 0.0, np.inf)

    if membership == "triangular":
        weights = np.clip(1.0 - scaled, 0.0, 1.0) * inside
    elif membership == "epanechnikov":
        weights = np.clip(1.0 - scaled * scaled, 0.0, 1.0) * inside
    elif membership == "quartic":
        base = np.clip(1.0 - scaled * scaled, 0.0, 1.0)
        weights = (base * base) * inside
    elif membership == "inverse_distance":
        if inverse_offset <= 0:
            raise ValueError("inverse_offset must be positive.")
        weights = (1.0 / (D + inverse_offset)) * inside
    elif membership == "inverse_square":
        if inverse_offset <= 0:
            raise ValueError("inverse_offset must be positive.")
        weights = (1.0 / (D * D + inverse_offset)) * inside
    else:
        raise ValueError(f"Unknown membership rule: {membership!r}.")

    return weights.astype(np.float64)


def compute_membership_matrix(
    X: NDArray[Any],
    landmark_points: NDArray[Any],
    config: MembershipConfig,
) -> tuple[Array, np.ndarray]:
    """Compute probabilistic memberships supported on epsilon balls."""

    X = _as_2d_float(X, "X")
    C = _as_2d_float(landmark_points, "landmark_points")

    eps = config.eps
    membership = config.membership
    sigma = config.effective_sigma
    metric = config.metric
    metric_kwargs = config.metric_kwargs
    taper_power = config.taper_power
    inverse_offset = config.inverse_offset
    out_of_cover_policy = config.out_of_cover_policy

    if eps <= 0:
        raise ValueError("eps must be positive.")
    if taper_power < 0:
        raise ValueError("taper_power cannot be negative.")

    valid_memberships = {
        "uniform",
        "gaussian",
        "compact_gaussian",
        "triangular",
        "epanechnikov",
        "quartic",
        "inverse_distance",
        "inverse_square",
    }
    if membership not in valid_memberships:
        raise ValueError(
            "membership must be one of " + ", ".join(sorted(valid_memberships)) + "."
        )

    if membership in {"gaussian", "compact_gaussian"}:
        if sigma is None or sigma <= 0:
            raise ValueError("sigma must be positive.")

    D = distance_to_landmarks(X, C, metric=metric, metric_kwargs=metric_kwargs)
    inside = D < eps

    weights = _radial_membership_weights(
        D=D,
        inside=inside,
        eps=eps,
        membership=membership,
        sigma=sigma,
        taper_power=taper_power,
        inverse_offset=inverse_offset,
    )

    row_sums = weights.sum(axis=1)
    uncovered = np.flatnonzero(row_sums <= 0.0)

    if uncovered.size:
        if out_of_cover_policy == "error":
            raise RuntimeError(
                f"{uncovered.size} points are outside the fixed cover. "
                f"First indices: {uncovered[:10].tolist()}"
            )
        if out_of_cover_policy != "nearest":
            raise ValueError("out_of_cover_policy must be 'error' or 'nearest'.")
        nearest = np.argmin(D[uncovered], axis=1)
        weights[uncovered] = 0.0
        weights[uncovered, nearest] = 1.0
        row_sums[uncovered] = 1.0

    W = weights / row_sums[:, None]

    if not np.allclose(W.sum(axis=1), 1.0, atol=1e-10):
        raise AssertionError("Membership rows do not sum to one.")
    if np.any(W < -1e-14):
        raise AssertionError("Membership matrix contains negative entries.")

    return W.astype(np.float64), uncovered.astype(np.int64)


def cover_from_membership(W: NDArray[Any], tol: float = 0.0) -> list[np.ndarray]:
    """Build cover index lists from positive memberships."""

    W = np.asarray(W, dtype=np.float64)
    return [np.flatnonzero(W[:, j] > tol).astype(np.int64) for j in range(W.shape[1])]


def summarize_static_pbm(
    W: NDArray[Any],
    graph: nx.Graph,
    landmarks: Sequence[int] | None,
    landmark_points: NDArray[Any],
    cover: Sequence[np.ndarray] | None,
    config: MembershipConfig,
) -> ProbabilisticBallMapper:
    """Construct summary matrices from an already-computed membership matrix."""

    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 2:
        raise ValueError("W must be a two-dimensional membership matrix.")
    if not np.allclose(W.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("Rows of W must sum to one.")

    n = max(W.shape[0], 1)
    vertex_mass = W.mean(axis=0)
    overlap_matrix = (W.T @ W) / float(n)

    graph = graph.copy()
    graph.add_nodes_from(range(W.shape[1]))
    adjacency_matrix = nx.to_numpy_array(
        graph, nodelist=list(range(W.shape[1])), dtype=float
    )

    return ProbabilisticBallMapper(
        landmarks=None if landmarks is None else [int(i) for i in landmarks],
        landmark_points=np.asarray(landmark_points, dtype=np.float64),
        cover=list(cover) if cover is not None else cover_from_membership(W),
        graph=graph,
        W=W,
        vertex_mass=vertex_mass,
        overlap_matrix=overlap_matrix,
        adjacency_matrix=adjacency_matrix,
        config=config,
    )


LandmarkSelection = Literal["random", "fps"]


@dataclass(frozen=True)
class LandmarkConfig:
    """Configuration for fast-ballmapper landmark selection.

    Parameters
    ----------
    selection:
        ``"fps"`` uses farthest-point sampling; ``"random"`` delegates to
        ``fast_ballmapper.compute_landmarks``.
    method:
        Neighbor-search backend used by fast-ballmapper.
    start_index:
        Optional starting point for farthest-point sampling.
    leaf_size:
        BallTree leaf size when ``method="ball_tree"``.
    """

    selection: LandmarkSelection = "fps"
    method: Literal["ball_tree", "faiss"] = "ball_tree"
    start_index: int | None = None
    leaf_size: int = 40

    def __post_init__(self) -> None:
        if self.selection not in {"random", "fps"}:
            raise ValueError("selection must be either 'random' or 'fps'.")
        if self.method not in {"ball_tree", "faiss"}:
            raise ValueError("method must be either 'ball_tree' or 'faiss'.")
        if self.start_index is not None and self.start_index < 0:
            raise ValueError("start_index cannot be negative.")
        if self.leaf_size <= 0:
            raise ValueError("leaf_size must be positive.")


def fit_pbm(
    X: NDArray[Any],
    config: MembershipConfig,
    landmark_config: LandmarkConfig | None = None,
) -> ProbabilisticBallMapper:
    """Fit a probabilistic Ball Mapper summary from a point cloud.

    This is the main construction routine. It first asks fast-ballmapper to
    select landmarks and build the ordinary Ball Mapper cover/graph, then
    computes a row-stochastic membership matrix over the resulting vertices.

    Parameters
    ----------
    X:
        Data matrix of shape ``(n_points, n_features)``.
    config:
        Membership and metric configuration.
    landmark_config:
        Landmark-selection configuration.

    Returns
    -------
    ProbabilisticBallMapper
        Static PBM summary containing landmarks, cover, graph, memberships,
        vertex masses, overlaps, and adjacency matrix.
    """

    X = _as_2d_float(X, "X")
    if landmark_config is None:
        landmark_config = LandmarkConfig()

    compute_landmarks, compute_landmarks_fps = _load_fast_ballmapper()

    if landmark_config.selection == "fps":
        landmarks, cover = compute_landmarks_fps(
            X,
            eps=config.eps,
            start_index=landmark_config.start_index,
            method=landmark_config.method,
            metric=config.metric,
            leaf_size=landmark_config.leaf_size,
            metric_kwargs=config.metric_kwargs,
        )

    elif landmark_config.selection == "random":
        landmarks, cover = compute_landmarks(
            X,
            eps=config.eps,
            method=landmark_config.method,
            metric=config.metric,
            leaf_size=landmark_config.leaf_size,
            metric_kwargs=config.metric_kwargs,
        )

    else:
        raise ValueError("landmark_config.selection must be either 'fps' or 'random'.")

    landmark_points = X[np.asarray(landmarks, dtype=int)]
    graph = _build_mapper_graph(cover)

    W, _ = compute_membership_matrix(
        X,
        landmark_points=landmark_points,
        config=config,
    )

    return summarize_static_pbm(
        W=W,
        graph=graph,
        landmarks=landmarks,
        landmark_points=landmark_points,
        cover=cover,
        config=config,
    )


def fit_pbm_from_landmarks(
    X: NDArray[Any],
    landmark_points: NDArray[Any],
    config: MembershipConfig,
) -> ProbabilisticBallMapper:
    """Project a dataset onto an already fixed landmark cover."""

    X = _as_2d_float(X, "X")
    landmark_points = _as_2d_float(landmark_points, "landmark_points")

    W, _ = compute_membership_matrix(
        X,
        landmark_points=landmark_points,
        config=config,
    )

    cover = cover_from_membership(W, tol=0.0)
    graph = _build_mapper_graph(cover)

    return summarize_static_pbm(
        W=W,
        graph=graph,
        landmarks=None,
        landmark_points=landmark_points,
        cover=cover,
        config=config,
    )


def normalized_degrees(pbm: ProbabilisticBallMapper) -> Array:
    """Normalized graph degrees in [0, 1]."""

    A = np.asarray(pbm.adjacency_matrix, dtype=np.float64)
    n = A.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=np.float64)
    return A.sum(axis=1) / float(n - 1)


def overlap_spread(pbm: ProbabilisticBallMapper) -> Array:
    """Fraction of each vertex mass shared with other vertices."""

    nu = np.asarray(pbm.vertex_mass, dtype=np.float64)
    diag = np.diag(np.asarray(pbm.overlap_matrix, dtype=np.float64))
    spread = np.zeros_like(nu)
    mask = nu > 0
    spread[mask] = 1.0 - diag[mask] / nu[mask]
    return np.clip(spread, 0.0, 1.0)


def vertex_feature_cost_matrix(
    pbm_a: ProbabilisticBallMapper,
    pbm_b: ProbabilisticBallMapper,
    p: float = 2.0,
    lambda_geo: float = 1.0,
    lambda_degree: float = 0.0,
    metric: str | None = None,
    metric_kwargs: Mapping[str, Any] | None = None,
) -> Array:
    """Ground cost matrix between PBM vertices.

    The returned matrix contains d_phi(i, j)^p, so it can be used directly in
    the discrete p-Wasserstein objective.
    """

    if p <= 0:
        raise ValueError("p must be positive.")
    if lambda_geo < 0 or lambda_degree < 0:
        raise ValueError("lambda weights must be nonnegative.")
    if lambda_geo == 0 and lambda_degree == 0:
        raise ValueError("At least one lambda weight must be positive.")

    m, n = pbm_a.n_vertices, pbm_b.n_vertices
    C = np.zeros((m, n), dtype=np.float64)

    if lambda_geo > 0:
        ground_metric = metric or pbm_a.metric
        C += (
            lambda_geo
            * pairwise_distances(
                pbm_a.landmark_points,
                pbm_b.landmark_points,
                metric=ground_metric,
                **dict(metric_kwargs or pbm_a.metric_kwargs or {}),
            )
            ** p
        )

    if lambda_degree > 0:
        da = normalized_degrees(pbm_a)[:, None]
        db = normalized_degrees(pbm_b)[None, :]
        C += lambda_degree * np.abs(da - db) ** p

    if not np.all(np.isfinite(C)):
        raise ValueError("The vertex cost matrix contains non-finite values.")
    return C


def _normalize_cost_matrix(
    C: NDArray[Any], eps: float = 1e-12
) -> tuple[Array, float, float]:
    """Shift and rescale a nonnegative cost matrix to the unit interval.

    Returns
    -------
    C_norm:
        Normalized matrix with minimum 0 and maximum 1, unless the input is
        constant, in which case the zero matrix is returned.
    offset:
        The value subtracted from the input matrix.
    scale:
        The positive value divided out after shifting.  If the input is
        constant, ``scale`` is 0.
    """

    C_arr = np.asarray(C, dtype=np.float64)
    if C_arr.size == 0:
        return C_arr.copy(), 0.0, 0.0
    if not np.all(np.isfinite(C_arr)):
        raise ValueError("Cannot normalize a cost matrix with non-finite values.")

    offset = float(np.min(C_arr))
    shifted = C_arr - offset
    scale = float(np.max(shifted))
    if scale <= eps:
        return np.zeros_like(C_arr, dtype=np.float64), offset, 0.0
    return (shifted / scale).astype(np.float64), offset, scale


def relation_matrix(
    pbm: ProbabilisticBallMapper,
    relation: Relation = "soft_overlap",
    normalize: bool = True,
) -> Array:
    """Within-graph relation matrix for FGW.

    The returned matrix C is a dissimilarity matrix on PBM vertices.

    relation="soft_overlap"
        Uses 1 - Q_ij / sqrt(nu_i nu_j), with diagonal set to 0.
        This emphasizes probabilistic shared membership.
    relation="shortest_path"
        Uses graph shortest-path distance; disconnected pairs receive one more
        than the largest finite distance before optional normalization.
    relation="adjacency"
        Uses 0 for equal/adjacent vertices and 1 for non-adjacent vertices.
    relation="landmark_distance"
        Uses metric distances between landmark coordinates.
    """

    if relation not in {
        "soft_overlap",
        "shortest_path",
        "adjacency",
        "landmark_distance",
    }:
        raise ValueError(
            "relation must be one of 'soft_overlap', 'shortest_path', "
            "'adjacency', or 'landmark_distance'."
        )

    n = pbm.n_vertices

    if relation == "soft_overlap":
        Q = np.asarray(pbm.overlap_matrix, dtype=np.float64)
        nu = np.asarray(pbm.vertex_mass, dtype=np.float64)
        denom = np.sqrt(np.outer(nu, nu))
        S = np.zeros_like(Q)
        mask = denom > 0
        S[mask] = Q[mask] / denom[mask]
        S = np.clip(S, 0.0, 1.0)
        C = 1.0 - S
        np.fill_diagonal(C, 0.0)

    elif relation == "shortest_path":
        C = np.full((n, n), np.inf, dtype=np.float64)
        np.fill_diagonal(C, 0.0)
        lengths = dict(nx.all_pairs_shortest_path_length(pbm.graph))
        for i, targets in lengths.items():
            if 0 <= int(i) < n:
                for j, dist in targets.items():
                    if 0 <= int(j) < n:
                        C[int(i), int(j)] = float(dist)
        finite = np.isfinite(C)
        max_finite = float(np.max(C[finite])) if np.any(finite) else 0.0
        C[~finite] = max_finite + 1.0

    elif relation == "adjacency":
        A = np.asarray(pbm.adjacency_matrix, dtype=np.float64)
        C = 1.0 - np.clip(A, 0.0, 1.0)
        np.fill_diagonal(C, 0.0)

    else:  # relation == "landmark_distance"
        C = pairwise_distances(
            pbm.landmark_points,
            pbm.landmark_points,
            metric=pbm.metric,
            **dict(pbm.metric_kwargs or {}),
        )
        C = np.asarray(C, dtype=np.float64)
        np.fill_diagonal(C, 0.0)

    if normalize:
        max_val = float(np.max(C)) if C.size else 0.0
        if max_val > 0:
            C = C / max_val

    if not np.all(np.isfinite(C)):
        raise ValueError("The relation matrix contains non-finite values.")
    return C.astype(np.float64)


def _positive_mass_indices(mass: NDArray[Any], tol: float) -> np.ndarray:
    mass = np.asarray(mass, dtype=np.float64)
    if tol < 0:
        raise ValueError("mass_tol must be nonnegative.")
    idx = np.flatnonzero(mass > tol)
    if idx.size == 0:
        raise ValueError("No vertices have mass above mass_tol.")
    return idx.astype(np.int64)


@dataclass(frozen=True)
class FGWConfig:
    """Configuration for fused Gromov--Wasserstein PBM comparison.

    Parameters
    ----------
    alpha:
        Trade-off between vertex feature cost and graph/overlap structure.
        Values near 0 emphasize feature geometry; values near 1 emphasize
        relation structure.
    relation:
        Within-graph relation matrix to compare.
    lambda_geo:
        Weight for landmark-coordinate feature cost.
    lambda_degree:
        Weight for normalized graph-degree feature cost.
    normalize_features:
        Whether to shift and rescale the cross-graph feature cost to [0, 1].
    normalize_relations:
        Whether to rescale relation matrices before FGW.
    mass_tol:
        Drop vertices with mass less than or equal to this tolerance.
    take_sqrt:
        Return ``sqrt(objective)`` instead of the raw objective.
    max_iter, tol_rel, tol_abs, verbose:
        Solver options forwarded to POT.
    """

    alpha: float = 0.5
    relation: Relation = "soft_overlap"
    lambda_geo: float = 1.0
    lambda_degree: float = 0.0
    metric: str | None = None
    metric_kwargs: Mapping[str, Any] | None = None
    normalize_features: bool = True
    normalize_relations: bool = True
    mass_tol: float = 0.0
    take_sqrt: bool = True
    max_iter: int = 1000
    tol_rel: float = 1e-9
    tol_abs: float = 1e-9
    verbose: bool = False

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must satisfy 0 < alpha < 1.")
        if self.relation not in Relation.__args__:
            raise ValueError(
                "relation must be one of " + ", ".join(Relation.__args__) + "."
            )
        if self.lambda_geo < 0 or self.lambda_degree < 0:
            raise ValueError("lambda weights must be nonnegative.")
        if self.lambda_geo == 0 and self.lambda_degree == 0:
            raise ValueError("At least one lambda weight must be positive.")
        if self.mass_tol < 0:
            raise ValueError("mass_tol must be nonnegative.")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if self.tol_rel < 0 or self.tol_abs < 0:
            raise ValueError("solver tolerances must be nonnegative.")


def fused_gromov_wasserstein_distance(
    pbm_a: ProbabilisticBallMapper,
    pbm_b: ProbabilisticBallMapper,
    config: FGWConfig | None = None,
    return_plan: bool = False,
) -> tuple[float, dict[str, Any]]:
    """Fused Gromov--Wasserstein comparison of two PBM graphs.

    This is the main comparison routine for independently fitted PBM graphs.
    It uses POT's conditional-gradient FGW solver.  The cross-graph feature
    cost M compares landmark geometry and optional vertex summaries, while the
    within-graph structure matrices C1 and C2 encode the chosen PBM relation.

    Parameters
    ----------
    alpha:
        Trade-off between feature cost and structure cost.  Values near 0 use
        mostly vertex features.  Values near 1 use mostly graph/overlap
        structure.  POT expects 0 < alpha < 1.
    normalize_features:
        If True, shift and rescale the cross-graph feature cost matrix to
        [0, 1]. This makes alpha easier to interpret relative to the normalized
        structural cost matrices.
    relation:
        Which within-graph relation matrix to use: "soft_overlap",
        "shortest_path", "adjacency", or "landmark_distance".
    return_plan:
        If True, return the optimal transport plan in info["transport_plan"].
    take_sqrt:
        If True, return sqrt(objective).  This is convenient when M contains
        squared feature distances and POT uses square_loss for relations.

    Returns
    -------
    distance:
        sqrt(FGW objective) if take_sqrt=True, otherwise the raw FGW objective.
    info:
        Dictionary containing the raw objective, feature cost, relation
        matrices, positive-mass vertex indices, and optionally the transport
        plan and POT log.
    """

    if config is None:
        config = FGWConfig()

    alpha = config.alpha
    relation = config.relation
    lambda_geo = config.lambda_geo
    lambda_degree = config.lambda_degree
    metric = config.metric
    metric_kwargs = config.metric_kwargs
    normalize_features = config.normalize_features
    normalize_relations = config.normalize_relations
    mass_tol = config.mass_tol
    take_sqrt = config.take_sqrt
    max_iter = config.max_iter
    tol_rel = config.tol_rel
    tol_abs = config.tol_abs
    verbose = config.verbose

    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must satisfy 0 < alpha < 1 for POT's FGW solver.")

    try:
        import ot
    except Exception as exc:
        raise ImportError(
            "fused_gromov_wasserstein_distance requires POT. "
            "Install it with `pip install POT`."
        ) from exc

    a_full = np.asarray(pbm_a.vertex_mass, dtype=np.float64)
    b_full = np.asarray(pbm_b.vertex_mass, dtype=np.float64)
    idx_a = _positive_mass_indices(a_full, mass_tol)
    idx_b = _positive_mass_indices(b_full, mass_tol)

    a = np.maximum(a_full[idx_a], 0.0)
    b = np.maximum(b_full[idx_b], 0.0)
    if a.sum() <= 0 or b.sum() <= 0:
        raise ValueError("Positive-mass vertices must have positive total mass.")
    a = a / a.sum()
    b = b / b.sum()

    M_full = vertex_feature_cost_matrix(
        pbm_a,
        pbm_b,
        p=2.0,
        lambda_geo=lambda_geo,
        lambda_degree=lambda_degree,
        metric=metric,
        metric_kwargs=metric_kwargs,
    )
    M_raw = np.asarray(M_full[np.ix_(idx_a, idx_b)], dtype=np.float64)
    if normalize_features:
        M, feature_cost_offset, feature_cost_scale = _normalize_cost_matrix(M_raw)
    else:
        M = M_raw
        feature_cost_offset = 0.0
        feature_cost_scale = 1.0

    C1_full = relation_matrix(pbm_a, relation=relation, normalize=normalize_relations)
    C2_full = relation_matrix(pbm_b, relation=relation, normalize=normalize_relations)
    C1 = np.asarray(C1_full[np.ix_(idx_a, idx_a)], dtype=np.float64)
    C2 = np.asarray(C2_full[np.ix_(idx_b, idx_b)], dtype=np.float64)

    if return_plan:
        result = ot.gromov.fused_gromov_wasserstein(
            M,
            C1,
            C2,
            p=a,
            q=b,
            loss_fun="square_loss",
            alpha=alpha,
            symmetric=True,
            armijo=False,
            log=True,
            max_iter=max_iter,
            tol_rel=tol_rel,
            tol_abs=tol_abs,
            verbose=verbose,
        )
        if isinstance(result, tuple):
            T, log = result
        else:
            T, log = result, {}
        objective = float(log.get("fgw_dist", np.nan))
        if not np.isfinite(objective):
            # Fallback for POT versions whose log key differs.
            feature_part = float(np.sum(T * M))
            structural_part = _square_loss_structure_cost(C1, C2, T)
            objective = (1.0 - alpha) * feature_part + alpha * structural_part
    else:
        result = ot.gromov.fused_gromov_wasserstein2(
            M,
            C1,
            C2,
            p=a,
            q=b,
            loss_fun="square_loss",
            alpha=alpha,
            symmetric=True,
            armijo=False,
            log=True,
            max_iter=max_iter,
            tol_rel=tol_rel,
            tol_abs=tol_abs,
            verbose=verbose,
        )
        if isinstance(result, tuple):
            objective, log = result
        else:
            objective, log = result, {}
        objective = float(objective)
        T = log.get("T") if isinstance(log, dict) else None

    objective = max(float(objective), 0.0)

    # Report the two terms evaluated at the returned transport plan.  These
    # diagnostics are important because alpha is meaningful only after the
    # feature and structure costs have been put on comparable numerical scales.
    feature_part = np.nan
    structural_part = np.nan
    weighted_feature_part = np.nan
    weighted_structural_part = np.nan
    if T is not None:
        T_arr = np.asarray(T, dtype=np.float64)
        feature_part = float(np.sum(T_arr * M))
        structural_part = _square_loss_structure_cost(C1, C2, T_arr)
        weighted_feature_part = float((1.0 - alpha) * feature_part)
        weighted_structural_part = float(alpha * structural_part)

    distance = float(np.sqrt(objective)) if take_sqrt else float(objective)

    info: dict[str, Any] = {
        "objective": float(objective),
        "alpha": float(alpha),
        "relation": relation,
        "lambda_geo": float(lambda_geo),
        "lambda_degree": float(lambda_degree),
        "normalize_features": bool(normalize_features),
        "normalize_relations": bool(normalize_relations),
        "feature_part": float(feature_part),
        "structural_part": float(structural_part),
        "weighted_feature_part": float(weighted_feature_part),
        "weighted_structural_part": float(weighted_structural_part),
        "feature_cost_offset": float(feature_cost_offset),
        "feature_cost_scale": float(feature_cost_scale),
        "raw_feature_cost_matrix": M_raw,
        "feature_cost_matrix": M,
        "source_relation_matrix": C1,
        "target_relation_matrix": C2,
        "source_vertex_indices": idx_a,
        "target_vertex_indices": idx_b,
        "source_weights": a,
        "target_weights": b,
        "pot_log": log,
    }
    if return_plan:
        info["transport_plan"] = np.asarray(T, dtype=np.float64)
    return distance, info


def _square_loss_structure_cost(C1: Array, C2: Array, T: Array) -> float:
    """Direct square-loss GW structural cost for a fixed transport plan."""

    # Sum_{i,k,j,l} (C1_ik - C2_jl)^2 T_ij T_kl.
    c1_sq = float(np.sum((C1 * C1) * np.outer(T.sum(axis=1), T.sum(axis=1))))
    c2_sq = float(np.sum((C2 * C2) * np.outer(T.sum(axis=0), T.sum(axis=0))))
    cross = float(np.sum(C1 * (T @ C2.T @ T.T)))
    return c1_sq + c2_sq - 2.0 * cross


def describe_pbm(pbm: ProbabilisticBallMapper) -> dict[str, Any]:
    """Small dictionary summary useful for notebooks and demos."""

    return {
        "n_points": pbm.n_points,
        "n_vertices": pbm.n_vertices,
        "n_edges": int(pbm.graph.number_of_edges()),
        "eps": pbm.eps,
        "membership": pbm.membership,
        "sigma": pbm.sigma,
        "vertex_mass_sum": float(np.sum(pbm.vertex_mass)),
    }


# User-friendly aliases for the public API. The longer names remain available
# for readers who prefer explicit mathematical terminology.
fit = fit_pbm
fit_from_landmarks = fit_pbm_from_landmarks
fgw_distance = fused_gromov_wasserstein_distance
compare = fused_gromov_wasserstein_distance
summarize = describe_pbm


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    X1 = rng.normal(size=(500, 2))
    X2 = rng.normal(loc=0.3, size=(500, 2))

    pbm_config = MembershipConfig(
        eps=0.6,
        membership="gaussian",
    )

    pbm_a = fit_pbm(X1, config=pbm_config)
    pbm_b = fit_pbm(X2, config=pbm_config)

    fgw_config = FGWConfig(
        alpha=0.5,
        relation="soft_overlap",
        lambda_geo=1.0,
        lambda_degree=0.0,
        normalize_features=True,
        normalize_relations=True,
    )

    d_fgw, info = fused_gromov_wasserstein_distance(
        pbm_a,
        pbm_b,
        config=fgw_config,
        return_plan=True,
    )

    print("PBM A:", describe_pbm(pbm_a))
    print("PBM B:", describe_pbm(pbm_b))
    print("FGW distance:", d_fgw)
    print("FGW objective:", info["objective"])
    print("Feature part:", info["feature_part"])
    print("Structure part:", info["structural_part"])
    print("Weighted feature part:", info["weighted_feature_part"])
    print("Weighted structure part:", info["weighted_structural_part"])
    print("Transport plan shape:", info["transport_plan"].shape)
