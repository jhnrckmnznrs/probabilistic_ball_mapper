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
GeometricScale = float | Literal["epsilon"]

_VALID_MEMBERSHIPS = frozenset(
    {
        "uniform",
        "gaussian",
        "compact_gaussian",
        "triangular",
        "epanechnikov",
        "quartic",
        "inverse_distance",
        "inverse_square",
    }
)
_VALID_RELATIONS = (
    "soft_overlap",
    "shortest_path",
    "adjacency",
    "landmark_distance",
)


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
        ``"error"`` preserves subordination by rejecting points outside every
        epsilon ball. ``"nearest"`` is an explicit extrapolation fallback; it
        keeps rows stochastic but its fallback memberships are not subordinate
        to the cover. For an everywhere-defined smooth extension, use
        :func:`compute_global_gaussian_membership_matrix` separately.
    """

    eps: float
    membership: Membership = "gaussian"
    sigma: float | None = None
    metric: str = "euclidean"
    metric_kwargs: Mapping[str, Any] | None = None
    taper_power: float = 2.0
    inverse_offset: float = 1e-12
    out_of_cover_policy: OutOfCoverPolicy = "error"

    def __post_init__(self) -> None:
        if not np.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("eps must be finite and positive.")
        if self.membership not in _VALID_MEMBERSHIPS:
            raise ValueError(
                "membership must be one of "
                + ", ".join(sorted(_VALID_MEMBERSHIPS))
                + "."
            )
        if self.sigma is not None and (not np.isfinite(self.sigma) or self.sigma <= 0):
            raise ValueError("sigma must be finite and positive when provided.")
        if not np.isfinite(self.taper_power) or self.taper_power < 0:
            raise ValueError("taper_power must be finite and nonnegative.")
        if not np.isfinite(self.inverse_offset) or self.inverse_offset <= 0:
            raise ValueError("inverse_offset must be finite and positive.")
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
    sample_weight: Array

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

    @property
    def outside_cover_indices(self) -> np.ndarray:
        """Indices of data points outside every geometric epsilon ball."""

        covered = np.zeros(self.n_points, dtype=bool)
        for members in self.cover:
            covered[np.asarray(members, dtype=np.int64)] = True
        return np.flatnonzero(~covered).astype(np.int64)


def _as_2d_float(X: NDArray[Any], name: str = "X") -> Array:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array.")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one nonempty point.")
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

        # Work with relative log-weights. Direct exponentiation can underflow
        # every eligible Gaussian weight when sigma is small.
        log_weights = np.full(D.shape, -np.inf, dtype=np.float64)
        covered = np.any(inside, axis=1)
        if np.any(covered):
            eligible = np.where(inside[covered], D[covered], np.inf)
            row_min = np.min(eligible, axis=1, keepdims=True)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                delta_squared = (eligible - row_min) * (eligible + row_min)
                gaussian_log = -delta_squared / (2.0 * sigma * sigma)

            tied_minimum = eligible == row_min
            gaussian_log[tied_minimum] = 0.0
            gaussian_log[~np.isfinite(gaussian_log)] = -np.inf

            if membership == "compact_gaussian" and taper_power > 0:
                taper = np.clip(1.0 - (eligible / eps) ** 2, 0.0, 1.0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    gaussian_log += taper_power * np.log(taper)

            gaussian_log[~inside[covered]] = -np.inf
            row_max = np.max(gaussian_log, axis=1, keepdims=True)
            if not np.all(np.isfinite(row_max)):
                raise FloatingPointError(
                    "Gaussian membership scores are numerically degenerate. "
                    "Use a less extreme sigma or taper_power."
                )
            log_weights[covered] = gaussian_log - row_max

        weights = np.zeros_like(D, dtype=np.float64)
        finite = np.isfinite(log_weights)
        weights[finite] = np.exp(log_weights[finite])
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
    """Compute localized memberships supported on open epsilon balls.

    The second return value contains the indices of points outside every ball.
    The default ``out_of_cover_policy="error"`` raises when this set is not
    empty. The explicit ``"nearest"`` fallback keeps rows stochastic, but its
    fallback entries are not subordinate to the cover.
    """

    X = _as_2d_float(X, "X")
    C = _as_2d_float(landmark_points, "landmark_points")
    D = distance_to_landmarks(
        X,
        C,
        metric=config.metric,
        metric_kwargs=config.metric_kwargs,
    )
    W, uncovered, _ = _membership_from_distances(D, config)
    return W, uncovered


def _membership_from_distances(
    D: Array,
    config: MembershipConfig,
) -> tuple[Array, np.ndarray, np.ndarray]:
    """Compute localized memberships from a precomputed distance matrix."""

    inside = D < config.eps

    weights = _radial_membership_weights(
        D=D,
        inside=inside,
        eps=config.eps,
        membership=config.membership,
        sigma=config.effective_sigma,
        taper_power=config.taper_power,
        inverse_offset=config.inverse_offset,
    )

    row_sums = weights.sum(axis=1)
    uncovered = np.flatnonzero(~np.any(inside, axis=1))
    covered_zero = np.flatnonzero(np.any(inside, axis=1) & (row_sums <= 0.0))
    if covered_zero.size:
        raise FloatingPointError(
            "A localized membership row underflowed despite having eligible landmarks."
        )

    if uncovered.size:
        if config.out_of_cover_policy == "error":
            raise RuntimeError(
                f"{uncovered.size} points are outside the fixed cover. "
                f"First indices: {uncovered[:10].tolist()}"
            )
        if config.out_of_cover_policy != "nearest":
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

    return W.astype(np.float64), uncovered.astype(np.int64), inside


def compute_global_gaussian_membership_matrix(
    X: NDArray[Any],
    landmark_points: NDArray[Any],
    sigma: float,
    *,
    metric: str = "euclidean",
    metric_kwargs: Mapping[str, Any] | None = None,
    cover_eps: float | None = None,
) -> tuple[Array, dict[str, NDArray[Any]]]:
    """Compute an everywhere-defined Gaussian landmark representation.

    This extension assigns mass relative to all landmarks, so it is not
    subordinate to the Ball Mapper cover. It is intended for projecting
    validation or test points onto fixed training landmarks, not for building
    the PBM cover or graph.

    The diagnostics contain ``min_landmark_distance``. If ``cover_eps`` is
    supplied, they also contain the Boolean ``outside_cover`` mask for the
    open-ball convention ``distance < cover_eps``.
    """

    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive.")
    if cover_eps is not None and (not np.isfinite(cover_eps) or cover_eps <= 0):
        raise ValueError("cover_eps must be finite and positive when provided.")

    X_arr = _as_2d_float(X, "X")
    C = _as_2d_float(landmark_points, "landmark_points")
    D = distance_to_landmarks(
        X_arr,
        C,
        metric=metric,
        metric_kwargs=metric_kwargs,
    )
    weights = _radial_membership_weights(
        D=D,
        inside=np.ones(D.shape, dtype=bool),
        eps=1.0,
        membership="gaussian",
        sigma=float(sigma),
        taper_power=0.0,
        inverse_offset=1.0,
    )
    W = weights / weights.sum(axis=1, keepdims=True)
    nearest_distance = np.min(D, axis=1)
    diagnostics: dict[str, NDArray[Any]] = {
        "min_landmark_distance": nearest_distance.astype(np.float64)
    }
    if cover_eps is not None:
        diagnostics["outside_cover"] = nearest_distance >= cover_eps

    if not np.all(np.isfinite(W)) or not np.allclose(W.sum(axis=1), 1.0, atol=1e-12):
        raise FloatingPointError("Global Gaussian normalization failed.")
    return W.astype(np.float64), diagnostics


def cover_from_membership(W: NDArray[Any], tol: float = 0.0) -> list[np.ndarray]:
    """Build cover index lists from positive memberships."""

    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 2:
        raise ValueError("W must be a two-dimensional array.")
    if not np.all(np.isfinite(W)):
        raise ValueError("W contains non-finite values.")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative.")
    return [np.flatnonzero(W[:, j] > tol).astype(np.int64) for j in range(W.shape[1])]


def _cover_from_inside(inside: np.ndarray) -> list[np.ndarray]:
    """Build the geometric open-ball cover from a Boolean incidence matrix."""

    return [
        np.flatnonzero(inside[:, j]).astype(np.int64) for j in range(inside.shape[1])
    ]


def _probability_weights(
    sample_weight: NDArray[Any] | None,
    n_points: int,
) -> Array:
    """Validate and normalize empirical sample weights."""

    if n_points <= 0:
        raise ValueError("At least one sample is required.")
    if sample_weight is None:
        return np.full(n_points, 1.0 / n_points, dtype=np.float64)

    weights = np.asarray(sample_weight, dtype=np.float64)
    if weights.ndim != 1 or weights.shape[0] != n_points:
        raise ValueError("sample_weight must have shape (n_points,).")
    if not np.all(np.isfinite(weights)):
        raise ValueError("sample_weight contains non-finite values.")
    if np.any(weights < 0):
        raise ValueError("sample_weight must be nonnegative.")
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("sample_weight must have finite positive total mass.")
    return (weights / total).astype(np.float64)


def summarize_static_pbm(
    W: NDArray[Any],
    graph: nx.Graph,
    landmarks: Sequence[int] | None,
    landmark_points: NDArray[Any],
    cover: Sequence[np.ndarray] | None,
    config: MembershipConfig,
    sample_weight: NDArray[Any] | None = None,
) -> ProbabilisticBallMapper:
    """Construct summary matrices from an already-computed membership matrix."""

    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 2 or W.shape[0] == 0 or W.shape[1] == 0:
        raise ValueError("W must be a two-dimensional membership matrix.")
    if not np.all(np.isfinite(W)):
        raise ValueError("W contains non-finite values.")
    if np.any(W < 0.0) or np.any(W > 1.0):
        raise ValueError("Entries of W must lie in [0, 1].")
    if not np.allclose(W.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("Rows of W must sum to one.")

    n_points, n_vertices = W.shape
    probability_weight = _probability_weights(sample_weight, n_points)
    landmark_array = _as_2d_float(landmark_points, "landmark_points")
    if landmark_array.shape[0] != n_vertices:
        raise ValueError("landmark_points must have one row per column of W.")

    if not isinstance(graph, nx.Graph):
        raise TypeError("graph must be a NetworkX graph.")
    if graph.is_directed():
        raise ValueError("graph must be undirected.")
    if graph.is_multigraph():
        raise ValueError("graph must not contain parallel edges.")
    if nx.number_of_selfloops(graph):
        raise ValueError("graph must not contain self-loops.")
    expected_nodes = set(range(n_vertices))
    unknown_nodes = set(graph.nodes) - expected_nodes
    if unknown_nodes:
        raise ValueError(
            "graph contains nodes outside the vertex range: "
            f"{sorted(unknown_nodes, key=str)[:10]}"
        )

    if landmarks is not None and len(landmarks) != n_vertices:
        raise ValueError("landmarks must contain one index per column of W.")

    if cover is None:
        validated_cover = cover_from_membership(W)
    else:
        if len(cover) != n_vertices:
            raise ValueError("cover must contain one member list per column of W.")
        validated_cover = []
        for members in cover:
            member_array = np.asarray(members)
            if member_array.ndim != 1:
                raise ValueError("Each cover member list must be one-dimensional.")
            if not np.issubdtype(member_array.dtype, np.integer):
                raise ValueError("Cover indices must be integers.")
            member_array = member_array.astype(np.int64, copy=False)
            if np.any(member_array < 0) or np.any(member_array >= n_points):
                raise ValueError("Cover indices must refer to rows of W.")
            validated_cover.append(np.unique(member_array))

    vertex_mass = W.T @ probability_weight
    overlap_matrix = W.T @ (probability_weight[:, None] * W)

    graph = graph.copy()
    graph.add_nodes_from(range(n_vertices))
    adjacency_matrix = nx.to_numpy_array(
        graph,
        nodelist=list(range(n_vertices)),
        dtype=float,
        weight=None,
    )

    return ProbabilisticBallMapper(
        landmarks=None if landmarks is None else [int(i) for i in landmarks],
        landmark_points=landmark_array,
        cover=validated_cover,
        graph=graph,
        W=W,
        vertex_mass=vertex_mass,
        overlap_matrix=overlap_matrix,
        adjacency_matrix=adjacency_matrix,
        sample_weight=probability_weight,
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
    sample_weight: NDArray[Any] | None = None,
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
    sample_weight:
        Optional nonnegative weights defining the empirical probability
        measure. They are normalized to sum to one.

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

    # fast-ballmapper currently queries closed radius balls. Moving to the
    # immediately smaller representable radius implements this package's open
    # convention d(x, c) < eps, including in FPS stopping at exact boundaries.
    strict_search_eps = float(np.nextafter(config.eps, -np.inf))
    if strict_search_eps <= 0:
        raise ValueError("eps is too small for strict-radius landmark search.")

    if landmark_config.selection == "fps":
        landmarks, _ = compute_landmarks_fps(
            X,
            eps=strict_search_eps,
            start_index=landmark_config.start_index,
            method=landmark_config.method,
            metric=config.metric,
            leaf_size=landmark_config.leaf_size,
            metric_kwargs=config.metric_kwargs,
        )

    elif landmark_config.selection == "random":
        landmarks, _ = compute_landmarks(
            X,
            eps=strict_search_eps,
            method=landmark_config.method,
            metric=config.metric,
            leaf_size=landmark_config.leaf_size,
            metric_kwargs=config.metric_kwargs,
        )

    else:
        raise ValueError("landmark_config.selection must be either 'fps' or 'random'.")

    landmark_indices = np.asarray(landmarks, dtype=np.int64)
    if landmark_indices.ndim != 1 or landmark_indices.size == 0:
        raise RuntimeError("fast-ballmapper returned no valid landmarks.")
    if np.any(landmark_indices < 0) or np.any(landmark_indices >= X.shape[0]):
        raise RuntimeError("fast-ballmapper returned an invalid landmark index.")
    if np.unique(landmark_indices).size != landmark_indices.size:
        raise RuntimeError("fast-ballmapper returned duplicate landmarks.")

    landmark_points = X[landmark_indices]
    D = distance_to_landmarks(
        X,
        landmark_points,
        metric=config.metric,
        metric_kwargs=config.metric_kwargs,
    )
    W, _, inside = _membership_from_distances(D, config)
    cover = _cover_from_inside(inside)
    graph = _build_mapper_graph(cover)

    return summarize_static_pbm(
        W=W,
        graph=graph,
        landmarks=landmarks,
        landmark_points=landmark_points,
        cover=cover,
        config=config,
        sample_weight=sample_weight,
    )


def fit_pbm_from_landmarks(
    X: NDArray[Any],
    landmark_points: NDArray[Any],
    config: MembershipConfig,
    sample_weight: NDArray[Any] | None = None,
) -> ProbabilisticBallMapper:
    """Construct a PBM using an already fixed landmark cover.

    The graph and cover always use strict geometric incidence ``d < eps``.
    They are never inferred from extrapolated membership entries.
    """

    X = _as_2d_float(X, "X")
    landmark_points = _as_2d_float(landmark_points, "landmark_points")

    D = distance_to_landmarks(
        X,
        landmark_points,
        metric=config.metric,
        metric_kwargs=config.metric_kwargs,
    )
    W, _, inside = _membership_from_distances(D, config)
    cover = _cover_from_inside(inside)
    graph = _build_mapper_graph(cover)

    return summarize_static_pbm(
        W=W,
        graph=graph,
        landmarks=None,
        landmark_points=landmark_points,
        cover=cover,
        config=config,
        sample_weight=sample_weight,
    )


def normalized_degrees(pbm: ProbabilisticBallMapper) -> Array:
    """Normalized graph degrees in [0, 1]."""

    A = np.asarray(pbm.adjacency_matrix, dtype=np.float64)
    n = A.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=np.float64)
    return A.sum(axis=1) / float(n - 1)


def vertex_feature_cost_matrix(
    pbm_a: ProbabilisticBallMapper,
    pbm_b: ProbabilisticBallMapper,
    p: float = 2.0,
    lambda_geo: float = 1.0,
    lambda_degree: float = 0.0,
    geo_scale: float = 1.0,
    metric: str | None = None,
    metric_kwargs: Mapping[str, Any] | None = None,
) -> Array:
    """Dimensionless ground-cost matrix between PBM vertices.

    Landmark distances are divided by the fixed ``geo_scale`` before taking
    the ``p``-th power. Choose this scale independently of the graph pair, for
    example as a common Ball Mapper epsilon or a training-set scale.
    """

    if not np.isfinite(p) or p <= 0:
        raise ValueError("p must be finite and positive.")
    if not np.isfinite(lambda_geo) or not np.isfinite(lambda_degree):
        raise ValueError("lambda weights must be finite.")
    if lambda_geo < 0 or lambda_degree < 0:
        raise ValueError("lambda weights must be nonnegative.")
    if lambda_geo == 0 and lambda_degree == 0:
        raise ValueError("At least one lambda weight must be positive.")
    if not np.isfinite(geo_scale) or geo_scale <= 0:
        raise ValueError("geo_scale must be finite and positive.")

    m, n = pbm_a.n_vertices, pbm_b.n_vertices
    C = np.zeros((m, n), dtype=np.float64)

    if lambda_geo > 0:
        if metric is None:
            if pbm_a.metric != pbm_b.metric:
                raise ValueError(
                    "PBMs use different metrics; pass an explicit comparison metric."
                )
            kwargs_a = dict(pbm_a.metric_kwargs or {})
            kwargs_b = dict(pbm_b.metric_kwargs or {})
            if kwargs_a != kwargs_b:
                raise ValueError(
                    "PBMs use different metric_kwargs; pass explicit metric_kwargs."
                )
            ground_metric = pbm_a.metric
            ground_kwargs = kwargs_a
        else:
            ground_metric = metric
            ground_kwargs = dict(metric_kwargs or {})

        C += (
            lambda_geo
            * (
                pairwise_distances(
                    pbm_a.landmark_points,
                    pbm_b.landmark_points,
                    metric=ground_metric,
                    **ground_kwargs,
                )
                / geo_scale
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


def relation_matrix(
    pbm: ProbabilisticBallMapper,
    relation: Relation = "soft_overlap",
    *,
    scale: float = 1.0,
    metric: str | None = None,
    metric_kwargs: Mapping[str, Any] | None = None,
    disconnected_distance: float | None = None,
) -> Array:
    """Dimensionless within-graph dissimilarity matrix for FGW.

    The raw relation is divided by a fixed positive ``scale``. It is never
    divided by its own maximum, because independent graph-wise normalization
    erases meaningful differences in overlap magnitude.

    relation="soft_overlap"
        Uses 1 - Q_ij / sqrt(nu_i nu_j), with diagonal set to 0.
        This emphasizes probabilistic shared membership.
    relation="shortest_path"
        Uses graph shortest-path distance; disconnected pairs receive one more
        than the largest finite distance unless ``disconnected_distance`` is
        explicitly supplied.
    relation="adjacency"
        Uses 0 for equal/adjacent vertices and 1 for non-adjacent vertices.
    relation="landmark_distance"
        Uses metric distances between landmark coordinates.
    """

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive.")
    if disconnected_distance is not None and (
        not np.isfinite(disconnected_distance) or disconnected_distance <= 0
    ):
        raise ValueError(
            "disconnected_distance must be finite and positive when provided."
        )

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
        if np.any(~finite):
            if disconnected_distance is None:
                fill_distance = max_finite + 1.0
            else:
                fill_distance = float(disconnected_distance)
                if fill_distance <= max_finite:
                    raise ValueError(
                        "disconnected_distance must exceed every finite shortest path."
                    )
            C[~finite] = fill_distance

    elif relation == "adjacency":
        A = np.asarray(pbm.adjacency_matrix, dtype=np.float64)
        C = 1.0 - np.clip(A, 0.0, 1.0)
        np.fill_diagonal(C, 0.0)

    else:  # relation == "landmark_distance"
        landmark_metric_kwargs = (
            dict(metric_kwargs)
            if metric_kwargs is not None
            else dict(pbm.metric_kwargs or {})
        )
        C = pairwise_distances(
            pbm.landmark_points,
            pbm.landmark_points,
            metric=metric or pbm.metric,
            **landmark_metric_kwargs,
        )
        C = np.asarray(C, dtype=np.float64)
        np.fill_diagonal(C, 0.0)

    C = C / scale

    if not np.all(np.isfinite(C)):
        raise ValueError("The relation matrix contains non-finite values.")
    return C.astype(np.float64)


def _positive_mass_indices(mass: NDArray[Any], tol: float) -> np.ndarray:
    mass = np.asarray(mass, dtype=np.float64)
    if mass.ndim != 1 or not np.all(np.isfinite(mass)) or np.any(mass < 0):
        raise ValueError("Vertex masses must be a finite nonnegative vector.")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("mass_tol must be finite and nonnegative.")
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
        Optional weight for normalized graph-degree cost. The default is zero
        because degree is relational rather than an intrinsic node feature.
    geo_scale:
        Fixed scale for landmark distances. ``"epsilon"`` uses the common
        epsilon and therefore requires both PBMs to have exactly the same
        epsilon. A numeric value should be fixed from training/reference data,
        not estimated separately for each compared pair.
    relation_scale:
        Fixed scale for relation dissimilarities. By default it is one for
        soft overlap, adjacency, and shortest paths, and ``geo_scale`` for
        landmark-distance relations.
    disconnected_distance:
        Optional fixed raw distance assigned to disconnected shortest-path
        pairs. It must exceed all finite shortest-path distances.
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
    geo_scale: GeometricScale = "epsilon"
    relation_scale: float | None = None
    disconnected_distance: float | None = None
    mass_tol: float = 0.0
    take_sqrt: bool = True
    max_iter: int = 1000
    tol_rel: float = 1e-9
    tol_abs: float = 1e-9
    verbose: bool = False

    def __post_init__(self) -> None:
        if not np.isfinite(self.alpha) or not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must satisfy 0 < alpha < 1.")
        if self.relation not in _VALID_RELATIONS:
            raise ValueError(
                "relation must be one of " + ", ".join(_VALID_RELATIONS) + "."
            )
        if not np.isfinite(self.lambda_geo) or not np.isfinite(self.lambda_degree):
            raise ValueError("lambda weights must be finite.")
        if self.lambda_geo < 0 or self.lambda_degree < 0:
            raise ValueError("lambda weights must be nonnegative.")
        if self.lambda_geo == 0 and self.lambda_degree == 0:
            raise ValueError("At least one lambda weight must be positive.")
        if isinstance(self.geo_scale, str):
            if self.geo_scale != "epsilon":
                raise ValueError("geo_scale must be positive or 'epsilon'.")
        elif not np.isfinite(self.geo_scale) or self.geo_scale <= 0:
            raise ValueError("geo_scale must be finite and positive.")
        if self.relation_scale is not None and (
            not np.isfinite(self.relation_scale) or self.relation_scale <= 0
        ):
            raise ValueError("relation_scale must be finite and positive.")
        if self.disconnected_distance is not None and (
            not np.isfinite(self.disconnected_distance)
            or self.disconnected_distance <= 0
        ):
            raise ValueError("disconnected_distance must be finite and positive.")
        if not np.isfinite(self.mass_tol) or self.mass_tol < 0:
            raise ValueError("mass_tol must be finite and nonnegative.")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if (
            not np.isfinite(self.tol_rel)
            or not np.isfinite(self.tol_abs)
            or self.tol_rel < 0
            or self.tol_abs < 0
        ):
            raise ValueError("solver tolerances must be finite and nonnegative.")


def _resolve_geo_scale(
    pbm_a: ProbabilisticBallMapper,
    pbm_b: ProbabilisticBallMapper,
    scale: GeometricScale,
) -> float:
    """Resolve the fixed geometric scale used by one FGW configuration."""

    if scale == "epsilon":
        if pbm_a.eps != pbm_b.eps:
            raise ValueError(
                "geo_scale='epsilon' requires PBMs with the same epsilon. "
                "Pass a fixed positive geo_scale to compare different epsilons."
            )
        return float(pbm_a.eps)
    return float(scale)


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

    The feature and structural costs are made dimensionless using fixed scales
    from ``config``. No cost matrix is normalized using statistics of the
    particular graph pair. ``return_plan=True`` includes the optimal transport
    plan in ``info["transport_plan"]``.

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
    mass_tol = config.mass_tol
    take_sqrt = config.take_sqrt
    max_iter = config.max_iter
    tol_rel = config.tol_rel
    tol_abs = config.tol_abs
    verbose = config.verbose

    needs_geo_scale = lambda_geo > 0 or relation == "landmark_distance"
    if needs_geo_scale:
        geo_scale = _resolve_geo_scale(pbm_a, pbm_b, config.geo_scale)
    elif isinstance(config.geo_scale, str):
        geo_scale = 1.0
    else:
        geo_scale = float(config.geo_scale)

    if config.relation_scale is None:
        relation_scale = geo_scale if relation == "landmark_distance" else 1.0
    else:
        relation_scale = float(config.relation_scale)

    relation_metric = metric
    relation_metric_kwargs = metric_kwargs
    if relation == "landmark_distance" and relation_metric is None:
        if pbm_a.metric != pbm_b.metric:
            raise ValueError(
                "PBMs use different metrics; pass an explicit comparison metric."
            )
        kwargs_a = dict(pbm_a.metric_kwargs or {})
        kwargs_b = dict(pbm_b.metric_kwargs or {})
        if kwargs_a != kwargs_b:
            raise ValueError(
                "PBMs use different metric_kwargs; pass explicit metric_kwargs."
            )
        relation_metric = pbm_a.metric
        relation_metric_kwargs = kwargs_a

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
        geo_scale=geo_scale,
        metric=metric,
        metric_kwargs=metric_kwargs,
    )
    M = np.asarray(M_full[np.ix_(idx_a, idx_b)], dtype=np.float64)

    C1_full = relation_matrix(
        pbm_a,
        relation=relation,
        scale=relation_scale,
        metric=relation_metric,
        metric_kwargs=relation_metric_kwargs,
        disconnected_distance=config.disconnected_distance,
    )
    C2_full = relation_matrix(
        pbm_b,
        relation=relation,
        scale=relation_scale,
        metric=relation_metric,
        metric_kwargs=relation_metric_kwargs,
        disconnected_distance=config.disconnected_distance,
    )
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
        structural_part = max(_square_loss_structure_cost(C1, C2, T_arr), 0.0)
        weighted_feature_part = float((1.0 - alpha) * feature_part)
        weighted_structural_part = float(alpha * structural_part)

    distance = float(np.sqrt(objective)) if take_sqrt else float(objective)

    log_for_info = dict(log) if isinstance(log, Mapping) else {}
    if not return_plan:
        log_for_info.pop("T", None)

    info: dict[str, Any] = {
        "objective": float(objective),
        "alpha": float(alpha),
        "relation": relation,
        "lambda_geo": float(lambda_geo),
        "lambda_degree": float(lambda_degree),
        "geo_scale": float(geo_scale),
        "relation_scale": float(relation_scale),
        "feature_part": float(feature_part),
        "structural_part": float(structural_part),
        "weighted_feature_part": float(weighted_feature_part),
        "weighted_structural_part": float(weighted_structural_part),
        "feature_cost_matrix": M,
        "source_relation_matrix": C1,
        "target_relation_matrix": C2,
        "source_vertex_indices": idx_a,
        "target_vertex_indices": idx_b,
        "source_weights": a,
        "target_weights": b,
        "pot_log": log_for_info,
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
        "n_outside_cover": int(pbm.outside_cover_indices.size),
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
        geo_scale="epsilon",
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
