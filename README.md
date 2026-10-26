# Probabilistic Ball Mapper

Probabilistic Ball Mapper is a small Python package built on top of
[`fast-ballmapper`](https://pypi.org/project/fast-ballmapper/). It keeps the
ordinary Ball Mapper cover construction, then adds probabilistic point-to-vertex
memberships and fused Gromov-Wasserstein (FGW) comparison using POT.

The package is intended for research, experiments, and notebooks where you want a
compact graph summary of a point cloud and a way to compare two such summaries.

## Features

- Automatic Ball Mapper landmark selection through `fast-ballmapper`.
- Fixed-landmark projection with `fit_from_landmarks`.
- Membership kernels: uniform, Gaussian, compact Gaussian, triangular,
  Epanechnikov, quartic, inverse distance, and inverse square.
- A numerically stable global Gaussian extension for validation/test points,
  kept separate from the localized cover memberships.
- Optional sample weights for nonuniform empirical probability measures.
- PBM summaries containing memberships, vertex masses, overlap matrices,
  adjacency matrices, and NetworkX graphs.
- Fused Gromov-Wasserstein comparison with fixed, dimensionless feature and
  relation scales.

## Installation

From a local checkout:

```bash
python -m pip install -e .
```

For development:

```bash
python -m pip install -e ".[dev]"
```

## Quick start

```python
import numpy as np
from probabilistic_ball_mapper import (
    FGWConfig,
    MembershipConfig,
    fit,
    fit_from_landmarks,
    fgw_distance,
)

rng = np.random.default_rng(42)
X_a = rng.normal(size=(500, 2))
X_b = rng.normal(loc=0.3, size=(500, 2))

membership = MembershipConfig(eps=0.6, membership="gaussian")
pbm_a = fit(X_a, config=membership)
pbm_b = fit(X_b, config=membership)

distance, info = fgw_distance(
    pbm_a,
    pbm_b,
    config=FGWConfig(
        alpha=0.5,
        relation="soft_overlap",
        geo_scale="epsilon",
    ),
    return_plan=True,
)

print(distance)
print(info["feature_part"], info["structural_part"])
```

For workflows where landmarks are already fixed, use `fit_from_landmarks`. This
path does not need `fast-ballmapper` at import time:

```python
landmarks = np.array([[0.0, 0.0], [1.0, 1.0]])
pbm = fit_from_landmarks(X_a, landmarks, config=MembershipConfig(eps=0.8))
```

Localized memberships use open balls and are subordinate to the cover. The
default policy raises if a point is outside all fixed balls. For an
everywhere-defined representation of validation or test points, use the
separate global Gaussian extension:

```python
from probabilistic_ball_mapper import compute_global_gaussian_membership_matrix

W_test, diagnostics = compute_global_gaussian_membership_matrix(
    X_test,
    pbm.landmark_points,
    sigma=0.4,
    cover_eps=pbm.eps,
)
outside = diagnostics["outside_cover"]
```

These global coordinates must not be used to reconstruct the PBM cover or
graph. The diagnostics retain nearest-landmark distance and cover status so
far-away points remain visible.

## FGW scaling

FGW combines node and relation costs only after making them dimensionless.
`geo_scale="epsilon"` divides landmark distances by a common Ball Mapper
radius. If graphs use different radii, provide a positive numeric `geo_scale`
fixed from training or reference data. `soft_overlap` and `adjacency` already
lie in `[0, 1]`; they use `relation_scale=1` by default. Use one fixed
`relation_scale` for unbounded relations such as shortest-path distance.

The package deliberately does not min--max normalize each compared pair or
each relation matrix independently, because doing so can erase real graph
differences. `lambda_degree` remains optional and defaults to zero because
degree is a relational descriptor rather than an intrinsic node feature.

## Public API

Recommended beginner-facing API:

```python
from probabilistic_ball_mapper import fit, fit_from_landmarks, fgw_distance, summarize
```

Explicit API preserved for technical users:

```python
from probabilistic_ball_mapper import (
    fit_pbm,
    fit_pbm_from_landmarks,
    fused_gromov_wasserstein_distance,
    describe_pbm,
)
```

## Development

```bash
python -m pytest
python -m ruff check .
python -m build
```

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE).

## Sparse PBM and spectral operators (experimental)

The existing dense API remains the default. Opt into the exact sparse path
when balls are localized and each point meets relatively few landmarks:

```python
from probabilistic_ball_mapper import (
    MembershipConfig, FGWConfig, fit, fit_from_landmarks,
    fgw_distance, spectral_operator,
)
from scipy.sparse.linalg import eigsh

cfg = MembershipConfig(eps=0.6)
summary = fit(X, config=cfg, storage="sparse")  # or fit_from_landmarks(..., storage="sparse")
L, active_indices = spectral_operator(summary)
eigenvalues, eigenvectors = eigsh(L, k=min(5, len(active_indices)-1),
                                  which="LA")  # k must be smaller than the operator dimension

d, info = fgw_distance(summary, other_summary,
    config=FGWConfig(solver="sparse_fw", relation="soft_overlap"),
    return_plan=True)
```

- Sparse CSR memberships store only entries for strict open-ball incidences.
  Geometric incidence, not numerical positive weight support, determines the
  graph. Samples outside a fixed cover still follow the configured policy.
- Vertex masses and soft overlaps are computed by exact weighted sparse
  products. Spectral operators act without explicitly forming the overlap.
- `solver="sparse_fw"` applies only to `relation="soft_overlap"`, and uses
  sparse similarity contractions and an exact line search. It **still holds
  a dense transport plan and feature cost** (quadratic in landmark count
  for two equal-size covers). Its result is a local stationary solution of a
  nonconvex problem; compare convergence and objective against POT.
- The sparse solver reports `iterations`, `converged`, `final_fw_gap`,
  `source_similarity_matrix`, and `target_similarity_matrix`, instead of
  dense relation matrices. Fixed `geo_scale` and `relation_scale` are honored.
- Existing POT FGW works on dense summaries. Convert explicitly via
  `sparse_summary.to_dense()` if needed; this can allocate large arrays.

For careful reproducibility and numerical comparisons, keep the `storage="dense"`
baseline. Sparse backends are opt-in pending wider cross-dataset benchmarks.

## Experimental FGW optimization controls

For `relation="soft_overlap"`, PBM similarities are positive semidefinite,
making the squared-loss FGW objective **concave** in the transport plan.
Consequently the Frank–Wolfe linear transport oracle can take full steps
without the general quadratic line search. This is a **local optimization
method**, not a proof that the returned coupling is globally optimal.

```python
cfg = FGWConfig(
    solver="sparse_fw",
    relation="soft_overlap",
    n_restarts=4,        # product coupling + 3 random LP vertices
    random_state=7,      # reproducible multi-start initialization
    pivot_steps=2,       # optional mass-preserving 2x2 cycle exchanges
    pivot_candidates=512,
    time_budget=None,    # optional soft seconds limit across restarts
)
distance, diagnostics = fgw_distance(pbm_a, pbm_b, config=cfg, return_plan=True)
print(diagnostics["restarts_completed"], diagnostics["restart_history"])
print(diagnostics["oracle_seconds"], diagnostics["contraction_seconds"])
print(diagnostics["final_fw_gap"])  # stationarity, NOT global optimality
```

With `n_restarts=1` and `pivot_steps=0` (both defaults), sparse
FGW uses one product initialization. The POT solver remains the default.
The transport plan and cross-node feature costs remain dense.

A complete derivation, limitations, exhaustive tiny-case reference solver,
and numerical experiment protocol appear in
[Concave squared-loss FGW optimization](docs/fgw_optimization.md).

To run the same-initialization comparison with POT, including equal-restart
and equal-wall-clock comparisons plus oracle/contraction profiling:

```bash
python benchmarks/bench_fgw_optimization.py --n 250 --landmarks 12 --restarts 4
```

The matched-time budget is soft: calls to EMD cannot be interrupted mid-solve.
Synthetic speedups are not a substitute for measurements on real datasets.
