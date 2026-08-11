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
