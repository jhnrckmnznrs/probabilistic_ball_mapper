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
- PBM summaries containing memberships, vertex masses, overlap matrices,
  adjacency matrices, and NetworkX graphs.
- Fused Gromov-Wasserstein comparison using landmark features and graph or
  overlap structure.

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
    config=FGWConfig(alpha=0.5, relation="soft_overlap"),
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

The GitHub Actions workflow runs linting, tests, and a package build on pushes
and pull requests.

## Documentation

API notes live in [`docs/api.md`](docs/api.md). A minimal MkDocs configuration is
included so the documentation can be expanded later.

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE).
