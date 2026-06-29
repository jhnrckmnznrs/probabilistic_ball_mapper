# API and documentation review

## Current public surface

Recommended beginner-facing imports:

```python
from probabilistic_ball_mapper import fit, fit_from_landmarks, fgw_distance, summarize
```

Explicit names are preserved for users who prefer full terminology:

- `fit_pbm`
- `fit_pbm_from_landmarks`
- `fused_gromov_wasserstein_distance`
- `describe_pbm`

## Release-readiness notes

Resolved for the initial GitHub release:

- Added an MIT license.
- Replaced placeholder packaging metadata with publishable metadata.
- Moved automatic landmark-selection imports behind `fit`, so membership utilities
  and fixed-landmark workflows import cleanly without `fast-ballmapper` present.
- Added tests for membership behavior and fixed-landmark PBM graph construction.
- Added CI for linting, tests, and package builds on Python 3.10--3.12.
- Added minimal MkDocs configuration and documentation index.

Recommended future improvements:

- Add a theory page explaining `eps`, membership kernels, relation matrices, and
  the interpretation of `alpha`.
- Add examples for each relation type: `soft_overlap`, `shortest_path`,
  `adjacency`, and `landmark_distance`.
- Consider adding a small `FGWResult` dataclass once the diagnostics dictionary
  stabilizes.
- Add citation metadata if this code accompanies a paper, preprint, or thesis.
