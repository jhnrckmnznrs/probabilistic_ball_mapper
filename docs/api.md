# API notes

## Recommended names

Use the short aliases in tutorials and README examples:

- `fit` for constructing a PBM from data.
- `fit_from_landmarks` for projecting data onto fixed landmarks.
- `fgw_distance` for comparing two PBM objects.
- `summarize` for a compact dictionary summary.

Keep the longer names as stable aliases for mathematical clarity and backward
compatibility:

- `fit_pbm`
- `fit_pbm_from_landmarks`
- `fused_gromov_wasserstein_distance`
- `describe_pbm`

## Documentation gaps to fill next

- Add a small theory page explaining PBM, membership kernels, and FGW.
- Add parameter tables for `MembershipConfig`, `LandmarkConfig`, and `FGWConfig`.
- Add examples for each relation type: `soft_overlap`, `shortest_path`,
  `adjacency`, and `landmark_distance`.
- Clarify the expected scale and interpretation of `alpha` when normalization is
  on or off.
