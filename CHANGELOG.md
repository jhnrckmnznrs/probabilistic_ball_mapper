# Changelog

## 0.2.0

- Makes localized memberships subordinate to strict open epsilon-ball covers by
  default and separates explicit nearest-neighbor extrapolation.
- Adds a stable global Gaussian landmark extension with out-of-cover
  diagnostics for validation and test data.
- Stabilizes Gaussian and compact-Gaussian normalization for very small
  bandwidths.
- Builds fixed-landmark covers and graphs from geometric incidence rather than
  extrapolated membership support.
- Supports nonuniform empirical measures through `sample_weight`.
- Replaces pair-dependent FGW normalization with fixed `geo_scale` and
  `relation_scale` parameters.
- Removes the overlap-spread node descriptor; normalized degree remains
  optional with zero default weight.
- Adds validation and regression tests for memberships, summaries, and FGW.

## 0.1.0

- Initial public release scaffold.
- Adds probabilistic memberships for Ball Mapper covers.
- Adds static PBM summaries with vertex masses, overlap matrices, and graphs.
- Adds fused Gromov-Wasserstein comparison helpers using POT.
