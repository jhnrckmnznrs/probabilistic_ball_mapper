# Changelog

## Unreleased — FGW optimization experiments

- Exploit positive semidefiniteness of PBM soft-overlap similarities:
  squared-loss FGW is concave on the transport polytope, so full
  Frank–Wolfe steps are nonincreasing and need no quadratic line search.
- Add reproducible random-vertex multi-start with best-objective selection,
  optional sparse 2x2 vertex exchanges, and soft time budgets.
- Report per-start convergence gaps and oracle/contraction timing.
- Add exhaustive small-polytope numerical reference and exact rational
  2x2 global-optimality certificates.
- Add matched-POT benchmarks (same initial plans, equal-restart and
  equal-time protocols) and mathematical explanation in documentation.
- The POT default is unchanged. Global FGW optimization remains nonconvex.

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
