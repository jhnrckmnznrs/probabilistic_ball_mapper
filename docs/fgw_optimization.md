# Concave squared-loss PBM-FGW: theory and implementation

This note describes the **soft-overlap** relation only. It does not apply to
arbitrary GW relation matrices, entropically regularized objectives, or
non-squared losses. The theorem assumes genuine nonnegative probability
memberships with each row summing to one.

## Definition 1. Localized probabilistic cover

Let `p_x >= 0` with `sum_x p_x=1`, and let `W_{xi} >= 0` satisfy
`sum_i W_{xi}=1`. Define vertex masses and weighted overlaps

\[
\nu_i=\sum_x p_x W_{xi},
\qquad Q_{ij}=\sum_xp_x W_{xi}W_{xj}.
\]

Restrict all matrices to vertices with `nu_i>0`. Set

\[
B=D_\nu^{-1/2}QD_\nu^{-1/2},\qquad
K=B+\operatorname{diag}(1-B_{11},\ldots,1-B_{mm}).
\]

Then `K_ii=1` and, for `i!=j`,
`K_ij=Q_ij/sqrt(nu_i nu_j)`. The repository's **soft-overlap**
dissimilarity is `C_ij = 1-K_ij`, including `C_ii=0`.

## Proposition 1. Positive semidefiniteness

**Claim.** `K\succeq0`.

**Proof.** Put `A=diag(sqrt(p))W D_nu^{-1/2}`. Then
`B=A^TA\succeq0`. Also, since `0<=W_{xi}<=1`,

\[
Q_{ii}=\sum_xp_xW_{xi}^2\le\sum_xp_xW_{xi}=\nu_i.
\]

Hence `1-B_ii>=0` and the diagonal correction is positive
semidefinite. Their sum `K` is positive semidefinite. \(\square\)

## Definition 2. Squared-loss fused GW

For two covers, let `a,b` denote strictly positive normalized vertex
masses, and let

\[
\Pi(a,b)=\{T\in\mathbb R_{\ge0}^{m\times n}:\;
 T\mathbf1_n=a,\quad T^\top\mathbf1_m=b\}.
\]

Thus every feasible `m x n` transport plan has row masses `a` and column masses `b`.
With fixed relation scale `s>0`, fixed cross-node cost `M`, and
`0<alpha<1`, set

\[
F(T)=(1-\alpha)\langle M,T\rangle+
\frac{\alpha}{s^2}\sum_{i,k,j,l}
(K^{(1)}_{ik}-K^{(2)}_{jl})^2 T_{ij}T_{kl}.
\]

This equals squared-loss FGW with `C_1=1-K_1`,
`C_2=1-K_2` because pairwise differences change only by sign.
Since the marginals are fixed, write

\[
c_1=\sum_{i,k}K^{(1)2}_{ik}a_i a_k,\quad
c_2=\sum_{j,l}K^{(2)2}_{jl}b_j b_l.
\]

Then

\[
F(T)=(1-\alpha)\langle M,T\rangle+
\frac{\alpha}{s^2}
\left(c_1+c_2-2\operatorname{tr}
(T^TK_1TK_2)\right).
\]

## Theorem 1. Concavity and full-step Frank–Wolfe

**Claim.** `F` is concave on `Pi(a,b)`. If
`S in argmin_{U in Pi(a,b)} <grad F(T),U>`, then
`F(S)<=F(T)`; an exact line search over `T+t(S-T)`
always admits a minimizer at `t=1` (possibly also at other t).

**Proof.** By Proposition 1, both `K_1,K_2` are positive
semidefinite. Thus

\[
\operatorname{tr}(T^TK_1TK_2)
=\|K_1^{1/2}TK_2^{1/2}\|_F^2
\]

is convex. Its negative is concave, and the other term of `F`
is affine. For `D=S-T`, the exact expansion is

\[
F(T+tD)-F(T)
=t\langle\nabla F(T),D\rangle
-\frac{2\alpha t^2}{s^2}
\|K_1^{1/2}DK_2^{1/2}\|_F^2.
\]

The linear coefficient is nonpositive by the linear OT oracle.
Consequently `F(T+D)<=F(T)`, and concavity implies that a
minimum on `[0,1]` is attained at an endpoint; the
minimizing oracle chooses the nonworse full step. \(\square\)

**Important qualification.** A zero Frank–Wolfe gap establishes
first-order stationarity but does *not* establish global optimality
of this nonconvex minimization problem. Multiple starts or
transport-vertex exchanges can find lower objective values.

## Corollary 1. Existence of a sparse global minimizer

A continuous concave function on a compact polytope attains its
minimum at some extreme point. A basic feasible transportation
polytope vertex has at most `m+n-1` nonzero entries.

**Proof.** Write any minimizer as a convex combination of extreme
points. By concavity, at least one extreme point has objective
no greater than that minimizer. Basic feasible transports use
at most `m+n-1` basic arcs because one of the `m+n` equality
constraints is redundant. \(\square\)

This is an *existence theorem*; the transport plan is still stored
densely in the current numerical implementation.

## Lemma 1. Exact 2x2 cycle exchange

For distinct rows `i,k` and columns `j,l`, set
`D=(e_i-e_k)(e_j-e_l)^T`. Both marginals of `D` vanish.
If `T_ij` and `T_kl` are positive, then
`t=-min(T_ij,T_kl)` gives a feasible endpoint `T+tD`.

With `u=e_i-e_k`, `v=e_j-e_l`, its exact objective change is

\[
F(T+tD)-F(T)
=t\langle\nabla F(T),D\rangle-
\frac{2\alpha t^2}{s^2}
(u^TK_1u)(v^TK_2v).
\]

Therefore cycle exchanges can be evaluated using only four
gradient entries and two pairwise similarity lookups.
Implemented in `rectangle_vertex_exchange`. This is **not**
a complete search over the neighboring bases of a general
transportation polytope.

## Computational protocol

1. **Equal-restart experiment:** same product start followed by the
   same random linear-OT vertices, same maximum iterations, compare
   both objectives using one *independent* squared-loss evaluator.
2. **Equal-time experiment:** give sparse multi-start the measured POT
   wall-clock budget and record the number of completed restarts.
   Budgets are soft, because individual LP/EMD oracle calls are not
   preemptible.
3. **Diagnostics:** report best/median objective over starts, marginal
   residuals, end-of-run FW gap, oracle time, contraction time, total
   time, and tracemalloc peak.
4. **Small exhaustive reference:** enumerate all candidate transport
   bases for tiny problems and check objective at each feasible vertex;
   this numerically exhausts the finite set up to floating-point
   tolerance, but is not a *formal exact-arithmetic* certificate.
5. **Real-world validation:** add fixed multi-donor single-cell PBM
   comparisons, varying landmark count, sparsity, epsilon, bandwidth,
   random seed, and feature/structural scales.

For the reproducible benchmark run:

```bash
python benchmarks/bench_fgw_optimization.py --n 250 --landmarks 12 --restarts 4
```

## Exact global certificate in the rational 2x2 case

For two source and two target vertices, every feasible coupling is

\[
T(t)=\begin{pmatrix}
t&a_1-t\\
b_1-t&b_2-a_1+t
\end{pmatrix},
\quad
\max(0,a_1-b_2)\le t\le\min(a_1,b_1).
\]

If both similarities are PSD, Theorem 1 implies that the global
minimum occurs at one of these **two endpoints**. When every input
weight, cost, similarity, and scale is rational, Python `Fraction`
evaluates both objectives exactly and gives a genuine exact-arithmetic
global certificate for that rational instance.

```python
from probabilistic_ball_mapper._fgw_optimization import (
    exact_rational_2x2_certificate
)

value, plan, certificate = exact_rational_2x2_certificate(
    M=[["0", "1/5"], ["9/10", "1/10"]],
    K1=[["1", "3/10"], ["3/10", "1"]],
    K2=[["1", "1/2"], ["1/2", "1"]],
    a=["3/5", "2/5"],
    b=["9/20", "11/20"],
    alpha="2/5",
)
```

Floating-point inputs are intentionally rejected. Do not treat this as a
general exact certificate for floating-point PBM summaries or large graphs.
