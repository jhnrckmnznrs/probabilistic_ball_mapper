"""Reproducible end-to-end fixed-landmark dense/sparse PBM benchmark.

Run: python benchmarks/bench_sparse_pbm.py --n 12000 --grid 20 --repeats 3
BLAS threading is restricted; this is not a universal speedup guarantee.
"""
import argparse
from statistics import median
from time import perf_counter

import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

from probabilistic_ball_mapper import MembershipConfig, fit_from_landmarks


def nbytes(x):
    if sparse.issparse(x):
        return x.data.nbytes + x.indices.nbytes + x.indptr.nbytes
    return x.nbytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=12000)
    parser.add_argument("--grid", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.n <= 0 or args.grid < 2 or args.repeats <= 0:
        parser.error("n and repeats must be positive; grid must be at least 2")
    rng = np.random.default_rng(123)
    X = rng.uniform(0, 1, size=(args.n, 2))
    line = np.linspace(0, 1, args.grid)
    landmarks = np.stack(np.meshgrid(line, line), axis=-1).reshape(-1, 2)
    # For a square grid, this covers the entire unit square by open balls.
    eps = 1.01 * np.sqrt(2) / (2 * (args.grid - 1))
    cfg = MembershipConfig(eps=eps, membership="uniform")
    timing = {"dense": [], "sparse": []}
    samples = {}
    with threadpool_limits(limits=1):
        for _ in range(args.repeats):
            for mode in ["dense", "sparse"]:
                start = perf_counter()
                value = fit_from_landmarks(X, landmarks, config=cfg, storage=mode)
                timing[mode].append(perf_counter()-start)
                samples[mode] = value
    dense = samples["dense"]
    sp = samples["sparse"]
    difference = abs(dense.overlap_matrix - sp.overlap_matrix.toarray()).max()
    print(f"n={args.n} landmarks={len(landmarks)} nnz_per_point={sp.W.nnz/args.n:.2f}")
    print(f"dense_seconds_median={median(timing['dense']):.6f}")
    print(f"sparse_seconds_median={median(timing['sparse']):.6f}")
    print(f"speedup={median(timing['dense'])/median(timing['sparse']):.2f}x")
    print(f"dense_W_MiB={nbytes(dense.W)/2**20:.4f}")
    print(f"sparse_W_MiB={nbytes(sp.W)/2**20:.4f}")
    print(f"dense_overlap_MiB={nbytes(dense.overlap_matrix)/2**20:.4f}")
    print(f"sparse_overlap_MiB={nbytes(sp.overlap_matrix)/2**20:.4f}")
    print(f"max_overlap_absolute_error={difference:.3e}")
    assert np.allclose(dense.overlap_matrix, sp.overlap_matrix.toarray(), atol=1e-12)


if __name__ == "__main__":
    main()
