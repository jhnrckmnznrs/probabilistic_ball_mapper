import numpy as np

from probabilistic_ball_mapper import MembershipConfig, compute_membership_matrix


def test_membership_rows_sum_to_one():
    X = np.array([[0.0], [0.5], [1.0]])
    landmarks = np.array([[0.0], [1.0]])
    config = MembershipConfig(eps=0.75, membership="triangular")

    W, uncovered = compute_membership_matrix(X, landmarks, config)

    assert W.shape == (3, 2)
    assert uncovered.size == 0
    np.testing.assert_allclose(W.sum(axis=1), np.ones(3))


def test_out_of_cover_nearest_policy():
    X = np.array([[0.0], [10.0]])
    landmarks = np.array([[0.0]])
    config = MembershipConfig(eps=0.5, membership="uniform", out_of_cover_policy="nearest")

    W, uncovered = compute_membership_matrix(X, landmarks, config)

    assert uncovered.tolist() == [1]
    np.testing.assert_allclose(W, np.ones((2, 1)))
