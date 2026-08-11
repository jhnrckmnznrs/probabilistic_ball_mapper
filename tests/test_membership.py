import numpy as np
import pytest

from probabilistic_ball_mapper import (
    MembershipConfig,
    compute_global_gaussian_membership_matrix,
    compute_membership_matrix,
)


def test_localized_membership_is_stochastic_and_subordinate():
    X = np.array([[0.0], [0.5], [1.0]])
    landmarks = np.array([[0.0], [1.0]])
    config = MembershipConfig(eps=0.75, membership="triangular")

    W, uncovered = compute_membership_matrix(X, landmarks, config)
    distances = np.abs(X - landmarks.T)

    assert W.shape == (3, 2)
    assert uncovered.size == 0
    np.testing.assert_allclose(W.sum(axis=1), np.ones(3))
    assert np.all(W[distances >= config.eps] == 0.0)


@pytest.mark.parametrize(
    "membership",
    [
        "uniform",
        "gaussian",
        "compact_gaussian",
        "triangular",
        "epanechnikov",
        "quartic",
        "inverse_distance",
        "inverse_square",
    ],
)
def test_all_membership_kernels_are_finite_and_stochastic(membership):
    X = np.array([[0.0], [0.5], [1.0]])
    landmarks = np.array([[0.0], [1.0]])
    config = MembershipConfig(eps=1.1, membership=membership)

    W, uncovered = compute_membership_matrix(X, landmarks, config)

    assert uncovered.size == 0
    assert np.all(np.isfinite(W))
    assert np.all(W >= 0.0)
    np.testing.assert_allclose(W.sum(axis=1), np.ones(3))


def test_open_ball_boundary_is_outside_and_default_policy_raises():
    X = np.array([[1.0]])
    landmarks = np.array([[0.0]])
    config = MembershipConfig(eps=1.0, membership="uniform")

    with pytest.raises(RuntimeError, match="outside the fixed cover"):
        compute_membership_matrix(X, landmarks, config)


def test_explicit_nearest_policy_reports_extrapolated_points():
    X = np.array([[0.0], [10.0]])
    landmarks = np.array([[0.0]])
    config = MembershipConfig(
        eps=0.5,
        membership="uniform",
        out_of_cover_policy="nearest",
    )

    W, uncovered = compute_membership_matrix(X, landmarks, config)

    assert uncovered.tolist() == [1]
    np.testing.assert_allclose(W, np.ones((2, 1)))


@pytest.mark.parametrize("membership", ["gaussian", "compact_gaussian"])
def test_gaussian_log_normalization_avoids_all_zero_underflow(membership):
    X = np.array([[0.0]])
    landmarks = np.array([[-1.0], [1.0]])
    config = MembershipConfig(
        eps=2.0,
        membership=membership,
        sigma=1e-6,
    )

    W, uncovered = compute_membership_matrix(X, landmarks, config)

    assert uncovered.size == 0
    np.testing.assert_allclose(W, np.array([[0.5, 0.5]]), atol=1e-14)


def test_global_gaussian_coordinates_are_stable_and_report_cover_status():
    X = np.array([[0.0], [1000.0]])
    landmarks = np.array([[-1.0], [1.0]])

    W, diagnostics = compute_global_gaussian_membership_matrix(
        X,
        landmarks,
        sigma=1e-6,
        cover_eps=2.0,
    )

    np.testing.assert_allclose(W.sum(axis=1), np.ones(2))
    np.testing.assert_allclose(W[0], np.array([0.5, 0.5]), atol=1e-14)
    assert diagnostics["outside_cover"].tolist() == [False, True]
    np.testing.assert_allclose(
        diagnostics["min_landmark_distance"],
        np.array([1.0, 999.0]),
    )
