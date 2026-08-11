import networkx as nx
import numpy as np
import pytest

from probabilistic_ball_mapper import (
    LandmarkConfig,
    MembershipConfig,
    fit,
    fit_from_landmarks,
    relation_matrix,
    summarize,
    summarize_static_pbm,
)


def test_fit_from_landmarks_builds_graph_without_fast_ballmapper():
    X = np.array([[0.0], [0.25], [0.75], [1.0]])
    landmarks = np.array([[0.0], [1.0]])
    config = MembershipConfig(eps=0.8, membership="uniform")

    pbm = fit_from_landmarks(X, landmarks, config=config)

    assert pbm.n_points == 4
    assert pbm.n_vertices == 2
    assert pbm.graph.number_of_edges() == 1
    np.testing.assert_allclose(pbm.vertex_mass.sum(), 1.0)
    assert summarize(pbm)["n_edges"] == 1


def test_automatic_fit_respects_open_ball_boundary():
    X = np.array([[0.0], [1.0]])
    pbm = fit(
        X,
        config=MembershipConfig(eps=1.0, membership="uniform"),
        landmark_config=LandmarkConfig(selection="fps", start_index=0),
    )

    assert pbm.n_vertices == 2
    assert sorted(members.tolist() for members in pbm.cover) == [[0], [1]]


def test_nearest_extrapolation_does_not_change_geometric_cover():
    X = np.array([[0.0], [10.0]])
    landmarks = np.array([[0.0]])
    config = MembershipConfig(
        eps=0.5,
        membership="uniform",
        out_of_cover_policy="nearest",
    )

    pbm = fit_from_landmarks(X, landmarks, config=config)

    assert pbm.cover[0].tolist() == [0]
    assert pbm.outside_cover_indices.tolist() == [1]
    np.testing.assert_allclose(pbm.W, np.ones((2, 1)))


def test_summary_uses_supplied_probability_measure():
    W = np.array([[1.0, 0.0], [0.5, 0.5]])
    graph = nx.Graph([(0, 1)])
    config = MembershipConfig(eps=1.0, membership="uniform")

    pbm = summarize_static_pbm(
        W=W,
        graph=graph,
        landmarks=None,
        landmark_points=np.array([[0.0], [1.0]]),
        cover=None,
        config=config,
        sample_weight=np.array([3.0, 1.0]),
    )

    np.testing.assert_allclose(pbm.sample_weight, np.array([0.75, 0.25]))
    np.testing.assert_allclose(pbm.vertex_mass, np.array([0.875, 0.125]))
    np.testing.assert_allclose(
        pbm.overlap_matrix,
        np.array([[0.8125, 0.0625], [0.0625, 0.0625]]),
    )


def test_summary_rejects_invalid_memberships():
    config = MembershipConfig(eps=1.0, membership="uniform")
    with pytest.raises(ValueError, match="Entries of W"):
        summarize_static_pbm(
            W=np.array([[1.2, -0.2]]),
            graph=nx.Graph(),
            landmarks=None,
            landmark_points=np.array([[0.0], [1.0]]),
            cover=None,
            config=config,
        )


def test_relation_matrix_adjacency_is_symmetric_with_zero_diagonal():
    X = np.array([[0.0], [0.25], [0.75], [1.0]])
    landmarks = np.array([[0.0], [1.0]])
    config = MembershipConfig(eps=0.8, membership="uniform")
    pbm = fit_from_landmarks(X, landmarks, config=config)

    C = relation_matrix(pbm, relation="adjacency")

    assert C.shape == (2, 2)
    np.testing.assert_allclose(C, C.T)
    np.testing.assert_allclose(np.diag(C), np.zeros(2))


def test_soft_overlap_relation_preserves_overlap_magnitude():
    config = MembershipConfig(eps=2.0, membership="uniform")
    graph = nx.Graph([(0, 1)])
    hard = summarize_static_pbm(
        W=np.array([[1.0, 0.0], [0.0, 1.0]]),
        graph=graph,
        landmarks=None,
        landmark_points=np.array([[0.0], [1.0]]),
        cover=None,
        config=config,
    )
    soft = summarize_static_pbm(
        W=np.array([[0.5, 0.5], [0.5, 0.5]]),
        graph=graph,
        landmarks=None,
        landmark_points=np.array([[0.0], [1.0]]),
        cover=None,
        config=config,
    )

    C_hard = relation_matrix(hard, relation="soft_overlap")
    C_soft = relation_matrix(soft, relation="soft_overlap")

    assert C_hard[0, 1] == pytest.approx(1.0)
    assert C_soft[0, 1] == pytest.approx(0.5)
