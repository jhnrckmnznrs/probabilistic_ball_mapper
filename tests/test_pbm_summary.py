import numpy as np

from probabilistic_ball_mapper import (
    MembershipConfig,
    fit_from_landmarks,
    relation_matrix,
    summarize,
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


def test_relation_matrix_adjacency_is_symmetric_with_zero_diagonal():
    X = np.array([[0.0], [0.25], [0.75], [1.0]])
    landmarks = np.array([[0.0], [1.0]])
    config = MembershipConfig(eps=0.8, membership="uniform")
    pbm = fit_from_landmarks(X, landmarks, config=config)

    C = relation_matrix(pbm, relation="adjacency")

    assert C.shape == (2, 2)
    np.testing.assert_allclose(C, C.T)
    np.testing.assert_allclose(np.diag(C), np.zeros(2))
