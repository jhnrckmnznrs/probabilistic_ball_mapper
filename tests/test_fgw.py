import networkx as nx
import numpy as np
import pytest

from probabilistic_ball_mapper import (
    FGWConfig,
    MembershipConfig,
    fgw_distance,
    fit_from_landmarks,
    summarize_static_pbm,
    vertex_feature_cost_matrix,
)


def _singleton_pbm(location: float, eps: float = 1.0):
    point = np.array([[location]])
    return fit_from_landmarks(
        point,
        point,
        config=MembershipConfig(eps=eps, membership="uniform"),
    )


def test_fixed_geometric_scaling_does_not_erase_singleton_separation():
    pbm_a = _singleton_pbm(0.0)
    pbm_b = _singleton_pbm(10.0)

    cost = vertex_feature_cost_matrix(pbm_a, pbm_b, geo_scale=2.0)
    distance, info = fgw_distance(
        pbm_a,
        pbm_b,
        config=FGWConfig(alpha=0.5, geo_scale=2.0),
    )

    np.testing.assert_allclose(cost, np.array([[25.0]]))
    np.testing.assert_allclose(info["feature_cost_matrix"], cost)
    assert distance == pytest.approx(np.sqrt(12.5))
    assert distance > 0.0


def test_epsilon_scaling_requires_a_common_epsilon():
    pbm_a = _singleton_pbm(0.0, eps=1.0)
    pbm_b = _singleton_pbm(0.0, eps=2.0)

    with pytest.raises(ValueError, match="same epsilon"):
        fgw_distance(pbm_a, pbm_b, config=FGWConfig(geo_scale="epsilon"))


def test_identical_pbm_has_zero_fgw_distance():
    X = np.array([[0.0], [0.25], [0.75], [1.0]])
    landmarks = np.array([[0.0], [1.0]])
    pbm = fit_from_landmarks(
        X,
        landmarks,
        config=MembershipConfig(eps=0.8, membership="gaussian"),
    )

    distance, info = fgw_distance(
        pbm,
        pbm,
        config=FGWConfig(alpha=0.5, geo_scale="epsilon"),
        return_plan=True,
    )

    assert distance == pytest.approx(0.0, abs=1e-7)
    np.testing.assert_allclose(
        info["transport_plan"].sum(axis=1),
        pbm.vertex_mass,
        atol=1e-7,
    )


def test_fgw_preserves_soft_overlap_magnitude_difference():
    graph = nx.Graph([(0, 1)])
    landmarks = np.array([[0.0], [1.0]])
    config = MembershipConfig(eps=2.0, membership="uniform")
    hard = summarize_static_pbm(
        W=np.array([[1.0, 0.0], [0.0, 1.0]]),
        graph=graph,
        landmarks=None,
        landmark_points=landmarks,
        cover=None,
        config=config,
    )
    soft = summarize_static_pbm(
        W=np.array([[0.5, 0.5], [0.5, 0.5]]),
        graph=graph,
        landmarks=None,
        landmark_points=landmarks,
        cover=None,
        config=config,
    )

    distance, _ = fgw_distance(
        hard,
        soft,
        config=FGWConfig(
            alpha=0.5,
            relation="soft_overlap",
            geo_scale=2.0,
        ),
    )

    assert distance > 0.0
