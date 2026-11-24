import numpy as np

from mot_research.analysis.component_action_features import (
    cosine_matrix, signed_competition_margins, upstream_direction_score,
)


def test_direction_score_shape_validity_and_detection_score_scaling():
    dets = np.array([[1, 0, 3, 2, .8], [1, 0, 3, 2, .4]], dtype=float)
    obs = np.array([[0, 0, 2, 2, .9], [0, 0, 2, 2, -1]], dtype=float)
    vel = np.array([[0, 1], [0, 1]], dtype=float)
    score = upstream_direction_score(dets, vel, obs, .2)
    assert score.shape == (2, 2)
    assert np.all(score[:, 1] == 0)
    assert np.isclose(score[1, 0], score[0, 0] / 2)


def test_direction_score_accepts_upstream_empty_velocity_shape():
    score = upstream_direction_score(np.array([[0, 0, 1, 1, .8]], float),
                                     np.array([], float), np.empty((0, 5)), .2)
    assert score.shape == (1, 0)
    upstream_shape = upstream_direction_score(np.array([[0, 0, 1, 1, .8]], float),
                                              np.array([], float), np.array([]), .2)
    assert upstream_shape.shape == (1, 0)


def test_cosine_invalid_rows_are_zero_and_masked():
    values, valid = cosine_matrix(np.array([[1, 0], [0, 0]], float),
                                  np.array([[1, 0], [0, 1]], float))
    assert values.tolist() == [[1.0, 0.0], [0.0, 0.0]]
    assert valid.tolist() == [[True, True], [False, False]]


def test_signed_competition_margins():
    row, col = signed_competition_margins(np.array([[3, 1], [2, 4]], float))
    assert row.tolist() == [[2.0, -2.0], [-2.0, 2.0]]
    assert col.tolist() == [[1.0, -3.0], [-1.0, 3.0]]
    one_row, one_col = signed_competition_margins(np.array([[5.0]]))
    assert one_row.item() == 0 and one_col.item() == 0
