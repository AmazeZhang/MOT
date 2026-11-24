import numpy as np
import pytest

from mot_research.analysis.multi_assignment_components import apply_action, components


def edges(*pairs):
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def test_three_way_union_merges_pairwise_overlapping_differences():
    found = components({
        "M": edges((0, 0), (1, 1), (2, 2)),
        "G": edges((0, 1), (1, 0), (2, 2)),
        "V": edges((0, 0), (1, 2), (2, 1)),
    }, 3, 3, reference="M")
    assert len(found) == 1
    assert found[0].detection_rows == (0, 1, 2)
    assert found[0].tracker_rows == (0, 1, 2)
    assert found[0].unique_actions(("M", "G", "V")) == ("M", "G", "V")


def test_common_edge_is_retained_when_connected_to_difference():
    found = components({
        "M": edges((0, 0), (1, 1)),
        "G": edges((0, 0), (1, 1)),
        "V": edges((0, 1), (1, 0)),
    }, 2, 2, reference="M")
    assert len(found) == 1
    assert found[0].action("M") == ((0, 0), (1, 1))
    assert found[0].action("G") == ((0, 0), (1, 1))
    assert found[0].unique_actions(("M", "G", "V")) == ("M", "V")


def test_identical_isolated_components_are_not_actions():
    found = components({"M": edges((0, 0)), "G": edges((0, 0)), "V": edges((0, 0))},
                       1, 1, reference="M")
    assert found == []


def test_unmatched_endpoint_and_action_application_remain_legal():
    found = components({"M": edges((0, 0)), "G": edges(), "V": edges((0, 1))},
                       1, 2, reference="M")
    assert len(found) == 1
    matched, unmatched_d, unmatched_t = apply_action(edges((0, 0)), found[0], "V", 1, 2)
    assert np.array_equal(matched, edges((0, 1)))
    assert unmatched_d.tolist() == []
    assert unmatched_t.tolist() == [0]


def test_non_one_to_one_candidate_is_rejected():
    with pytest.raises(ValueError):
        components({"M": edges((0, 0)), "G": edges((0, 0), (0, 1)), "V": edges()},
                   1, 2, reference="M")
