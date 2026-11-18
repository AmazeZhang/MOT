import numpy as np
import pytest

from mot_research.analysis.r1_components import components, switched_result


def edges(*pairs):
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def test_swap_is_one_indivisible_component():
    a0 = edges((0, 0), (1, 1))
    a1 = edges((0, 1), (1, 0))
    found = components(a0, a1, 2, 2)
    assert len(found) == 1
    matched, unmatched_d, unmatched_t = switched_result(a0, found[0], 2, 2)
    assert np.array_equal(matched, a1)
    assert len(unmatched_d) == len(unmatched_t) == 0


def test_disjoint_components_leave_other_matches_untouched():
    a0 = edges((0, 0), (1, 1), (2, 2))
    a1 = edges((0, 1), (1, 0), (2, 2), (3, 3))
    found = components(a0, a1, 4, 4)
    assert len(found) == 2
    swapped = next(part for part in found if (0, 1) in part.a1_edges)
    matched, unmatched_d, unmatched_t = switched_result(a0, swapped, 4, 4)
    assert set(map(tuple, matched)) == {(0, 1), (1, 0), (2, 2)}
    assert unmatched_d.tolist() == [3]
    assert unmatched_t.tolist() == [3]


def test_invalid_non_one_to_one_proposals_are_rejected():
    with pytest.raises(ValueError):
        components(edges((0, 0), (0, 1)), edges(), 2, 2)
