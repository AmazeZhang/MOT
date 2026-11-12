from __future__ import annotations

import pytest

from mot_research.evaluation.aggregation_integrity import canonicalize_comparators


def seed(value: float) -> dict[str, object]:
    comparator = {
        "overall": {"sample_count": 4, "box_iou_mean": value},
        "empty": {"sample_count": 0, "box_iou_mean": None},
    }
    return {
        "seed": 1,
        "dev": {
            "learned_gru": {"untouched": value},
            "zero_delta": comparator,
            "constant_velocity": comparator,
        },
    }


def test_comparator_tolerance_canonicalizes_only_comparators() -> None:
    first = seed(0.8)
    second = seed(0.8 + 3e-16)
    values, maximum = canonicalize_comparators([first, second], absolute_tolerance=1e-12)
    assert maximum["zero_delta"] < 1e-12
    assert values[0]["dev"]["zero_delta"] == values[1]["dev"]["zero_delta"]
    assert values[0]["dev"]["learned_gru"] != values[1]["dev"]["learned_gru"]


def test_comparator_tolerance_rejects_material_or_structural_difference() -> None:
    with pytest.raises(ValueError, match="float mismatch"):
        canonicalize_comparators([seed(0.8), seed(0.81)], absolute_tolerance=1e-12)
    changed = seed(0.8)
    changed["dev"]["zero_delta"]["overall"]["sample_count"] = 5
    with pytest.raises(ValueError, match="value mismatch"):
        canonicalize_comparators([seed(0.8), changed], absolute_tolerance=1e-12)
