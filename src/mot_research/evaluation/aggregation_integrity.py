"""Integrity checks for aggregating independently computed offline comparators."""

from __future__ import annotations

import copy
from typing import Any


def _maximum_numeric_difference(first: Any, second: Any, path: str, tolerance: float) -> float:
    if type(first) is not type(second):
        raise ValueError(f"comparator type mismatch at {path}: {type(first)} != {type(second)}")
    if isinstance(first, dict):
        if set(first) != set(second):
            raise ValueError(f"comparator keys mismatch at {path}")
        return max(
            (_maximum_numeric_difference(first[key], second[key], f"{path}/{key}", tolerance) for key in first),
            default=0.0,
        )
    if isinstance(first, list):
        if len(first) != len(second):
            raise ValueError(f"comparator list length mismatch at {path}")
        return max(
            (_maximum_numeric_difference(a, b, f"{path}/{index}", tolerance) for index, (a, b) in enumerate(zip(first, second))),
            default=0.0,
        )
    if isinstance(first, float):
        difference = abs(first - second)
        if difference > tolerance:
            raise ValueError(f"comparator float mismatch at {path}: {difference} > {tolerance}")
        return difference
    if first != second:
        raise ValueError(f"comparator value mismatch at {path}: {first!r} != {second!r}")
    return 0.0


def canonicalize_comparators(
    seed_metrics: list[dict[str, Any]],
    *,
    absolute_tolerance: float,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Verify comparator trees within tolerance, then use one canonical copy."""
    if not seed_metrics:
        raise ValueError("seed metrics must not be empty")
    if absolute_tolerance < 0:
        raise ValueError("absolute tolerance must be non-negative")
    values = copy.deepcopy(seed_metrics)
    reference = values[0]["dev"]
    maximum: dict[str, float] = {}
    for name in ("zero_delta", "constant_velocity"):
        maximum[name] = 0.0
        for item in values[1:]:
            difference = _maximum_numeric_difference(
                reference[name], item["dev"][name], f"/{name}", absolute_tolerance
            )
            maximum[name] = max(maximum[name], difference)
            item["dev"][name] = copy.deepcopy(reference[name])
    return values, maximum
