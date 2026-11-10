"""Official TrackEval sanity cases for the MOT 2D evaluation path."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import trackeval


EXPECTED = {
    "perfect": {"HOTA": 1.0, "DetA": 1.0, "AssA": 1.0, "MOTA": 1.0, "IDSW": 0, "IDF1": 1.0},
    "miss": {"HOTA": 0.5, "DetA": 0.5, "AssA": 0.5, "MOTA": 0.5, "IDSW": 0, "IDF1": 2.0 / 3.0},
    "swap": {
        "HOTA": np.sqrt(1.0 / 3.0),
        "DetA": 1.0,
        "AssA": 1.0 / 3.0,
        "MOTA": 0.5,
        "IDSW": 2,
        "IDF1": 0.5,
    },
}


def evaluate_fixtures(fixture_root: Path) -> dict[str, dict[str, float | int]]:
    """Evaluate three tiny MOTChallenge-format sequences with official TrackEval."""
    fixture_root = fixture_root.resolve()
    with TemporaryDirectory(prefix="mot-trackeval-") as output_dir:
        evaluator_config = trackeval.Evaluator.get_default_eval_config()
        evaluator_config.update(
            {
                "USE_PARALLEL": False,
                "PRINT_RESULTS": False,
                "PRINT_ONLY_COMBINED": False,
                "PRINT_CONFIG": False,
                "TIME_PROGRESS": False,
                "OUTPUT_SUMMARY": False,
                "OUTPUT_DETAILED": False,
                "PLOT_CURVES": False,
                "DISPLAY_LESS_PROGRESS": True,
            }
        )
        dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        dataset_config.update(
            {
                "GT_FOLDER": str(fixture_root / "gt"),
                "TRACKERS_FOLDER": str(fixture_root / "trackers"),
                "OUTPUT_FOLDER": output_dir,
                "TRACKERS_TO_EVAL": ["sanity"],
                "CLASSES_TO_EVAL": ["pedestrian"],
                "BENCHMARK": "MOT17",
                "SPLIT_TO_EVAL": "val",
                "SKIP_SPLIT_FOL": True,
                "SEQMAP_FILE": str(fixture_root / "seqmap.txt"),
                "DO_PREPROC": False,
                "PRINT_CONFIG": False,
            }
        )
        metrics = [
            trackeval.metrics.HOTA(),
            trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
            trackeval.metrics.Identity({"PRINT_CONFIG": False}),
        ]
        raw_results, messages = trackeval.Evaluator(evaluator_config).evaluate(
            [trackeval.datasets.MotChallenge2DBox(dataset_config)], metrics
        )

    dataset_results = raw_results["MotChallenge2DBox"]["sanity"]
    if messages["MotChallenge2DBox"]["sanity"] != "Success":
        raise RuntimeError(messages["MotChallenge2DBox"]["sanity"])
    results: dict[str, dict[str, float | int]] = {}
    for sequence in EXPECTED:
        values: dict[str, Any] = dataset_results[sequence]["pedestrian"]
        results[sequence] = {
            "HOTA": float(np.mean(values["HOTA"]["HOTA"])),
            "DetA": float(np.mean(values["HOTA"]["DetA"])),
            "AssA": float(np.mean(values["HOTA"]["AssA"])),
            "MOTA": float(values["CLEAR"]["MOTA"]),
            "IDSW": int(values["CLEAR"]["IDSW"]),
            "IDF1": float(values["Identity"]["IDF1"]),
        }
    return results


def assert_expected(results: dict[str, dict[str, float | int]]) -> None:
    """Raise an assertion error if any official metric differs from the fixture contract."""
    for sequence, expected_metrics in EXPECTED.items():
        for metric, expected in expected_metrics.items():
            actual = results[sequence][metric]
            if metric == "IDSW":
                assert actual == expected, f"{sequence}.{metric}: {actual} != {expected}"
            else:
                assert np.isclose(actual, expected, atol=1e-12), (
                    f"{sequence}.{metric}: {actual} != {expected}"
                )
