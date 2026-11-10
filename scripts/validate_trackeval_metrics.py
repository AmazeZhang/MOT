#!/usr/bin/env python3
"""Run pinned TrackEval against the repository's three artificial MOT cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mot_research.evaluation.trackeval_sanity import assert_expected, evaluate_fixtures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=Path("tests/fixtures/evaluation"),
        help="MOTChallenge-format sanity fixture root",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON result path")
    args = parser.parse_args()

    results = evaluate_fixtures(args.fixtures)
    assert_expected(results)
    payload = {"status": "PASS", "results": results}
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
