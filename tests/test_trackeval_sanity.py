from pathlib import Path

from mot_research.evaluation.trackeval_sanity import assert_expected, evaluate_fixtures


def test_official_trackeval_artificial_cases() -> None:
    fixture_root = Path(__file__).parent / "fixtures" / "evaluation"
    results = evaluate_fixtures(fixture_root)
    assert_expected(results)
