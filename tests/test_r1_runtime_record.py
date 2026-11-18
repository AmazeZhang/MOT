from pathlib import Path
from types import SimpleNamespace

import pytest

from mot_research.analysis import runtime_record as record


def test_peak_rss_is_optional_and_darwin_is_converted_to_kib(monkeypatch):
    monkeypatch.setattr(record, "_resource", None)
    assert record.peak_rss_kib() is None

    fake_resource = SimpleNamespace(
        RUSAGE_SELF=0,
        getrusage=lambda _: SimpleNamespace(ru_maxrss=4096),
    )
    monkeypatch.setattr(record, "_resource", fake_resource)
    monkeypatch.setattr(record.sys, "platform", "darwin")
    assert record.peak_rss_kib() == 4


def test_runtime_record_detects_changed_source(tmp_path: Path, monkeypatch):
    source = tmp_path / "example.py"
    source.write_text("before\n")
    monkeypatch.setattr(
        record.subprocess, "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="abc123\n"),
    )
    provenance = record.runtime_record(tmp_path, ("example.py",))
    assert provenance["git_head"] == "abc123"
    assert provenance["code_sha256"]["example.py"] == record.sha256_file(source)
    record.verify_code_unchanged(tmp_path, provenance)
    source.write_text("after\n")
    with pytest.raises(RuntimeError, match="source changed during run"):
        record.verify_code_unchanged(tmp_path, provenance)
