"""Small, portable provenance and process-memory helpers for R1 runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import subprocess
import sys
from pathlib import Path

try:
    import resource as _resource
except ImportError:  # Windows does not provide resource.
    _resource = None


DEPENDENCIES = ("numpy", "scipy", "filterpy", "lap", "PyYAML")
R1_SHARED_CODE_PATHS = (
    "scripts/run_p1_natural_closed_loop.py",
    "src/mot_research/analysis/failure_events.py",
    "src/mot_research/analysis/r1_components.py",
    "src/mot_research/analysis/runtime_record.py",
    "src/mot_research/data/motion_samples.py",
    "src/mot_research/tracking/first_round_adapter.py",
    "src/mot_research/tracking/natural_motion_policy.py",
    "third_party/OC_SORT/trackers/ocsort_tracker/association.py",
    "third_party/OC_SORT/trackers/ocsort_tracker/kalmanfilter.py",
    "third_party/OC_SORT/trackers/ocsort_tracker/ocsort.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def peak_rss_kib() -> int | None:
    """Return peak RSS in KiB, or None when the OS cannot report it."""
    if _resource is None:
        return None
    try:
        usage = _resource.getrusage(_resource.RUSAGE_SELF)
    except (AttributeError, OSError):
        return None
    value = int(usage.ru_maxrss)
    return value // 1024 if sys.platform == "darwin" else value


def runtime_record(repo_root: Path, code_paths: tuple[str, ...]) -> dict[str, object]:
    """Capture versions and source hashes before processing input data."""
    root = repo_root.resolve()
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    versions: dict[str, str | None] = {}
    for package in DEPENDENCIES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    paths = tuple(dict.fromkeys(code_paths))
    hashes = {name: sha256_file(root / name) for name in paths}
    return {
        "git_head": commit,
        "python_version": sys.version.split()[0],
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": sys.platform,
        "dependencies": versions,
        "code_sha256": hashes,
    }


def verify_code_unchanged(repo_root: Path, record: dict[str, object]) -> None:
    """Fail a run if any recorded source changed after provenance capture."""
    hashes = record["code_sha256"]
    if not isinstance(hashes, dict):
        raise TypeError("runtime record lacks code hashes")
    root = repo_root.resolve()
    changed = [name for name, expected in hashes.items()
               if sha256_file(root / name) != expected]
    if changed:
        raise RuntimeError(f"source changed during run: {changed}")
