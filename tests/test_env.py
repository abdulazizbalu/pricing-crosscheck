import pytest

from pricing_crosscheck import env, stages
from pricing_crosscheck.cli import main


@pytest.fixture(autouse=True)
def environment_stage_only(monkeypatch):
    """CLI mechanics are tested on the environment stage only; the measurement
    stages are exercised by the fast-measurements CI job."""
    monkeypatch.setattr(stages, "STAGES", [stages.environment_stage])
    import pricing_crosscheck.cli as cli

    monkeypatch.setattr(cli, "write_report", lambda *a, **k: None)


def test_versions_are_recorded():
    versions = env.library_versions()
    for key in ("python", "QuantLib", "numpy", "scipy"):
        assert versions[key] and versions[key] != "not installed"
    assert versions["QuantLib"] == versions["QuantLib_compiled"]


def test_content_hash_is_stable():
    assert env.content_hash() == env.content_hash()
    assert len(env.content_hash()) == 64


def test_hash_covers_tracked_code_and_lock():
    names = {p.relative_to(env.REPO_ROOT).as_posix() for p in env.hashed_files()}
    assert "src/pricing_crosscheck/env.py" in names
    assert "requirements.lock" in names
    assert not any(n.startswith("tests/") for n in names)


def test_hash_excludes_python_and_platform():
    v = env.library_versions()
    h = env.hashed_versions(v)
    assert "platform" not in h and "python" not in h
    assert h["QuantLib"] == v["QuantLib"]


def test_run_and_check_roundtrip(tmp_path):
    assert main(["run", "--profile", "fast", "--out", str(tmp_path), "--allow-unlocked"]) == 0
    assert (tmp_path / "fast" / "manifest.json").exists()
    assert main(["status", "--profile", "fast", "--results", str(tmp_path)]) == 0
    assert main(["check", "--profile", "fast", "--results", str(tmp_path), "--allow-unlocked"]) == 0


def test_check_refuses_same_directory(tmp_path):
    assert main(["run", "--profile", "fast", "--out", str(tmp_path), "--allow-unlocked"]) == 0
    code = main(["check", "--profile", "fast", "--results", str(tmp_path), "--regenerated", str(tmp_path), "--allow-unlocked"])
    assert code == 2


def test_missing_results(tmp_path):
    assert main(["status", "--profile", "full", "--results", str(tmp_path)]) == 1
    assert main(["status", "--profile", "full", "--results", str(tmp_path), "--allow-missing"]) == 0


def test_allow_missing_still_regenerates(tmp_path):
    regen = tmp_path / "regen"
    code = main(["check", "--profile", "fast", "--results", str(tmp_path / "none"), "--allow-missing",
                 "--allow-unlocked", "--regenerated", str(regen)])
    assert code == 0
    assert (regen / "fast" / "manifest.json").exists()
