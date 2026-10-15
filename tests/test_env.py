from pricing_crosscheck import env
from pricing_crosscheck.cli import main


def test_versions_are_recorded():
    versions = env.library_versions()
    for key in ("python", "QuantLib", "numpy", "scipy"):
        assert versions[key] and versions[key] != "not installed"
    assert versions["QuantLib"] == versions["QuantLib_compiled"]


def test_content_hash_is_stable():
    assert env.content_hash() == env.content_hash()
    assert len(env.content_hash()) == 64


def test_hash_covers_code_and_specs():
    names = {p.relative_to(env.REPO_ROOT).as_posix() for p in env.hashed_files()}
    assert "src/pricing_crosscheck/env.py" in names
    assert "requirements.lock" in names


def test_run_and_check_roundtrip(tmp_path):
    assert main(["run", "--profile", "fast", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "fast" / "manifest.json").exists()
    assert main(["check", "--profile", "fast", "--results", str(tmp_path)]) == 0
