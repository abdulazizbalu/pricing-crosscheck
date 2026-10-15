"""Environment metadata and content hashing for reproducible results."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from importlib import metadata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Packages whose exact versions are written into every results file.
TRACKED_PACKAGES = ("QuantLib", "numpy", "scipy", "matplotlib")

# Files whose content determines the numbers in results/.
HASHED_GLOBS = ("src/pricing_crosscheck/**/*.py", "specs/*.json", "requirements.lock")


def library_versions() -> dict[str, str]:
    """Exact versions of the interpreter and the numerical libraries in use."""
    versions = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": f"{sys.platform}-{platform.machine()}",
    }
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    import QuantLib as ql  # imported here so a missing wheel fails loudly only when needed

    versions["QuantLib_compiled"] = ql.__version__
    return versions


def hashed_versions(versions: dict[str, str]) -> dict[str, str]:
    """The subset of ``library_versions`` that enters the content hash."""
    out = {k: v for k, v in versions.items() if k != "platform"}
    out["python"] = ".".join(out["python"].split(".")[:2])
    return out


def hashed_files(root: Path = REPO_ROOT) -> list[Path]:
    files: set[Path] = set()
    for pattern in HASHED_GLOBS:
        files.update(p for p in root.glob(pattern) if p.is_file())
    return sorted(files)


def content_hash(root: Path = REPO_ROOT, versions: dict[str, str] | None = None) -> str:
    """SHA-256 over the code, the specs, the lock file and the library versions.

    The platform string and the Python patch release are excluded so the hash
    is stable across CI runners with identical library versions; both are
    still recorded in the header of every results file.
    """
    versions = hashed_versions(versions or library_versions())
    digest = hashlib.sha256()
    for path in hashed_files(root):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
        digest.update(b"\0")
    digest.update(json.dumps(versions, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def metadata_lines(profile: str, root: Path = REPO_ROOT) -> list[str]:
    """Comment lines written at the top of every CSV file in results/."""
    versions = library_versions()
    lines = [f"# profile: {profile}", f"# content_hash: {content_hash(root, versions)}"]
    lines += [f"# {key}: {value}" for key, value in versions.items()]
    return lines
