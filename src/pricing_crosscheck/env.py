"""Environment metadata and content hashing for reproducible results."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
from fnmatch import fnmatch
from importlib import metadata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCK_FILE = "requirements.lock"

# Packages whose exact versions are written into every results file and hashed.
TRACKED_PACKAGES = ("QuantLib", "numpy", "scipy", "matplotlib")

# Files whose content determines the numbers in results/.
HASHED_PATTERNS = ("src/pricing_crosscheck/*.py", "src/pricing_crosscheck/**/*.py", "specs/*.json", LOCK_FILE)
REQUIRED_FILES = ("src/pricing_crosscheck/env.py", LOCK_FILE)


def check_repo_root(root: Path = REPO_ROOT) -> Path:
    """The package must run from a source checkout (editable install)."""
    missing = [f for f in REQUIRED_FILES if not (root / f).is_file()]
    if missing:
        raise RuntimeError(
            f"{root} is not a pricing-crosscheck checkout (missing {missing}); "
            "install with `pip install --no-deps -e .` from the repository"
        )
    return root


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
    import QuantLib as ql

    versions["QuantLib_compiled"] = ql.__version__
    return versions


def hashed_versions(versions: dict[str, str]) -> dict[str, str]:
    """The subset of ``library_versions`` that enters the content hash.

    Platform and Python version are recorded in every file header but not
    hashed: committed results are generated with CPython 3.12 on linux-x86_64,
    and the CI check compares numbers with a tolerance, not the hash alone.
    """
    return {k: v for k, v in versions.items() if k not in ("platform", "python", "implementation")}


def _git_files(root: Path) -> list[str] | None:
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True, timeout=30
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return [p for p in out.decode("utf-8").split("\0") if p]


def hashed_files(root: Path = REPO_ROOT) -> list[Path]:
    """Tracked files that determine results (git index if available)."""
    check_repo_root(root)
    candidates = _git_files(root)
    if candidates is None:
        candidates = [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()]
    selected = sorted(
        rel for rel in candidates if any(fnmatch(rel, pat) for pat in HASHED_PATTERNS) and (root / rel).is_file()
    )
    if not selected:
        raise RuntimeError("no files selected for the content hash")
    return [root / rel for rel in selected]


def content_hash(root: Path = REPO_ROOT, versions: dict[str, str] | None = None) -> str:
    """SHA-256 over tracked code, specs, the lock file and the library versions."""
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
    lines.append(f"# all other packages: {LOCK_FILE}")
    return lines


_PIN = re.compile(r"^\s*([A-Za-z0-9_.\-]+)==([^\s;]+)\s*(?:;\s*(.+))?$")


def _marker_applies(marker: str | None) -> bool:
    if not marker:
        return True
    m = re.fullmatch(r'python_version\s*(<|<=|>|>=|==)\s*"(\d+)\.(\d+)"', marker.strip())
    if not m:
        raise ValueError(f"unsupported marker in {LOCK_FILE}: {marker}")
    op, major, minor = m.group(1), int(m.group(2)), int(m.group(3))
    current, wanted = sys.version_info[:2], (major, minor)
    return {
        "<": current < wanted,
        "<=": current <= wanted,
        ">": current > wanted,
        ">=": current >= wanted,
        "==": current == wanted,
    }[op]


def _normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def lock_mismatches(root: Path = REPO_ROOT) -> list[str]:
    """Differences between the installed distributions and the lock file."""
    problems = []
    installed = {_normalise(d.metadata["Name"]): d.version for d in metadata.distributions()}
    for line in (root / LOCK_FILE).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _PIN.match(line)
        if not m:
            problems.append(f"unparsed lock line: {line}")
            continue
        name, version, marker = m.groups()
        if not _marker_applies(marker):
            continue
        have = installed.get(_normalise(name))
        if have != version:
            problems.append(f"{name}: lock {version}, installed {have}")
    return problems
