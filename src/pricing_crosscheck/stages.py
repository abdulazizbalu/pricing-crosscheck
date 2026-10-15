"""Registry of measurement stages run by ``python -m pricing_crosscheck run``."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator

from pricing_crosscheck import env
from pricing_crosscheck.results import Table

PROFILES = ("fast", "full")

# A stage receives (profile, jobs, out_root) and yields result tables.
Stage = Callable[[str, int, Path], Iterator[Table]]


def environment_stage(profile: str, jobs: int, out_root: Path) -> Iterator[Table]:
    table = Table("environment", ["key", "value"])
    # Full versions, including platform and Python patch release, are in the header.
    for key, value in env.hashed_versions(env.library_versions()).items():
        table.add(key=key, value=value)
    yield table


STAGES: list[Stage] = [environment_stage]


def run_stages(profile: str, jobs: int = 1, out_root: Path | None = None) -> Iterator[Table]:
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}")
    for stage in STAGES:
        yield from stage(profile, jobs, out_root or env.REPO_ROOT / "results")
