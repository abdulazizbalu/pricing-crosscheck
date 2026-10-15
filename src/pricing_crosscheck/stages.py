"""Registry of measurement stages run by ``python -m pricing_crosscheck run``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterator

from pricing_crosscheck import env
from pricing_crosscheck.results import Table

PROFILES = ("fast", "full")

# A stage receives (profile, jobs, out_root, context) and yields result tables.
# ``context`` carries tables produced by earlier stages (by name).
Stage = Callable[[str, int, Path, dict[str, Any]], Iterator[Table]]


def environment_stage(profile: str, jobs: int, out_root: Path, ctx: dict[str, Any]) -> Iterator[Table]:
    table = Table("environment", ["key", "value"])
    # Python version and platform are in the file header, not in the table,
    # so the table compares equal across runners with identical libraries.
    for key, value in env.hashed_versions(env.library_versions()).items():
        table.add(key=key, value=value)
    yield table


def prices_stage(profile: str, jobs: int, out_root: Path, ctx: dict[str, Any]) -> Iterator[Table]:
    from pricing_crosscheck.pricing import price_table

    yield price_table(profile, jobs)


def identities_stage(profile: str, jobs: int, out_root: Path, ctx: dict[str, Any]) -> Iterator[Table]:
    from pricing_crosscheck.identities import call_spread_table, identity_table

    yield identity_table(ctx["prices"], profile, jobs)
    yield call_spread_table(jobs)


STAGES: list[Stage] = [environment_stage, prices_stage, identities_stage]


def run_stages(profile: str, jobs: int = 1, out_root: Path | None = None) -> Iterator[Table]:
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}")
    ctx: dict[str, Any] = {}
    for stage in STAGES:
        for table in stage(profile, jobs, out_root or env.REPO_ROOT / "results", ctx):
            ctx[table.name] = table
            yield table
