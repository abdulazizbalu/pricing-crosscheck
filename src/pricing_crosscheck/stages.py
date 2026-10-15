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


def comparison_stage(profile: str, jobs: int, out_root: Path, ctx: dict[str, Any]) -> Iterator[Table]:
    from pricing_crosscheck.compare import comparison_table, summary_table
    from pricing_crosscheck.disagreements import disagreement_table

    comparisons = comparison_table(ctx["prices"])
    yield comparisons
    yield summary_table(comparisons)
    yield disagreement_table(comparisons, jobs)


def convergence_stage(profile: str, jobs: int, out_root: Path, ctx: dict[str, Any]) -> Iterator[Table]:
    from pricing_crosscheck.convergence_runs import convergence_tables
    from pricing_crosscheck.experiments import experiments_table

    yield from convergence_tables(profile, jobs)
    yield experiments_table(profile, jobs)


def greeks_stage(profile: str, jobs: int, out_root: Path, ctx: dict[str, Any]) -> Iterator[Table]:
    from pricing_crosscheck.greeks import greeks_tables

    yield from greeks_tables(profile, jobs)


STAGES: list[Stage] = [
    environment_stage, prices_stage, identities_stage, comparison_stage, convergence_stage, greeks_stage,
]


def write_report(profile: str, directory: Path, tables: dict[str, Table], publish: bool) -> None:
    """summary.md next to the CSV files; for the committed full profile also the
    charts in docs/figures and the generated block in README.md."""
    from pricing_crosscheck import report

    (directory / "summary.md").write_text(report.summary_markdown(tables, profile), encoding="utf-8")
    if publish:
        report.charts(tables, env.REPO_ROOT / "docs" / "figures")
        (env.REPO_ROOT / "docs" / "findings.md").write_text(report.findings_markdown(tables), encoding="utf-8")
        report.update_readme(tables, env.REPO_ROOT / "README.md")


def run_stages(profile: str, jobs: int = 1, out_root: Path | None = None) -> Iterator[Table]:
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}")
    ctx: dict[str, Any] = {}
    for stage in STAGES:
        for table in stage(profile, jobs, out_root or env.REPO_ROOT / "results", ctx):
            ctx[table.name] = table
            yield table
