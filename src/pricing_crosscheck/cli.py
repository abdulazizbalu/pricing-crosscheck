"""Command line entry point: ``python -m pricing_crosscheck <command>``."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from importlib import metadata
from pathlib import Path

from pricing_crosscheck import env
from pricing_crosscheck.results import (
    compare_tables,
    manifest_problems,
    read_manifest,
    write_manifest,
    write_table,
)
from pricing_crosscheck.stages import PROFILES, run_stages

DEFAULT_RESULTS = env.REPO_ROOT / "results"

# Tolerances for the reproducibility check (re-run versus committed CSV).
# They absorb last-bit differences between CPUs (SIMD code paths in numpy and
# libm), not modelling differences.
CHECK_RTOL = 1e-7
CHECK_ATOL = 1e-10


def _distributions() -> dict[str, str]:
    return {d.metadata["Name"]: d.version for d in sorted(metadata.distributions(), key=lambda d: d.metadata["Name"].lower())}


def _require_locked_env(allow_unlocked: bool) -> None:
    problems = env.lock_mismatches()
    if problems and not allow_unlocked:
        for line in problems:
            print(line)
        raise SystemExit("installed packages differ from requirements.lock (use --allow-unlocked to override)")
    for line in problems:
        print(f"warning: {line}")


def _run(profile: str, out_root: Path, jobs: int) -> Path:
    env.check_repo_root()
    directory = out_root / profile
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    versions = env.library_versions()
    digest = env.content_hash(versions=versions)
    header = env.metadata_lines(profile)
    for table in run_stages(profile, jobs=jobs, out_root=out_root):
        write_table(table, directory, header)
    write_manifest(directory, digest, versions, profile, _distributions())
    return directory


def _is_current(directory: Path) -> bool:
    manifest = read_manifest(directory)
    return bool(manifest) and manifest.get("content_hash") == env.content_hash() and not manifest_problems(directory)


def cmd_run(args: argparse.Namespace) -> int:
    _require_locked_env(args.allow_unlocked)
    out_root = Path(args.out)
    if args.skip_if_current and _is_current(out_root / args.profile):
        print(f"{args.profile}: results are current, nothing to do")
        return 0
    directory = _run(args.profile, out_root, args.jobs)
    print(f"wrote {directory}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Compare committed results with results regenerated from the current code."""
    _require_locked_env(args.allow_unlocked)
    committed = Path(args.results) / args.profile
    if not committed.exists():
        print(f"no committed results in {committed}")
        return 0 if args.allow_missing else 1
    problems = manifest_problems(committed)
    manifest = read_manifest(committed) or {}
    if manifest.get("content_hash") != env.content_hash():
        problems.append("committed manifest content_hash does not match the current code/specs/versions")
    with tempfile.TemporaryDirectory() as tmp:
        out_root = Path(args.regenerated) if args.regenerated else Path(tmp)
        if out_root.resolve() == Path(args.results).resolve():
            print("--regenerated must differ from --results")
            return 2
        if args.regenerated and _is_current(out_root / args.profile):
            fresh = out_root / args.profile
            print(f"reusing cached regenerated results in {fresh} (same content hash and cache key)")
        else:
            fresh = _run(args.profile, out_root, args.jobs)
        names_c = sorted(p.name for p in committed.glob("*.csv"))
        names_f = sorted(p.name for p in fresh.glob("*.csv"))
        if names_c != names_f:
            problems.append(f"file sets differ: committed {names_c}, regenerated {names_f}")
        for name in sorted(set(names_c) & set(names_f)):
            problems += compare_tables(committed / name, fresh / name, CHECK_RTOL, CHECK_ATOL)
    for line in problems[:50]:
        print(line)
    if problems:
        print(f"{len(problems)} problem(s) comparing committed and regenerated {args.profile} results")
        return 1
    print(f"{args.profile}: regenerated results match the committed files")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    directory = Path(args.results) / args.profile
    manifest = read_manifest(directory)
    if manifest is None:
        print(f"{args.profile}: no committed results")
        return 0 if args.allow_missing else 1
    problems = manifest_problems(directory)
    current = env.content_hash()
    if manifest.get("content_hash") != current:
        problems.append(f"stale: committed {manifest.get('content_hash')}, current {current}")
    for line in problems:
        print(f"{args.profile}: {line}")
    if problems:
        return 1
    print(f"{args.profile}: current ({current}), files match the manifest")
    return 0


def cmd_verify_env(args: argparse.Namespace) -> int:
    problems = env.lock_mismatches()
    for line in problems:
        print(line)
    if problems:
        return 1
    print("installed packages match requirements.lock")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pricing_crosscheck")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="generate results for a profile")
    p_run.add_argument("--profile", choices=PROFILES, default="full")
    p_run.add_argument("--out", default=str(DEFAULT_RESULTS))
    p_run.add_argument("--jobs", type=int, default=1)
    p_run.add_argument("--skip-if-current", action="store_true")
    p_run.add_argument("--allow-unlocked", action="store_true", help="run even if packages differ from the lock file")
    p_run.set_defaults(func=cmd_run)

    p_check = sub.add_parser("check", help="regenerate a profile and compare with committed results")
    p_check.add_argument("--profile", choices=PROFILES, default="fast")
    p_check.add_argument("--results", default=str(DEFAULT_RESULTS))
    p_check.add_argument("--jobs", type=int, default=1)
    p_check.add_argument(
        "--regenerated",
        help="directory for the regenerated results; reused if its manifest is current (CI cache)",
    )
    p_check.add_argument("--allow-missing", action="store_true", help="succeed if nothing is committed yet")
    p_check.add_argument("--allow-unlocked", action="store_true")
    p_check.set_defaults(func=cmd_check)

    p_status = sub.add_parser("status", help="committed manifest current and files intact?")
    p_status.add_argument("--profile", choices=PROFILES, default="full")
    p_status.add_argument("--results", default=str(DEFAULT_RESULTS))
    p_status.add_argument("--allow-missing", action="store_true")
    p_status.set_defaults(func=cmd_status)

    sub.add_parser("verify-env", help="installed packages match requirements.lock?").set_defaults(func=cmd_verify_env)
    sub.add_parser("hash", help="print the content hash").set_defaults(
        func=lambda a: print(env.content_hash()) or 0
    )
    sub.add_parser("env", help="print library versions").set_defaults(
        func=lambda a: print(json.dumps(env.library_versions(), indent=2)) or 0
    )
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
