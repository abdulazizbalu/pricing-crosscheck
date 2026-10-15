"""Reading, writing and comparing the CSV files in results/."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import numbers
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

ENCODING = "utf-8"


@dataclass
class Table:
    """A named result table: column order is fixed, rows are dicts."""

    name: str
    columns: list[str]
    rows: list[dict[str, Any]] = field(default_factory=list)

    def add(self, **row: Any) -> None:
        missing = set(self.columns) - set(row)
        extra = set(row) - set(self.columns)
        if missing or extra:
            raise ValueError(f"{self.name}: missing {sorted(missing)}, unexpected {sorted(extra)}")
        self.rows.append(row)


def _format(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool) or getattr(getattr(value, "dtype", None), "kind", "") == "b":
        return "true" if value else "false"
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        value = float(value)
        if math.isnan(value):
            return "nan"
        return repr(value)
    return str(value)


def write_table(table: Table, directory: Path, header: Iterable[str]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{table.name}.csv"
    with path.open("w", encoding=ENCODING, newline="") as handle:
        for line in header:
            handle.write(line + "\n")
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(table.columns)
        for row in table.rows:
            writer.writerow([_format(row[c]) for c in table.columns])
    return path


def read_table(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding=ENCODING, newline="") as handle:
        lines = handle.readlines()
    start = 0
    while start < len(lines) and lines[start].startswith("#"):
        start += 1  # only the leading header block is metadata
    lines = lines[start:]
    reader = csv.DictReader(lines)
    rows = list(reader)
    return list(reader.fieldnames or []), rows


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(
    directory: Path,
    content_hash: str,
    versions: dict[str, str],
    profile: str,
    distributions: dict[str, str] | None = None,
) -> Path:
    files = {p.name: file_sha256(p) for p in sorted(directory.glob("*.csv"))}
    manifest = {
        "profile": profile,
        "content_hash": content_hash,
        "versions": versions,
        "installed_distributions": distributions or {},
        "files": files,
    }
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding=ENCODING)
    return path


def manifest_problems(directory: Path) -> list[str]:
    """Check that the CSV files in ``directory`` are exactly those in its manifest."""
    manifest = read_manifest(directory)
    if manifest is None:
        return [f"{directory}: no manifest.json"]
    recorded = manifest.get("files", {})
    present = {p.name: file_sha256(p) for p in sorted(directory.glob("*.csv"))}
    problems = []
    for name in sorted(set(recorded) | set(present)):
        if name not in present:
            problems.append(f"{name}: listed in manifest but missing")
        elif name not in recorded:
            problems.append(f"{name}: present but not listed in manifest")
        elif recorded[name] != present[name]:
            problems.append(f"{name}: content differs from the manifest checksum")
    return problems


def read_manifest(directory: Path) -> dict[str, Any] | None:
    path = directory / "manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding=ENCODING))


def _as_float(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def compare_tables(expected: Path, actual: Path, rtol: float, atol: float) -> list[str]:
    """Return human-readable differences between two result CSV files.

    Numeric cells must agree within ``atol + rtol * max(|a|, |b|)``; other
    cells must match exactly. Comment lines (versions, hash) are ignored.
    """
    problems: list[str] = []
    cols_e, rows_e = read_table(expected)
    cols_a, rows_a = read_table(actual)
    if cols_e != cols_a:
        return [f"{expected.name}: columns differ"]
    if len(rows_e) != len(rows_a):
        return [f"{expected.name}: {len(rows_e)} rows expected, {len(rows_a)} found"]
    for i, (re_, ra) in enumerate(zip(rows_e, rows_a)):
        for col in cols_e:
            a, b = re_[col], ra[col]
            if a == b:
                continue
            fa, fb = _as_float(a), _as_float(b)
            if fa is None or fb is None:
                problems.append(f"{expected.name} row {i} {col}: {a!r} != {b!r}")
                continue
            if math.isnan(fa) and math.isnan(fb):
                continue
            if not (math.isfinite(fa) and math.isfinite(fb)):
                if fa != fb:
                    problems.append(f"{expected.name} row {i} {col}: {a} != {b}")
                continue
            if not abs(fa - fb) <= atol + rtol * max(abs(fa), abs(fb)):
                problems.append(f"{expected.name} row {i} {col}: {a} != {b}")
    return problems
