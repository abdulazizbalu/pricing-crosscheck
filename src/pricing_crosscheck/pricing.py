"""Price every grid point with every engine of protocol section 4."""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Iterable, Sequence

from pricing_crosscheck import spec
from pricing_crosscheck.engines import own
from pricing_crosscheck.engines import quantlib as qle
from pricing_crosscheck.engines.quantlib import EngineResult
from pricing_crosscheck.results import Table
from pricing_crosscheck.spec import BarrierPoint, VanillaPoint

INSTRUMENTS = ("european", "american", "digital", "barrier")
PRICE_COLUMNS = [
    "instrument", "point_id", "option", "barrier_type", "barrier", "strike", "days", "vol", "rate", "div",
    "engine", "value", "std_error", "uncertainty", "status",
]


def engine_names(instrument: str) -> list[str]:
    return list(spec.load()["engines"][instrument])


def is_mc(instrument: str, engine: str) -> bool:
    cfg = spec.load()["engines"][instrument][engine]
    return engine.endswith("_mc") or engine.endswith("_mc_lsm") or cfg.get("class", "").startswith("MC")


def samples_of(instrument: str, engine: str, overrides: dict[str, Any] | None = None) -> int | None:
    cfg = {**spec.load()["engines"][instrument][engine], **(overrides or {})}
    return int(cfg["samples"]) if "samples" in cfg else None


def subset_ids(instrument: str, engine: str, profile: str) -> set[str] | None:
    """Point ids an engine is restricted to (MC subsets), or None for the full grid."""
    grid = spec.load()["engines"][instrument][engine].get("grid")
    if grid is None:
        return None
    if instrument == "barrier":
        return {p.point_id for p in spec.barrier_points(profile, grid)}
    return {p.point_id for p in spec.vanilla_points(profile, grid)}


def seed(instrument: str, point_id: str, engine: str, samples: int | None, run: str = "") -> int:
    tag = spec.run_tag(samples=samples) + (f";{run}" if run else "")
    return spec.seed_for(instrument, point_id, engine, tag)


def price(
    instrument: str,
    point: VanillaPoint | BarrierPoint,
    engine: str,
    overrides: dict[str, Any] | None = None,
    run: str = "",
) -> EngineResult:
    """One engine at one point; MC seeds follow protocol section 12."""
    s = None
    if is_mc(instrument, engine):
        s = seed(instrument, point.point_id, engine, samples_of(instrument, engine, overrides), run)
    if engine.startswith("ql_"):
        if instrument == "barrier":
            return qle.price_barrier(point, engine, s, overrides)
        return qle.price_vanilla(point, instrument, engine, s, overrides)
    if instrument == "barrier":
        return own.price_barrier(point, engine, s, overrides)
    return own.price_vanilla(point, instrument, engine, s, overrides)


def _row(instrument: str, p: VanillaPoint | BarrierPoint, engine: str, r: EngineResult) -> dict[str, Any]:
    is_barrier = isinstance(p, BarrierPoint)
    return {
        "instrument": instrument,
        "point_id": p.point_id,
        "option": p.option,
        "barrier_type": p.barrier_type if is_barrier else "",
        "barrier": p.barrier if is_barrier else "",
        "strike": p.strike,
        "days": p.days,
        "vol": p.vol,
        "rate": p.rate,
        "div": p.div,
        "engine": engine,
        "value": r.value,
        "std_error": r.std_error,
        "uncertainty": r.uncertainty,
        "status": r.status,
    }


def _price_point(task: tuple[str, VanillaPoint | BarrierPoint, tuple[str, ...]]) -> list[dict[str, Any]]:
    instrument, point, engines = task
    return [_row(instrument, point, e, price(instrument, point, e)) for e in engines]


def points_for(instrument: str, profile: str) -> list[VanillaPoint | BarrierPoint]:
    if instrument == "barrier":
        return spec.barrier_points(profile)
    return spec.vanilla_points(profile)


REFERENCES = {"european": "cf", "digital": "cf", "barrier": "cf", "american": "ref_pde"}


def subset_points(instrument: str, engine: str, profile: str) -> list[VanillaPoint | BarrierPoint]:
    grid = spec.load()["engines"][instrument][engine]["grid"]
    if instrument == "barrier":
        return spec.barrier_points(profile, grid)
    return spec.vanilla_points(profile, grid)


def tasks_for(profile: str, instruments: Iterable[str] = INSTRUMENTS) -> list[tuple[str, Any, tuple[str, ...]]]:
    """Engines without a ``grid`` run on the main grid; engines with one run on
    that subset only. The reference runs on both, so every engine value has a
    reference at the same point (subsets need not lie inside the main grid)."""
    tasks = []
    for instrument in instruments:
        names = engine_names(instrument)
        ref = REFERENCES[instrument]
        subsets = {e: subset_ids(instrument, e, profile) for e in names}
        main = points_for(instrument, profile)
        main_ids = {p.point_id for p in main}
        extra: dict[str, VanillaPoint | BarrierPoint] = {}
        for e in names:
            if subsets[e] is not None:
                for p in subset_points(instrument, e, profile):
                    if p.point_id not in main_ids:
                        extra.setdefault(p.point_id, p)
        for p in main + [extra[k] for k in sorted(extra)]:
            in_main = p.point_id in main_ids
            engines = tuple(
                e for e in names
                if (subsets[e] is None and (in_main or e == ref)) or (subsets[e] is not None and p.point_id in subsets[e])
            )
            tasks.append((instrument, p, engines))
    return tasks


def parallel_map(fn: Callable[[Any], Any], items: Sequence[Any], jobs: int) -> list[Any]:
    """Order-preserving map; results do not depend on ``jobs``."""
    if jobs <= 1:
        return [fn(x) for x in items]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(fn, items, chunksize=1))


def price_table(profile: str, jobs: int = 1) -> Table:
    table = Table("prices", PRICE_COLUMNS)
    for rows in parallel_map(_price_point, tasks_for(profile), jobs):
        for row in rows:
            table.add(**row)
    return table


def by_key(table: Table) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Index price rows by (instrument, point_id, engine)."""
    return {(r["instrument"], r["point_id"], r["engine"]): r for r in table.rows}


def usable(row: dict[str, Any] | None) -> bool:
    return row is not None and row["status"] == "ok" and math.isfinite(row["value"])
