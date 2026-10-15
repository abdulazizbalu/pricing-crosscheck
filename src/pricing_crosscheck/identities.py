"""Identity checks and observations of protocol section 9."""

from __future__ import annotations

import math
from typing import Any

from pricing_crosscheck import spec
from pricing_crosscheck.pricing import by_key, is_mc, parallel_map, price, tasks_for, usable
from pricing_crosscheck.reference import closed_form as cf
from pricing_crosscheck.results import Table
from pricing_crosscheck.spec import BarrierPoint, VanillaPoint

IDENTITY_COLUMNS = [
    "identity", "instrument", "engine", "point_id", "lhs", "rhs", "residual", "tol", "passed", "informative", "note",
]
EXACT_ENGINES = ("cf", "ql_analytic")
# Pre-declared in protocol section 9: QuantLib's FD knock-in is consistent with
# vanilla minus knock-out on the same grid, so in-out parity carries no information.
BY_CONSTRUCTION = {("in_out_parity", "ql_fd")}
MACHINE_LEVEL = 1e-12


def _tol(instrument: str, engine: str, magnitude: float, std_errors: list[float]) -> float:
    if engine in EXACT_ENGINES:
        return spec.tolerance("exact", magnitude)
    tol = spec.tolerance("numerical", magnitude, instrument)
    if is_mc(instrument, engine):
        sigmas = spec.load()["tolerances"]["mc_sigmas"]
        tol += sigmas * sum(s for s in std_errors if math.isfinite(s))
    return tol


def _note(identity: str, engine: str, instrument: str, residual: float, magnitude: float) -> tuple[bool, str]:
    if (identity, engine) in BY_CONSTRUCTION:
        return False, "holds by construction (declared in protocol section 9)"
    if engine not in EXACT_ENGINES and abs(residual) <= MACHINE_LEVEL * max(1.0, magnitude):
        return False, "holds to machine precision for a numerical engine (not informative, protocol section 9)"
    return True, ""


def _add(table: Table, identity: str, instrument: str, engine: str, point_id: str, lhs: float, rhs: float,
         tol: float, passed: bool, magnitude: float) -> None:
    residual = lhs - rhs
    informative, note = _note(identity, engine, instrument, residual, magnitude)
    table.add(identity=identity, instrument=instrument, engine=engine, point_id=point_id, lhs=lhs, rhs=rhs,
              residual=residual, tol=tol, passed=passed, informative=informative, note=note)


def _other_option(p: VanillaPoint) -> VanillaPoint:
    return VanillaPoint("put" if p.option == "call" else "call", p.strike, p.days, p.vol, p.rate, p.div, p.spot)


def _vanilla_counterpart(task: tuple[str, VanillaPoint]) -> tuple[str, str, float]:
    engine, p = task
    r = price("european", p, engine)
    return engine, p.point_id, r.value if r.ok else float("nan")


def identity_table(prices: Table, profile: str, jobs: int = 1) -> Table:
    idx = by_key(prices)
    table = Table("identities", IDENTITY_COLUMNS)
    engines = {inst: [e for e in spec.load()["engines"][inst]] for inst in ("european", "digital", "american", "barrier")}

    # Put-call parity and digital call + put.
    for p in spec.vanilla_points(profile):
        if p.option != "call":
            continue
        q = _other_option(p)
        fwd_s, pv_k = p.spot * math.exp(-p.div * p.T), p.strike * math.exp(-p.rate * p.T)
        for e in engines["european"]:
            c, pu = idx.get(("european", p.point_id, e)), idx.get(("european", q.point_id, e))
            if usable(c) and usable(pu):
                mag = max(abs(c["value"]), abs(pu["value"]), fwd_s, pv_k)
                tol = _tol("european", e, mag, [c["std_error"], pu["std_error"]])
                lhs = c["value"] - pu["value"]
                _add(table, "put_call_parity", "european", e, p.point_id, lhs, fwd_s - pv_k,
                     tol, abs(lhs - (fwd_s - pv_k)) <= tol, mag)
        df = math.exp(-p.rate * p.T)
        for e in engines["digital"]:
            c, pu = idx.get(("digital", p.point_id, e)), idx.get(("digital", q.point_id, e))
            if usable(c) and usable(pu):
                mag = max(abs(c["value"]), abs(pu["value"]), df)
                tol = _tol("digital", e, mag, [c["std_error"], pu["std_error"]])
                lhs = c["value"] + pu["value"]
                _add(table, "digital_call_plus_put", "digital", e, p.point_id, lhs, df, tol, abs(lhs - df) <= tol, mag)

    # American >= European and the no-early-exercise cases (main grid and MC subset).
    for _, p, _ in tasks_for(profile, ["american"]):
        euro = cf.european(p.spot, p.strike, p.T, p.rate, p.div, p.vol, p.option)
        no_exercise = (p.option == "call" and p.div == 0.0 and p.rate >= 0.0) or (
            p.option == "put" and p.rate <= 0.0 and p.div >= 0.0
        )
        for e in engines["american"]:
            a = idx.get(("american", p.point_id, e))
            if not usable(a):
                continue
            mag = max(abs(a["value"]), abs(euro))
            tol = _tol("american", e, mag, [a["std_error"]])
            _add(table, "american_geq_european", "american", e, p.point_id, a["value"], euro, tol,
                 a["value"] >= euro - tol, mag)
            if no_exercise:
                name = f"no_early_exercise_{p.option}"
                _add(table, name, "american", e, p.point_id, a["value"], euro, tol, abs(a["value"] - euro) <= tol, mag)

    # In-out parity; vanilla from the same engine family (protocol section 9).
    barrier_pts = [b for b in spec.barrier_points(profile) if not b.is_knock_in]
    family = {"ql_analytic": "ql_analytic", "ql_crr": "ql_crr", "ql_fd": "ql_fd"}
    tasks = sorted({(family[e], b.vanilla()) for b in barrier_pts for e in engines["barrier"] if e in family},
                   key=lambda t: (t[0], t[1].point_id))
    vanilla = {(e, pid): v for e, pid, v in parallel_map(_vanilla_counterpart, tasks, jobs)}
    for b in barrier_pts:
        ki_id = BarrierPoint(b.barrier_type.replace("-out", "-in"), b.option, b.barrier, b.strike, b.days, b.vol,
                             b.rate, b.div, b.spot).point_id
        euro = cf.european(b.spot, b.strike, b.T, b.rate, b.div, b.vol, b.option)
        for e in engines["barrier"]:
            ko, ki = idx.get(("barrier", b.point_id, e)), idx.get(("barrier", ki_id, e))
            if not (usable(ko) and usable(ki)):
                continue
            van = vanilla.get((family[e], b.vanilla().point_id), float("nan")) if e in family else euro
            if not math.isfinite(van):
                continue
            mag = max(abs(ko["value"]), abs(ki["value"]), abs(van))
            tol = _tol("barrier", e, mag, [ko["std_error"], ki["std_error"]])
            lhs = ko["value"] + ki["value"]
            _add(table, "in_out_parity", "barrier", e, b.point_id, lhs, van, tol, abs(lhs - van) <= tol, mag)
    return table


CALL_SPREAD_COLUMNS = ["engine", "point_id", "h", "spread", "digital_cf", "difference"]


def _spread_value(task: tuple[str, VanillaPoint, float]) -> tuple[str, str, float, float]:
    engine, p, h = task
    lo = VanillaPoint(p.option, p.strike - h, p.days, p.vol, p.rate, p.div, p.spot)
    hi = VanillaPoint(p.option, p.strike + h, p.days, p.vol, p.rate, p.div, p.spot)
    if engine == "cf":
        v_lo = cf.european(lo.spot, lo.strike, lo.T, lo.rate, lo.div, lo.vol, lo.option)
        v_hi = cf.european(hi.spot, hi.strike, hi.T, hi.rate, hi.div, hi.vol, hi.option)
    else:
        v_lo, v_hi = price("european", lo, engine).value, price("european", hi, engine).value
    sign = 1.0 if p.option == "call" else -1.0
    return engine, p.point_id, h, sign * (v_lo - v_hi) / (2 * h)


def call_spread_table(jobs: int = 1) -> Table:
    """Observation (protocol section 9): digital as the limit of a call (put) spread."""
    s = spec.load()
    points = [VanillaPoint(o, c["strike"], c["days"], c["vol"], c["rate"], c["div"])
              for c in s["convergence"]["points"] for o in ("call", "put")]
    tasks = [(e, p, h) for e in ("cf", "ql_analytic", "ql_fd") for p in points for h in (1.0, 0.1, 0.01, 0.001)]
    table = Table("call_spread", CALL_SPREAD_COLUMNS)
    pmap = {p.point_id: p for p in points}
    for engine, pid, h, spread in parallel_map(_spread_value, tasks, jobs):
        p = pmap[pid]
        dig = cf.digital(p.spot, p.strike, p.T, p.rate, p.div, p.vol, p.option)
        table.add(engine=engine, point_id=pid, h=h, spread=spread, digital_cf=dig, difference=spread - dig)
    return table


def summarise(table: Table) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for r in table.rows:
        k = (r["identity"], r["engine"])
        d = out.setdefault(k, {"n": 0, "failed": 0, "informative": r["informative"], "max_abs_residual": 0.0})
        d["n"] += 1
        d["failed"] += 0 if r["passed"] else 1
        d["max_abs_residual"] = max(d["max_abs_residual"], abs(r["residual"]))
    return out
