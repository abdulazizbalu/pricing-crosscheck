"""Differences to the reference and per-engine summaries (protocol sections 5, 6, 11)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from pricing_crosscheck import spec
from pricing_crosscheck.pricing import REFERENCES, is_mc, usable
from pricing_crosscheck.results import Table

EXACT_ENGINES = {"ql_analytic"}
APPROXIMATIONS = {"ql_baw", "ql_bjs"}
SELF_CONVERGENCE = {("american", "own_pde")}  # shares code with ref_pde (protocol section 5, rule 4)

COMPARISON_COLUMNS = [
    "instrument", "point_id", "option", "barrier_type", "barrier", "strike", "days", "vol", "rate", "div",
    "engine", "value", "reference", "diff", "rel_diff", "std_error", "z", "tol", "ref_uncertainty",
    "status", "excluded", "disagree", "labels",
]


def _f(x: Any) -> float:
    if x in ("", None):
        return float("nan")
    return float(x)


def tolerance_class(instrument: str, engine: str) -> str:
    if engine in EXACT_ENGINES:
        return "exact"
    if is_mc(instrument, engine):
        return "mc"
    return "numerical"


def labels(row: dict[str, Any]) -> list[str]:
    lab = spec.load()["labels"]
    S = 100.0
    out = []
    days, vol, rate, div, K = int(row["days"]), float(row["vol"]), float(row["rate"]), float(row["div"]), float(row["strike"])
    T = days / 365.0
    if days <= lab["near_expiry_max_days"]:
        out.append("near_expiry")
    if vol <= lab["near_zero_vol_max"]:
        out.append("near_zero_vol")
    if row.get("barrier") not in ("", None) and abs(float(row["barrier"]) / S - 1.0) <= lab["near_barrier_max_rel_distance"]:
        out.append("near_barrier")
    fwd = S * math.exp((rate - div) * T)
    if abs(math.log(K / fwd)) / (vol * math.sqrt(T)) > lab["deep_moneyness_min_sd"]:
        out.append("deep_moneyness")
    if rate < 0:
        out.append("negative_rate")
    return out


def comparison_table(prices: Table) -> Table:
    s = spec.load()
    sigmas = s["tolerances"]["mc_sigmas"]
    frac = s["tolerances"]["reference_uncertainty_fraction"]
    refs = {(r["instrument"], r["point_id"]): r for r in prices.rows if r["engine"] == REFERENCES[r["instrument"]]}
    table = Table("comparisons", COMPARISON_COLUMNS)
    for r in prices.rows:
        inst, eng = r["instrument"], r["engine"]
        if eng == REFERENCES[inst]:
            continue
        ref = refs[(inst, r["point_id"])]
        refv = ref["value"]
        u = _f(ref["uncertainty"])
        cls = tolerance_class(inst, eng)
        if cls == "exact":
            tol = spec.tolerance("exact", refv)
        else:
            tol = spec.tolerance("numerical", refv, inst)
        se = _f(r["std_error"])
        ok = usable(r)
        diff = r["value"] - refv if ok else float("nan")
        if cls == "mc" and ok:
            tol += sigmas * se
        excluded = bool(math.isfinite(u) and u > frac * spec.tolerance("numerical", refv, inst))
        disagree = bool(ok and not excluded and abs(diff) > tol)
        rel = diff / abs(refv) if ok and abs(refv) > 1e-12 else float("nan")
        # z is undefined when every antithetic pair gives the same payoff (SE at rounding level)
        z = diff / se if ok and cls == "mc" and se > 1e-12 else float("nan")
        table.add(
            instrument=inst, point_id=r["point_id"], option=r["option"], barrier_type=r["barrier_type"],
            barrier=r["barrier"], strike=r["strike"], days=r["days"], vol=r["vol"], rate=r["rate"], div=r["div"],
            engine=eng, value=r["value"], reference=refv, diff=diff, rel_diff=rel, std_error=se, z=z, tol=tol,
            ref_uncertainty=u, status=r["status"], excluded=excluded, disagree=disagree,
            labels=";".join(labels(r)),
        )
    return table


SUMMARY_COLUMNS = [
    "instrument", "engine", "points", "not_available", "excluded", "compared", "disagreements",
    "median_abs_diff", "p95_abs_diff", "max_abs_diff", "max_abs_rel_diff", "max_abs_z", "note",
]


def summary_table(comparisons: Table) -> Table:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in comparisons.rows:
        groups.setdefault((r["instrument"], r["engine"]), []).append(r)
    order = {i: n for n, i in enumerate(("european", "digital", "barrier", "american"))}
    table = Table("summary", SUMMARY_COLUMNS)
    for (inst, eng), rows in sorted(groups.items(), key=lambda kv: (order[kv[0][0]], kv[0][1])):
        avail = [r for r in rows if r["status"] == "ok"]
        used = [r for r in avail if not r["excluded"]]
        d = np.array([abs(r["diff"]) for r in used]) if used else np.array([])
        rel = [abs(r["rel_diff"]) for r in used if math.isfinite(r["rel_diff"])]
        z = [abs(r["z"]) for r in used if math.isfinite(r["z"])]
        note = ""
        if (inst, eng) in SELF_CONVERGENCE:
            note = "self-convergence check (shares code with ref_pde)"
        elif eng in APPROXIMATIONS:
            note = "analytic approximation"
        table.add(
            instrument=inst, engine=eng, points=len(rows), not_available=len(rows) - len(avail),
            excluded=len(avail) - len(used), compared=len(used),
            disagreements=sum(1 for r in used if r["disagree"]),
            median_abs_diff=float(np.median(d)) if len(d) else float("nan"),
            p95_abs_diff=float(np.percentile(d, 95)) if len(d) else float("nan"),
            max_abs_diff=float(d.max()) if len(d) else float("nan"),
            max_abs_rel_diff=max(rel) if rel else float("nan"),
            max_abs_z=max(z) if z else float("nan"),
            note=note,
        )
    return table
