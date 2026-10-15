"""Greeks: analytic, QuantLib engine Greeks and bump-and-revalue (protocol section 8)."""

from __future__ import annotations

import math
from typing import Any

from pricing_crosscheck import spec
from pricing_crosscheck.engines import quantlib as qle
from pricing_crosscheck.pricing import parallel_map
from pricing_crosscheck.reference import closed_form as cf
from pricing_crosscheck.results import Table
from pricing_crosscheck.spec import VanillaPoint

GREEKS = ("delta", "gamma", "vega", "theta")
GREEK_COLUMNS = [
    "instrument", "point_id", "method", "greek", "value", "reference", "diff", "tol", "disagree", "labels",
]
SWEEP_COLUMNS = ["instrument", "point_id", "greek", "eps", "value", "reference", "abs_error"]
PDE_IDENTITY_COLUMNS = ["instrument", "point_id", "engine", "residual", "scale"]
QL_GREEK_ENGINES = {"european": ("ql_analytic", "ql_crr", "ql_fd", "ql_fd_damped"),
                    "digital": ("ql_analytic", "ql_fd", "ql_fd_damped")}


def _price_fn(instrument: str):
    return cf.european if instrument == "european" else cf.digital


def _greeks_fn(instrument: str):
    return cf.european_greeks if instrument == "european" else cf.digital_greeks


def bumped(instrument: str, p: VanillaPoint, bumps: dict[str, Any]) -> dict[str, float]:
    """Bump-and-revalue on the closed form with the documented bump set."""
    f = _price_fn(instrument)
    S, K, T, r, q, v, o = p.spot, p.strike, p.T, p.rate, p.div, p.vol, p.option
    hs = bumps["spot_rel"] * S
    hv = bumps["vol_abs"]
    up, mid, dn = f(S + hs, K, T, r, q, v, o), f(S, K, T, r, q, v, o), f(S - hs, K, T, r, q, v, o)
    out = {
        "delta": (up - dn) / (2 * hs),
        "gamma": (up - 2 * mid + dn) / (hs * hs),
        "vega": (f(S, K, T, r, q, v + hv, o) - f(S, K, T, r, q, max(v - hv, 0.0), o)) / (v + hv - max(v - hv, 0.0)),
    }
    if "time_days" in bumps:
        h = bumps["time_days"] / 365.0
        out["theta"] = (f(S, K, max(T - h, 0.0), r, q, v, o) - mid) / h
    else:
        h = bumps["time_rel"] * T
        out["theta"] = -(f(S, K, T + h, r, q, v, o) - f(S, K, T - h, r, q, v, o)) / (2 * h)
    return out


def _labels(p: VanillaPoint) -> str:
    from pricing_crosscheck.compare import labels

    return ";".join(labels({"days": p.days, "vol": p.vol, "rate": p.rate, "div": p.div, "strike": p.strike,
                            "barrier": ""}))


def _point_rows(task: tuple[str, VanillaPoint]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    instrument, p = task
    s = spec.load()
    ref = _greeks_fn(instrument)(p.spot, p.strike, p.T, p.rate, p.div, p.vol, p.option)
    lab = _labels(p)
    rows, ident = [], []

    def add(method: str, values: dict[str, float], exact: bool) -> None:
        for g in GREEKS:
            x = values.get(g, float("nan"))
            if not math.isfinite(x):
                continue
            tol = spec.tolerance("exact", ref[g]) if exact else spec.greek_tolerance(g, ref[g], instrument)
            d = x - ref[g]
            rows.append({"instrument": instrument, "point_id": p.point_id, "method": method, "greek": g, "value": x,
                         "reference": ref[g], "diff": d, "tol": tol, "disagree": abs(d) > tol, "labels": lab})

    for e in QL_GREEK_ENGINES[instrument]:
        r = qle.price_vanilla(p, instrument, e, want_greeks=True)
        if not r.ok:
            continue
        add(e, r.greeks, exact=(e == "ql_analytic"))
        g = r.greeks
        if all(math.isfinite(g[k]) for k in ("delta", "gamma", "theta")):
            terms = [g["theta"], 0.5 * p.vol**2 * p.spot**2 * g["gamma"], (p.rate - p.div) * p.spot * g["delta"],
                     -p.rate * r.value]
            ident.append({"instrument": instrument, "point_id": p.point_id, "engine": e, "residual": sum(terms),
                          "scale": max(abs(t) for t in terms)})
    for name, bumps in s["greeks"]["bumps"].items():
        add(f"bump_{name}", bumped(instrument, p, bumps), exact=False)
    return rows, ident


def greeks_tables(profile: str, jobs: int = 1) -> tuple[Table, Table, Table]:
    tasks = [(inst, p) for inst in s_instruments() for p in spec.vanilla_points(profile)]
    out = parallel_map(_point_rows, tasks, jobs)
    greeks = Table("greeks", GREEK_COLUMNS)
    ident = Table("greeks_pde_identity", PDE_IDENTITY_COLUMNS)
    for rows, idrows in out:
        for r in rows:
            greeks.add(**r)
        for r in idrows:
            ident.add(**r)
    return greeks, ident, sweep_table()


def s_instruments() -> list[str]:
    return list(spec.load()["greeks"]["instruments"])


def sweep_table() -> Table:
    s = spec.load()["greeks"]
    table = Table("greeks_sweep", SWEEP_COLUMNS)
    for inst in s_instruments():
        f, gf = _price_fn(inst), _greeks_fn(inst)
        for sp in s["sweep_points"]:
            p = VanillaPoint("call", sp["strike"], sp["days"], sp["vol"], sp["rate"], sp["div"])
            S, K, T, r, q, v = p.spot, p.strike, p.T, p.rate, p.div, p.vol
            ref = gf(S, K, T, r, q, v, "call")
            for eps in s["sweep_eps"]:
                hs, hv, ht = eps * S, eps * v, eps * T
                mid = f(S, K, T, r, q, v, "call")
                up, dn = f(S + hs, K, T, r, q, v, "call"), f(S - hs, K, T, r, q, v, "call")
                vals = {
                    "delta": (up - dn) / (2 * hs),
                    "gamma": (up - 2 * mid + dn) / (hs * hs),
                    "vega": (f(S, K, T, r, q, v + hv, "call") - f(S, K, T, r, q, v - hv, "call")) / (2 * hv),
                    "theta": -(f(S, K, T + ht, r, q, v, "call") - f(S, K, T - ht, r, q, v, "call")) / (2 * ht),
                }
                for g in GREEKS:
                    table.add(instrument=inst, point_id=f"{sp['id']}|{p.point_id}", greek=g, eps=eps,
                              value=vals[g], reference=ref[g], abs_error=abs(vals[g] - ref[g]))
    return table
