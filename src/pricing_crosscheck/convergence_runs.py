"""Convergence measurements of protocol section 10."""

from __future__ import annotations

import math
from typing import Any

from pricing_crosscheck import spec
from pricing_crosscheck.convergence import fitted_order
from pricing_crosscheck.engines.own import american_reference
from pricing_crosscheck.pricing import parallel_map, price
from pricing_crosscheck.reference import closed_form as cf
from pricing_crosscheck.results import Table
from pricing_crosscheck.spec import BarrierPoint, VanillaPoint

CONVERGENCE_COLUMNS = [
    "experiment", "instrument", "point_id", "engine", "variant", "resolution", "value", "reference", "error",
    "std_error", "status",
]
ORDER_COLUMNS = [
    "experiment", "instrument", "point_id", "engine", "variant", "fitted_order", "levels_used", "local_orders",
    "last_abs_error", "measurable",
]


def _vanilla_points(profile: str) -> list[VanillaPoint]:
    pts = [VanillaPoint(o, c["strike"], c["days"], c["vol"], c["rate"], c["div"])
           for c in spec.load()["convergence"]["points"] for o in ("call", "put")]
    return pts[:1] if profile == "fast" else pts


def _barrier_points(profile: str) -> list[BarrierPoint]:
    pts = [BarrierPoint(c["barrier_type"], c["option"], c["barrier"], c["strike"], c["days"], c["vol"], c["rate"],
                        c["div"]) for c in spec.load()["convergence"]["barrier_points"]]
    return pts[:1] if profile == "fast" else pts


def _reference(instrument: str, p) -> float:
    a = (p.spot, p.strike, p.T, p.rate, p.div, p.vol, p.option)
    if instrument == "european":
        return cf.european(*a)
    if instrument == "digital":
        return cf.digital(*a)
    if instrument == "american":
        return american_reference(p)[0]
    return cf.barrier(p.spot, p.strike, p.barrier, p.T, p.rate, p.div, p.vol, p.option, p.barrier_type)


def _ladder_tasks(profile: str) -> list[tuple]:
    c = spec.load()["convergence"]
    steps, nodes = c["tree_steps"], c["pde_nodes"]
    samples = [2**k for k in c["mc_samples_log2"]]
    per_year = c["mc_monitoring_steps_per_year"]
    tasks: list[tuple] = []
    for p in _vanilla_points(profile):
        for inst, engines in (("european", ("ql_crr", "ql_lr", "own_crr")), ("american", ("ql_crr", "own_crr"))):
            for e in engines:
                # QuantLib's LR tree is used with odd step counts (protocol deviation 1);
                # even counts are a separate experiment (experiments.lr_even_odd).
                ladder = [n if n % 2 or e != "ql_lr" else n + 1 for n in steps]
                tasks += [("tree", inst, p, e, "", n, {"steps": n}) for n in ladder]
        for inst in ("european", "digital", "american"):
            # own_pde is not an American engine in the spec (ref_pde is the American PDE).
            for e in ("ql_fd", "ql_fd_damped") + (("own_pde",) if inst != "american" else ()):
                tasks += [("pde", inst, p, e, "", n,
                           {"tGrid": n, "xGrid": n} if e.startswith("ql") else {"space_nodes": n, "time_steps": n})
                          for n in nodes]
        for inst, engines in (("european", ("ql_mc", "own_mc")), ("digital", ("own_mc",))):
            for e in engines:
                tasks += [("mc", inst, p, e, "", n, {"samples": n}) for n in samples]
    for b in _barrier_points(profile):
        tasks += [("tree", "barrier", b, "ql_crr", "", n, {"steps": n}) for n in steps]
        for e in ("ql_fd", "own_pde"):
            tasks += [("pde", "barrier", b, e, "", n,
                       {"tGrid": n, "xGrid": n} if e == "ql_fd" else {"space_nodes": n, "time_steps": n})
                      for n in nodes]
        for m in per_year:
            spd = m / 365.0
            for corrected in (True, False):
                variant = "corrected" if corrected else "uncorrected"
                tasks.append(("monitoring", "barrier", b, "ql_mc", variant, m,
                              {"steps_per_day": spd, "isBiased": not corrected}))
                tasks.append(("monitoring", "barrier", b, "own_mc", variant, m,
                              {"steps_per_day": spd, "bridge_correction": corrected}))
    return tasks


def _run(task: tuple) -> dict[str, Any]:
    experiment, inst, p, engine, variant, resolution, overrides = task
    run = f"convergence;{experiment};{variant};{resolution}"
    r = price(inst, p, engine, overrides=overrides, run=run)
    ref = _reference(inst, p)
    return {
        "experiment": experiment, "instrument": inst, "point_id": p.point_id, "engine": engine, "variant": variant,
        "resolution": resolution, "value": r.value, "reference": ref,
        "error": r.value - ref if r.ok else float("nan"), "std_error": r.std_error, "status": r.status,
    }


def convergence_tables(profile: str, jobs: int = 1) -> tuple[Table, Table]:
    rows = parallel_map(_run, _ladder_tasks(profile), jobs)
    conv = Table("convergence", CONVERGENCE_COLUMNS)
    for row in rows:
        conv.add(**row)
    orders = Table("orders", ORDER_COLUMNS)
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["experiment"], row["instrument"], row["point_id"], row["engine"], row["variant"])
        groups.setdefault(key, []).append(row)
    for key, grp in groups.items():
        if key[0] in ("mc", "monitoring"):
            continue  # MC: error and SE reported per level; no order fit (protocol section 7)
        ok = [g for g in grp if g["status"] == "ok" and math.isfinite(g["error"])]
        res = [g["resolution"] for g in ok]
        err = [g["error"] for g in ok]
        if key[0] == "tree" and key[3] in ("own_crr", "ql_crr") and key[1] != "barrier":
            # CRR orders are fitted on even step counts (protocol section 7)
            pairs = [(n, e) for n, e in zip(res, err) if n % 2 == 0]
            res, err = [n for n, _ in pairs], [e for _, e in pairs]
        fit = fitted_order(res, err, ok[0]["reference"] if ok else 1.0)
        orders.add(
            experiment=key[0], instrument=key[1], point_id=key[2], engine=key[3], variant=key[4],
            fitted_order=fit.order, levels_used=";".join(f"{x:g}" for x in fit.levels_used),
            local_orders=";".join(f"{x:.3f}" for x in fit.local_orders),
            last_abs_error=abs(err[-1]) if err else float("nan"), measurable=fit.measurable,
        )
    return conv, orders
