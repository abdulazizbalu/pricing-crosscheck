"""Disagreement map: refinement experiments and the ordered decision list of protocol section 11."""

from __future__ import annotations

import math
from typing import Any

from pricing_crosscheck import spec
from pricing_crosscheck.compare import APPROXIMATIONS
from pricing_crosscheck.diagnostics import LR_MECHANISM, lr_collapsed
from pricing_crosscheck.pricing import parallel_map, price, samples_of
from pricing_crosscheck.results import Table
from pricing_crosscheck.spec import BarrierPoint, VanillaPoint

DISAGREEMENT_COLUMNS = [
    "instrument", "point_id", "engine", "labels", "diff0", "tol0", "diff1", "diff2", "se0", "se1", "se2",
    "local_order", "category", "note",
]
NO_REFINEMENT = {"ql_qdfp", "ql_analytic"}
TREES = {"ql_crr", "ql_lr", "own_crr"}
FD = {"ql_fd", "ql_fd_damped"}
MC = {"ql_mc", "own_mc"}
LSM = {"ql_mc_lsm"}


def _point(row: dict[str, Any]) -> VanillaPoint | BarrierPoint:
    if row["instrument"] == "barrier":
        return BarrierPoint(row["barrier_type"], row["option"], float(row["barrier"]), float(row["strike"]),
                            int(row["days"]), float(row["vol"]), float(row["rate"]), float(row["div"]))
    return VanillaPoint(row["option"], float(row["strike"]), int(row["days"]), float(row["vol"]),
                        float(row["rate"]), float(row["div"]))


def _cfg(instrument: str, engine: str) -> dict[str, Any]:
    engines = spec.load()["engines"]
    cfg = engines[instrument][engine]
    if instrument == "digital" and engine in engines["european"]:
        cfg = {**engines["european"][engine], **cfg}
    return cfg


def _resolution_overrides(instrument: str, engine: str, factor: int) -> dict[str, Any]:
    c = _cfg(instrument, engine)
    if engine in TREES or (instrument == "barrier" and engine == "ql_crr"):
        steps = int(c["steps"]) * factor
        if engine == "ql_lr" and steps % 2 == 0:
            steps += 1  # LR is used with odd step counts only (protocol deviation 1)
        return {"steps": steps}
    if engine in FD or (instrument == "barrier" and engine == "ql_fd"):
        return {"tGrid": int(c["tGrid"]) * factor, "xGrid": int(c["xGrid"]) * factor}
    if engine == "own_pde":
        return {"space_nodes": int(c["space_nodes"]) * factor, "time_steps": int(c["time_steps"]) * factor}
    raise ValueError(f"no resolution parameter for {engine}")


def _local_order(d0: float, d2: float) -> float:
    if d0 == 0 or d2 == 0 or not (math.isfinite(d0) and math.isfinite(d2)):
        return float("nan")
    return math.log2(abs(d0) / abs(d2)) / 2.0


def classify(row: dict[str, Any]) -> dict[str, Any]:
    s = spec.load()
    ref_s = s["refinement"]
    sigmas = s["tolerances"]["mc_sigmas"]
    inst, eng = row["instrument"], row["engine"]
    p = _point(row)
    ref = row["reference"]
    tol_num = spec.tolerance("numerical", ref, inst)
    out = {"instrument": inst, "point_id": row["point_id"], "engine": eng, "labels": row["labels"],
           "diff0": row["diff"], "tol0": row["tol"], "diff1": float("nan"), "diff2": float("nan"),
           "se0": row["std_error"], "se1": float("nan"), "se2": float("nan"), "local_order": float("nan"),
           "category": "", "note": ""}

    if eng in APPROXIMATIONS:
        out["category"] = "approximation error"
        out["note"] = f"ref_uncertainty={row['ref_uncertainty']:.3g}"
        return out
    if eng == "ql_lr":
        # Protocol deviation 9: a confirmed mechanism takes precedence over the
        # refinement rules (refinement can hide it when the 4x tree no longer collapses).
        c = _cfg(inst, eng)
        if lr_collapsed(p.option, p.spot, p.strike, p.T, p.rate, p.div, p.vol, int(c["steps"])):
            out["category"] = LR_MECHANISM
            return out
    if eng in NO_REFINEMENT:
        out["category"] = "not explained"
        out["note"] = "no refinement parameter"
        return out

    if eng in MC and inst != "american":
        base = samples_of(inst, eng)
        res = []
        for f in ref_s["mc_sample_factors"]:
            r = price(inst, p, eng, overrides={"samples": base * f}, run=f"refine={f}")
            res.append(r)
        out["diff1"], out["diff2"] = res[0].value - ref, res[1].value - ref
        out["se1"], out["se2"] = res[0].std_error, res[1].std_error
        if abs(out["diff2"]) <= tol_num + sigmas * out["se2"]:
            out["category"] = "MC noise"
            return out
        if inst == "barrier":
            spd = float(_cfg(inst, eng)["steps_per_day"])
            mon = []
            for f in ref_s["mc_barrier_monitoring_factors"]:
                r = price(inst, p, eng, overrides={"steps_per_day": spd * f}, run=f"monitoring={f}")
                mon.append(r)
            m0, m2 = row["diff"], mon[-1].value - ref
            order = _local_order(m0, m2)
            out["note"] = f"monitoring x{ref_s['mc_barrier_monitoring_factors'][-1]}: diff={m2:.3g} se={mon[-1].std_error:.3g}"
            if (math.isfinite(order) and order >= ref_s["min_local_order"]) or abs(m2) <= tol_num + sigmas * mon[-1].std_error:
                out["category"] = "monitoring discretization"
                out["local_order"] = order
                return out
        out["category"] = "MC bias, not explained"
        return out

    if eng in LSM:
        c = _cfg(inst, eng)
        res = []
        for ft, fs in zip(ref_s["lsm_time_step_factors"], ref_s["mc_sample_factors"]):
            r = price(inst, p, eng, overrides={"timeSteps": int(c["timeSteps"]) * ft, "samples": int(c["samples"]) * fs},
                      run=f"refine={ft}x{fs}")
            res.append(r)
        out["diff1"], out["diff2"] = res[0].value - ref, res[1].value - ref
        out["se1"], out["se2"] = res[0].std_error, res[1].std_error
        if abs(out["diff2"]) <= tol_num + sigmas * out["se2"]:
            out["category"] = "Bermudan/LSM discretization or MC noise, resolved"
        else:
            out["category"] = "LSM bias, not explained"
        return out

    # Deterministic numerical engines.
    d = []
    for f in ref_s["factors"]:
        r = price(inst, p, eng, overrides=_resolution_overrides(inst, eng, f))
        d.append(r.value - ref if r.ok else float("nan"))
    d0, d1, d2 = row["diff"], d[0], d[1]
    out["diff1"], out["diff2"] = d1, d2
    out["local_order"] = _local_order(d0, d2)
    if inst == "american" and eng == "own_pde":
        out["note"] = "self-convergence check (shares code with ref_pde)"
    if math.isfinite(d2) and abs(d2) <= row["tol"]:
        out["category"] = "discretization, resolved at 4x"
    elif (math.isfinite(d2) and abs(d0) > abs(d1) > abs(d2)
          and out["local_order"] >= ref_s["min_local_order"]):
        out["category"] = "discretization, converging slowly"
    else:
        out["category"] = "not explained"
    return out


def disagreement_table(comparisons, jobs: int = 1) -> Table:
    rows = [r for r in comparisons.rows if r["disagree"]]
    table = Table("disagreements", DISAGREEMENT_COLUMNS)
    for r in parallel_map(classify, rows, jobs):
        table.add(**r)
    return table
