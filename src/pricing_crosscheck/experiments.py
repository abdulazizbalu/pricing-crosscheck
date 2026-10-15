"""Refinement experiments that back the explanations in the documentation (protocol section 11).

Each experiment varies one resolution parameter of one engine at fixed points
and records the error against the reference. The documentation states an
explanation only where one of these tables supports it.
"""

from __future__ import annotations

from typing import Any

from pricing_crosscheck import spec
from pricing_crosscheck.engines import quantlib as qle
from pricing_crosscheck.pricing import parallel_map
from pricing_crosscheck.reference import closed_form as cf
from pricing_crosscheck.reference import pde
from pricing_crosscheck.results import Table
from pricing_crosscheck.spec import BarrierPoint, VanillaPoint

EXPERIMENT_COLUMNS = [
    "experiment", "instrument", "point_id", "engine", "quantity", "parameter", "value", "reference", "error",
]

ATM_1Y = VanillaPoint("call", 100.0, 365, 0.3, 0.05, 0.03)
ATM_7D = VanillaPoint("call", 100.0, 7, 0.3, 0.05, 0.03)
ATM_1D = VanillaPoint("call", 100.0, 1, 0.3, 0.05, 0.03)


def _lr_steps(profile: str) -> list[int]:
    base = [50, 100, 200, 400, 800, 1600] if profile == "full" else [100, 400]
    return sorted(base + [n + 1 for n in base])


def _grid(profile: str) -> list[int]:
    return [50, 100, 200, 400, 800, 1600] if profile == "full" else [100, 400]


def _tasks(profile: str) -> list[tuple]:
    t: list[tuple] = []
    # 1. LR tree: even versus odd step counts.
    for n in _lr_steps(profile):
        t.append(("lr_even_odd", "european", ATM_1Y, "ql_lr", "price", n))
    # 2. QL FD theta and gamma near expiry versus the time and space grid.
    for p in (ATM_1D, ATM_7D, ATM_1Y):
        for n in _grid(profile):
            t.append(("fd_theta_vs_tgrid", "european", p, "ql_fd", "theta", n))
            for e in ("ql_fd", "ql_fd_damped"):
                t.append(("fd_gamma_vs_xgrid", "european", p, e, "gamma", n))
        # QuantLib FD theta against a one-sided closed-form difference over 0.99 days
        # (both grids at 1600); the reference column holds that difference.
        t.append(("fd_theta_one_day", "european", p, "ql_fd", "theta", 1600))
    # 3. QL CRR barrier: in-out parity residual and knock-out error versus steps.
    for b in _barrier_points():
        for n in [125, 250, 500, 1000, 2000, 4000] if profile == "full" else [500, 1000]:
            t.append(("crr_barrier_in_out", "barrier", b, "ql_crr", "in_out_residual", n))
            t.append(("crr_barrier_knock_out", "barrier", b, "ql_crr", "price", n))
    # 4. Digital PDE: strike alignment and payoff smoothing (own_pde) next to QL FD.
    for n in _grid(profile):
        t.append(("digital_strike_alignment", "digital", ATM_1Y, "own_pde_aligned_smoothed", "price", n))
        t.append(("digital_strike_alignment", "digital", ATM_1Y, "own_pde_unaligned_pointwise", "price", n))
        t.append(("digital_strike_alignment", "digital", ATM_1Y, "ql_fd", "price", n))
    return t


def _barrier_points() -> list[BarrierPoint]:
    out = []
    for c in spec.load()["convergence"]["barrier_points"]:
        out.append(BarrierPoint(c["barrier_type"], c["option"], c["barrier"], c["strike"], c["days"], c["vol"],
                                c["rate"], c["div"]))
    return out


def _run(task: tuple) -> dict[str, Any]:
    experiment, inst, p, engine, quantity, n = task
    a = (p.spot, p.strike, p.T, p.rate, p.div, p.vol, p.option)
    if experiment == "lr_even_odd":
        value = qle.price_vanilla(p, inst, engine, overrides={"steps": n}).value
        ref = cf.european(*a)
    elif experiment in ("fd_theta_vs_tgrid", "fd_gamma_vs_xgrid"):
        # Vary one grid dimension; the other stays at the protocol setting (400).
        ov = {"tGrid": n, "xGrid": 400} if quantity == "theta" else {"tGrid": 400, "xGrid": n}
        value = qle.price_vanilla(p, inst, engine, overrides=ov, want_greeks=True).greeks[quantity]
        ref = cf.european_greeks(*a)[quantity]
    elif experiment == "fd_theta_one_day":
        value = qle.price_vanilla(p, inst, engine, overrides={"tGrid": n, "xGrid": n}, want_greeks=True).greeks["theta"]
        h = min(0.99 / 365.0, 0.99 * p.T)
        ref = (cf.european(p.spot, p.strike, p.T - h, p.rate, p.div, p.vol, p.option) - cf.european(*a)) / h
    elif experiment == "crr_barrier_in_out":
        ki = BarrierPoint(p.barrier_type.replace("-out", "-in"), p.option, p.barrier, p.strike, p.days, p.vol,
                          p.rate, p.div)
        ko_v = qle.price_barrier(p, engine, overrides={"steps": n}).value
        ki_v = qle.price_barrier(ki, engine, overrides={"steps": n}).value
        van = qle.price_vanilla(p.vanilla(), "european", "ql_crr", overrides={"steps": n}).value
        value, ref = ko_v + ki_v, van
    elif experiment == "crr_barrier_knock_out":
        value = qle.price_barrier(p, engine, overrides={"steps": n}).value
        ref = cf.barrier(p.spot, p.strike, p.barrier, p.T, p.rate, p.div, p.vol, p.option, p.barrier_type)
    elif experiment == "digital_strike_alignment":
        ref = cf.digital(*a)
        if engine == "ql_fd":
            value = qle.price_vanilla(p, "digital", "ql_fd", overrides={"tGrid": n, "xGrid": n}).value
        else:
            aligned = engine == "own_pde_aligned_smoothed"
            value = pde.vanilla(*a, n, n, payoff="digital", align_strike=aligned, smooth_payoff=aligned).value
    else:
        raise ValueError(experiment)
    return {"experiment": experiment, "instrument": inst, "point_id": p.point_id, "engine": engine,
            "quantity": quantity, "parameter": n, "value": value, "reference": ref, "error": value - ref}


def experiments_table(profile: str, jobs: int = 1) -> Table:
    table = Table("experiments", EXPERIMENT_COLUMNS)
    for row in parallel_map(_run, _tasks(profile), jobs):
        table.add(**row)
    return table
