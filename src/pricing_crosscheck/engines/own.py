"""Our own engines (closed forms, CRR, PDE, MC) at their protocol settings.

Used in two roles: as references (``cf``, ``ref_pde``), which must not fail,
and as engines compared on the grid (``own_*``), whose failures are recorded
like any other engine's.
"""

from __future__ import annotations

import math
from typing import Any

from pricing_crosscheck import spec
from pricing_crosscheck.engines.quantlib import EngineResult
from pricing_crosscheck.reference import closed_form as cf
from pricing_crosscheck.reference import mc, pde, tree
from pricing_crosscheck.spec import BarrierPoint, VanillaPoint

REFERENCE_ENGINES = ("cf", "ref_pde")


class NotApplicable(Exception):
    """An own engine that does not price this instrument by design."""


def _cfg(instrument: str, engine: str, overrides: dict[str, Any] | None) -> dict[str, Any]:
    return {**spec.load()["engines"][instrument][engine], **(overrides or {})}


def _finish(result: EngineResult, engine: str) -> EngineResult:
    if not math.isfinite(result.value):
        if engine in REFERENCE_ENGINES:
            raise RuntimeError(f"reference {engine} returned {result.value}")
        result.status = f"not available: non-finite value {result.value}"
    return result


def _guard(engine: str, fn) -> EngineResult:
    try:
        return _finish(fn(), engine)
    except NotApplicable as exc:
        return EngineResult(status=f"not applicable: {exc}")


def american_reference(p: VanillaPoint) -> tuple[float, float]:
    """``ref_pde`` value and its uncertainty u = |V_finest - V_previous|."""
    cfg = spec.load()["engines"]["american"]["ref_pde"]
    vals = [
        pde.vanilla(p.spot, p.strike, p.T, p.rate, p.div, p.vol, p.option, m, n,
                    rannacher_steps=cfg["rannacher_steps"], american=True).value
        for m, n in cfg["levels"]
    ]
    if not all(math.isfinite(v) for v in vals):
        raise RuntimeError(f"ref_pde non-finite at {p.point_id}: {vals}")
    return vals[-1], abs(vals[-1] - vals[-2])


def price_vanilla(
    p: VanillaPoint, instrument: str, engine: str, seed: int | None = None, overrides: dict[str, Any] | None = None
) -> EngineResult:
    args = (p.spot, p.strike, p.T, p.rate, p.div, p.vol, p.option)
    payoff = "digital" if instrument == "digital" else "vanilla"
    american = instrument == "american"

    def run() -> EngineResult:
        if engine == "cf":
            fn = cf.digital if payoff == "digital" else cf.european
            if american:
                raise NotApplicable("no closed form for American options")
            return EngineResult(value=fn(*args))
        if engine == "ref_pde":
            value, u = american_reference(p)
            return EngineResult(value=value, uncertainty=u)
        c = _cfg(instrument, engine, overrides)
        if engine == "own_crr":
            if payoff == "digital":
                raise NotApplicable("own_crr prices vanilla payoffs only")
            return EngineResult(value=tree.crr(*args, int(c["steps"]), american=american))
        if engine == "own_pde":
            r = pde.vanilla(*args, int(c["space_nodes"]), int(c["time_steps"]),
                            rannacher_steps=int(c["rannacher_steps"]), american=american, payoff=payoff)
            return EngineResult(value=r.value)
        if engine == "own_mc":
            if american:
                raise NotApplicable("own_mc does not price American options")
            r = mc.terminal(*args, int(c["samples"]), seed, payoff)
            return EngineResult(value=r.value, std_error=r.std_error)
        raise ValueError(f"unknown own engine {engine!r}")

    return _guard(engine, run)


def price_barrier(
    p: BarrierPoint, engine: str, seed: int | None = None, overrides: dict[str, Any] | None = None
) -> EngineResult:
    args = (p.spot, p.strike, p.barrier, p.T, p.rate, p.div, p.vol, p.option)

    def run() -> EngineResult:
        if engine == "cf":
            return EngineResult(value=cf.barrier(*args, p.barrier_type))
        c = _cfg("barrier", engine, overrides)
        if engine == "own_pde":
            if p.is_knock_in:
                raise NotApplicable("own_pde prices knock-outs only (protocol section 4)")
            r = pde.knock_out(*args, p.is_down, int(c["space_nodes"]), int(c["time_steps"]),
                              rannacher_steps=int(c["rannacher_steps"]))
            return EngineResult(value=r.value)
        if engine == "own_mc":
            steps = max(1, int(round(c["steps_per_day"] * p.days)))
            r = mc.barrier(*args, p.barrier_type, int(c["samples"]), seed, steps, bridge=bool(c["bridge_correction"]))
            return EngineResult(value=r.value, std_error=r.std_error)
        raise ValueError(f"unknown own engine {engine!r}")

    return _guard(engine, run)
