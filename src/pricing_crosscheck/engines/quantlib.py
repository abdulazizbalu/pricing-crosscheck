"""QuantLib engines under test, configured from ``specs/phase1.json``.

Each call builds a fresh market (flat curves, constant volatility) and a fresh
engine. Exceptions and non-finite values are returned as ``status`` strings,
never raised: protocol section 4, "Engine failures".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import QuantLib as ql

from pricing_crosscheck import spec
from pricing_crosscheck.spec import BarrierPoint, VanillaPoint

VALUATION_DATE = ql.Date(15, 1, 2025)
DAY_COUNT = ql.Actual365Fixed()
GREEKS = ("delta", "gamma", "vega", "theta")
NAN = float("nan")

_BARRIER_TYPES = {
    "down-and-out": ql.Barrier.DownOut,
    "down-and-in": ql.Barrier.DownIn,
    "up-and-out": ql.Barrier.UpOut,
    "up-and-in": ql.Barrier.UpIn,
}


@dataclass
class EngineResult:
    value: float = NAN
    std_error: float = NAN
    uncertainty: float = NAN  # reference uncertainty (ref_pde only)
    greeks: dict[str, float] = field(default_factory=lambda: {g: NAN for g in GREEKS})
    status: str = "ok"

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _check_date() -> None:
    expected = spec.load()["market"]["valuation_date"]
    if VALUATION_DATE.ISO() != expected:
        raise RuntimeError(f"valuation date {VALUATION_DATE.ISO()} differs from spec {expected}")


def _process(spot: float, rate: float, div: float, vol: float) -> ql.BlackScholesMertonProcess:
    ql.Settings.instance().evaluationDate = VALUATION_DATE

    def quote(x: float) -> ql.QuoteHandle:
        return ql.QuoteHandle(ql.SimpleQuote(x))

    risk_free = ql.YieldTermStructureHandle(ql.FlatForward(VALUATION_DATE, quote(rate), DAY_COUNT, ql.Continuous))
    dividend = ql.YieldTermStructureHandle(ql.FlatForward(VALUATION_DATE, quote(div), DAY_COUNT, ql.Continuous))
    volatility = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(VALUATION_DATE, ql.NullCalendar(), quote(vol), DAY_COUNT)
    )
    return ql.BlackScholesMertonProcess(quote(spot), dividend, risk_free, volatility)


def _option_type(option: str) -> int:
    return ql.Option.Call if option == "call" else ql.Option.Put


def _maturity(days: int) -> ql.Date:
    return VALUATION_DATE + int(days)


def _vanilla_engine(name: str, cfg: dict[str, Any], process, seed: int | None, overrides: dict[str, Any]):
    c = {**cfg, **overrides}
    cls = c.get("class")
    if cls == "AnalyticEuropeanEngine":
        return ql.AnalyticEuropeanEngine(process)
    if cls == "BinomialCRRVanillaEngine":
        return ql.BinomialCRRVanillaEngine(process, int(c["steps"]))
    if cls == "BinomialLRVanillaEngine":
        return ql.BinomialLRVanillaEngine(process, int(c["steps"]))
    if cls == "FdBlackScholesVanillaEngine":
        return ql.FdBlackScholesVanillaEngine(process, int(c["tGrid"]), int(c["xGrid"]), int(c["dampingSteps"]))
    if cls == "MCEuropeanEngine":
        return ql.MCEuropeanEngine(
            process,
            c["rng"],
            timeSteps=int(c["timeSteps"]),
            antitheticVariate=bool(c["antithetic"]),
            requiredSamples=int(c["samples"]),
            seed=_valid_seed(seed),
        )
    if cls == "BaroneAdesiWhaleyApproximationEngine":
        return ql.BaroneAdesiWhaleyApproximationEngine(process)
    if cls == "BjerksundStenslandApproximationEngine":
        return ql.BjerksundStenslandApproximationEngine(process)
    if cls == "QdFpAmericanEngine":
        scheme = getattr(ql.QdFpAmericanEngine, c["scheme"])()
        return ql.QdFpAmericanEngine(process, scheme)
    if cls == "MCAmericanEngine":
        return ql.MCAmericanEngine(
            process,
            c["rng"],
            timeSteps=int(c["timeSteps"]),
            antitheticVariate=bool(c["antithetic"]),
            requiredSamples=int(c["samples"]),
            seed=_valid_seed(seed),
            polynomOrder=int(c["polynomOrder"]),
            polynomType=getattr(ql.LsmBasisSystem, c["polynomType"]),
            nCalibrationSamples=int(c["calibrationSamples"]),
        )
    raise ValueError(f"{name}: unsupported engine class {cls!r}")


def _valid_seed(seed: int | None) -> int:
    if seed is None or seed <= 0:
        raise ValueError("Monte Carlo engines need a positive seed (0 means clock-based in QuantLib)")
    return int(seed)


def _evaluate(instrument, want_greeks: bool, is_mc: bool) -> EngineResult:
    res = EngineResult()
    try:
        value = instrument.NPV()
    except RuntimeError as exc:  # QuantLib errors surface as RuntimeError
        msg = str(exc).strip()[:200]
        res.status = ("not available (expected refusal): " if "non-plain payoff" in msg else "not available: ") + msg
        return res
    if not math.isfinite(value):
        res.status = f"not available: non-finite value {value}"
        return res
    res.value = float(value)
    if is_mc:
        try:
            res.std_error = float(instrument.errorEstimate())
        except RuntimeError:
            res.std_error = NAN
        if not (math.isfinite(res.std_error) and res.std_error >= 0):
            # without a standard error the MC tolerance would silently shrink
            res.status = "not available: no standard error estimate"
            res.value = NAN
            return res
    if want_greeks:
        for g in GREEKS:
            try:
                x = float(getattr(instrument, g)())
                res.greeks[g] = x if math.isfinite(x) else NAN
            except RuntimeError:
                res.greeks[g] = NAN
    return res


def price_vanilla(
    point: VanillaPoint,
    instrument: str,
    engine: str,
    seed: int | None = None,
    overrides: dict[str, Any] | None = None,
    want_greeks: bool = False,
) -> EngineResult:
    """European, American or digital (cash-or-nothing, pays 1)."""
    _check_date()
    engines = spec.load()["engines"]
    cfg = engines[instrument][engine]
    if instrument == "digital" and engine in engines["european"]:
        # Settings the digital entry does not repeat are those of the European entry.
        cfg = {**engines["european"][engine], **cfg}
    process = _process(point.spot, point.rate, point.div, point.vol)
    if instrument == "digital":
        payoff = ql.CashOrNothingPayoff(_option_type(point.option), point.strike, 1.0)
    else:
        payoff = ql.PlainVanillaPayoff(_option_type(point.option), point.strike)
    if instrument == "american":
        exercise = ql.AmericanExercise(VALUATION_DATE, _maturity(point.days))
    else:
        exercise = ql.EuropeanExercise(_maturity(point.days))
    option = ql.VanillaOption(payoff, exercise)
    try:
        option.setPricingEngine(_vanilla_engine(engine, cfg, process, seed, overrides or {}))
    except RuntimeError as exc:
        return EngineResult(status=f"not available: {str(exc).strip()[:200]}")
    return _evaluate(option, want_greeks, is_mc=cfg.get("class", "").startswith("MC"))


def price_barrier(
    point: BarrierPoint,
    engine: str,
    seed: int | None = None,
    overrides: dict[str, Any] | None = None,
) -> EngineResult:
    _check_date()
    c = {**spec.load()["engines"]["barrier"][engine], **(overrides or {})}
    process = _process(point.spot, point.rate, point.div, point.vol)
    option = ql.BarrierOption(
        _BARRIER_TYPES[point.barrier_type],
        point.barrier,
        0.0,
        ql.PlainVanillaPayoff(_option_type(point.option), point.strike),
        ql.EuropeanExercise(_maturity(point.days)),
    )
    cls = c["class"]
    try:
        if cls == "AnalyticBarrierEngine":
            eng = ql.AnalyticBarrierEngine(process)
        elif cls == "BinomialCRRBarrierEngine":
            eng = ql.BinomialCRRBarrierEngine(process, int(c["steps"]))
        elif cls == "FdBlackScholesBarrierEngine":
            eng = ql.FdBlackScholesBarrierEngine(process, int(c["tGrid"]), int(c["xGrid"]), int(c["dampingSteps"]))
        elif cls == "MCBarrierEngine":
            steps = max(1, int(round(c["steps_per_day"] * point.days)))
            eng = ql.MCBarrierEngine(
                process,
                c["rng"],
                timeSteps=steps,
                brownianBridge=bool(c["brownianBridge"]),
                antitheticVariate=bool(c["antithetic"]),
                requiredSamples=int(c["samples"]),
                isBiased=bool(c["isBiased"]),
                seed=_valid_seed(seed),
            )
        else:
            raise ValueError(f"{engine}: unsupported engine class {cls!r}")
        option.setPricingEngine(eng)
    except RuntimeError as exc:
        return EngineResult(status=f"not available: {str(exc).strip()[:200]}")
    return _evaluate(option, False, is_mc=cls.startswith("MC"))
