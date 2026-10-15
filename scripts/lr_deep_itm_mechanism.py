"""Finding 8: QuantLib BinomialLRVanillaEngine for deep in-the-money puts at very
small sigma * sqrt(T).

For every QuantLib LR point of results/full/comparisons.csv with |diff| > 1, and
for a few control points, this script
  1. recomputes the Leisen-Reimer tree parameters with the formulas of QuantLib
     1.43 (ql/methods/lattices/binomialtree.cpp, LeisenReimer constructor, and
     ql/math/distributions/binomialdistribution.hpp, PeizerPrattMethod2Inversion),
  2. prices the option with the QuantLib installed in the current environment,
and writes results/investigations/lr_deep_itm_ql<version>.csv.

Usage: python scripts/lr_deep_itm_mechanism.py   (run once per QuantLib version)
"""

import csv
import math
import sys
from pathlib import Path

import QuantLib as ql

ROOT = Path(__file__).resolve().parents[1]
STEPS = (1001, 2003, 4005)
CONTROLS = [  # (option, strike, days, vol, rate, div)
    ("put", 120.0, 7, 0.01, 0.0, 0.0),
    ("put", 200.0, 7, 0.1, 0.0, 0.0),
    ("call", 50.0, 1, 0.01, 0.0, 0.0),
    ("call", 50.0, 7, 0.01, 0.0, 0.0),
    ("call", 80.0, 1, 0.01, 0.0, 0.0),
    ("call", 80.0, 7, 0.01, 0.0, 0.0),
    ("call", 95.0, 1, 0.01, 0.0, 0.0),
]


def peizer_pratt_2_v143(z: float, n: int) -> tuple[float, float]:
    """QuantLib 1.43 formula; returns (probability, exp term)."""
    x = (z / (n + 1.0 / 3.0 + 0.1 / (n + 1.0))) ** 2 * (n + 1.0 / 6.0)
    e = math.exp(-x)
    return 0.5 + (1 if z > 0 else -1) * math.sqrt(0.25 * (1.0 - e)), e


def lr_parameters_v143(S, K, T, r, q, sigma, steps):
    n = steps if steps % 2 else steps + 1
    dt = T / n
    drift_per_step = (r - q - 0.5 * sigma * sigma) * dt
    variance = sigma * sigma * T
    ermqdt = math.exp(drift_per_step + 0.5 * variance / n)
    d2 = (math.log(S / K) + drift_per_step * n) / math.sqrt(variance)
    pu, e = peizer_pratt_2_v143(d2, n)
    pdash, _ = peizer_pratt_2_v143(d2 + math.sqrt(variance), n)
    with_nan = float("nan")
    up = ermqdt * pdash / pu if pu != 0.0 else with_nan  # C++: 0.0/0.0 -> NaN
    down = (ermqdt - pu * up) / (1.0 - pu) if pu != 1.0 and math.isfinite(up) else with_nan
    return n, d2, e, pu, up, down


def quantlib_price(option, K, days, vol, r, q, steps):
    today = ql.Date(15, 1, 2025)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()

    def h(x):
        return ql.QuoteHandle(ql.SimpleQuote(x))

    proc = ql.BlackScholesMertonProcess(
        h(100.0),
        ql.YieldTermStructureHandle(ql.FlatForward(today, h(q), dc)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, h(r), dc)),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), h(vol), dc)),
    )
    typ = ql.Option.Put if option == "put" else ql.Option.Call
    opt = ql.VanillaOption(ql.PlainVanillaPayoff(typ, K), ql.EuropeanExercise(today + days))
    opt.setPricingEngine(ql.AnalyticEuropeanEngine(proc))
    analytic = opt.NPV()
    opt.setPricingEngine(ql.BinomialLRVanillaEngine(proc, steps))
    try:
        return analytic, opt.NPV(), ""
    except RuntimeError as exc:
        return analytic, float("nan"), str(exc).strip()[:80]


def points():
    path = ROOT / "results" / "full" / "comparisons.csv"
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(line for line in fh if not line.startswith("#")))
    pts = []
    for r in rows:
        if r["engine"] == "ql_lr" and r["status"] == "ok" and abs(float(r["diff"])) > 1.0:
            pts.append((r["option"], float(r["strike"]), int(r["days"]), float(r["vol"]),
                        float(r["rate"]), float(r["div"])))
    return pts + CONTROLS


def main() -> None:
    out = ROOT / "results" / "investigations" / f"lr_deep_itm_ql{ql.__version__}.csv"
    cols = ["option", "strike", "days", "vol", "rate", "div", "steps", "tree_steps", "d2", "exp_term",
            "pu_v143", "up_v143", "down_v143", "analytic", "lr_price", "lr_error", "installed_quantlib"]
    with out.open("w", encoding="utf-8", newline="") as fh:
        fh.write(f"# QuantLib {ql.__version__}, Python {sys.version.split()[0]}\n")
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(cols)
        for (opt, K, D, v, r, q) in points():
            for steps in STEPS:
                n, d2, e, pu, up, down = lr_parameters_v143(100.0, K, D / 365.0, r, q, v, steps)
                analytic, lr, err = quantlib_price(opt, K, D, v, r, q, steps)
                w.writerow([opt, K, D, v, r, q, steps, n, repr(d2), repr(e), repr(pu), repr(up), repr(down),
                            repr(analytic), repr(lr), err, ql.__version__])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
