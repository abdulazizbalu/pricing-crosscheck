"""Minimal reproduction: QuantLib BinomialLRVanillaEngine for a deep in-the-money
European put with very small sigma * sqrt(T). Prints the LR value next to the
closed form, QuantLib's analytic engine and QuantLib's CRR tree.

Usage: python scripts/repro_lr_deep_itm.py
"""

import QuantLib as ql

today = ql.Date(15, 1, 2025)
ql.Settings.instance().evaluationDate = today
dc = ql.Actual365Fixed()


def process(spot, r, q, vol):
    h = lambda x: ql.QuoteHandle(ql.SimpleQuote(x))  # noqa: E731
    return ql.BlackScholesMertonProcess(
        h(spot),
        ql.YieldTermStructureHandle(ql.FlatForward(today, h(q), dc)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, h(r), dc)),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), h(vol), dc)),
    )


print(f"QuantLib {ql.__version__}")
print("strike days vol steps | analytic | LR | CRR")
for strike, days, vol in [(200.0, 1, 0.01), (200.0, 7, 0.01), (200.0, 7, 0.1), (120.0, 7, 0.01), (200.0, 365, 0.01)]:
    p = process(100.0, 0.0, 0.0, vol)
    opt = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Put, strike), ql.EuropeanExercise(today + days))
    opt.setPricingEngine(ql.AnalyticEuropeanEngine(p))
    analytic = opt.NPV()
    for steps in (101, 1001):
        opt.setPricingEngine(ql.BinomialLRVanillaEngine(p, steps))
        lr = opt.NPV()
        opt.setPricingEngine(ql.BinomialCRRVanillaEngine(p, steps))
        crr = opt.NPV()
        print(f"{strike:6g} {days:4d} {vol:4g} {steps:5d} | {analytic:.6f} | {lr:.6f} | {crr:.6f}")
