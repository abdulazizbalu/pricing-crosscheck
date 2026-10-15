# Draft follow-up to QuantLib pull request #2729 (not posted)

**Title:** Follow-up to #2729: compute the Leisen-Reimer down probability without 1 - pu

## Summary

Pull request #2729 (merged 2026-09-07, in QuantLib 1.44) fixed the Leisen-Reimer
tree for deep in-the-money puts at small sigma*sqrt(T) and added a check that
the up probability lies strictly inside (0, 1). Its description notes that at
the other end pu rounds to 1, and its results table reports 176 Leisen-Reimer
cases that raise an error on its 3360-point grid after the change. So the
errors for deep in-the-money calls are a known and accepted consequence of the
fix, not a regression.

This is a proposal for a small follow-up that would let most of those cases
price: compute `pd_` and `down_` from the mirrored small branch PP(-z) instead
of from `1 - pu_`. Below are a reproduction of one such case, the proposed
change, a NumPy check and a test sketch. I would be glad to submit it as a
pull request if this direction is acceptable.

## Reproduction (Python, QuantLib 1.44 from PyPI)

```python
import QuantLib as ql

today = ql.Date(15, 1, 2025)
ql.Settings.instance().evaluationDate = today
dc = ql.Actual365Fixed()
h = lambda x: ql.QuoteHandle(ql.SimpleQuote(x))
process = ql.BlackScholesMertonProcess(
    h(100.0),
    ql.YieldTermStructureHandle(ql.FlatForward(today, h(0.0), dc)),
    ql.YieldTermStructureHandle(ql.FlatForward(today, h(0.0), dc)),
    ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), h(0.01), dc)),
)
option = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call, 80.0), ql.EuropeanExercise(today + 1))
option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
print(option.NPV())          # 20.0
option.setPricingEngine(ql.BinomialLRVanillaEngine(process, 1001))
print(option.NPV())          # 1.43: 20.0; 1.44: RuntimeError: invalid up probability: 1
```

Closed form: 20.0 (intrinsic value; sigma*sqrt(T) is about 5e-4, d2 is about 426).
With 1.44: `RuntimeError: invalid up probability: 1`, as intended by the
guard in #2729. With the proposed change: 20.0.

Further cases (all with zero rates, sigma 1%) from our runs, QuantLib 1.43 vs
1.44 at 1001, 2003 and 4005 steps: calls with strike 50 (1 and 7 days) and
strike 80 (1 day) price correctly in 1.43 and raise in 1.44 at every step
count; calls with strike 80 at 7 days (d2 about 161) and strike 95 at 1 day
price correctly in both.

## Mechanism

`PeizerPrattMethod2Inversion(z, n)` returns `0.5*(1.0 + root)` for z > 0, with
`root = sqrt(1 - exp(-x))`. When exp(-x) is below about 1.1e-16 this is exactly
1, so the new check `pu_ > 0.0 && pu_ < 1.0` in the `LeisenReimer` constructor
rejects the tree. In exact arithmetic pu < 1 and 1 - pu = PP(-z), which the
small branch represents accurately (down to exp underflow), so the information
is available.

In 1.43 the same trees had `pu_ == 1` and `down_ = 0/0 = NaN`; all weight
followed the all-up path, whose node price is finite (`pow(NaN, 0) == 1`), so
the price was right by accident. The guard added in #2729 is the right
behaviour for the current formulas; the proposal only removes the
cancellation that makes it fire on the call side.

## Proposed change

Compute the down-side quantities from the mirrored small branch instead of
from `1 - pu_`:

```cpp
// in LeisenReimer::LeisenReimer, sketch
pu_ = PeizerPrattMethod2Inversion(d2, oddSteps);
pd_ = PeizerPrattMethod2Inversion(-d2, oddSteps);          // = 1 - pu_ exactly, no cancellation
Real pdash_u = PeizerPrattMethod2Inversion(d2 + std::sqrt(variance), oddSteps);
Real pdash_d = PeizerPrattMethod2Inversion(-(d2 + std::sqrt(variance)), oddSteps);
QL_REQUIRE(pu_ > 0.0 && pd_ > 0.0, "invalid probabilities: " << pu_ << ", " << pd_);
up_   = ermqdt * pdash_u / pu_;
down_ = ermqdt * pdash_d / pd_;                            // instead of (ermqdt - pu_*up_)/(1 - pu_)
```

This uses PP(-z) = 1 - PP(z), which holds for method 2 in exact arithmetic
(the inversion is antisymmetric about 0.5), and avoids the cancellation in
`(ermqdt - pu_*up_)/(1 - pu_)`. `probability()` would need to return `pd_` as
stored rather than `1 - pu_` where that is computed. Only points where exp(-x)
underflows to 0 (|d2| above roughly 27*sqrt(n)) would still be rejected, on
both sides.

A NumPy implementation of this variant (`scripts/lr_suggested_fix.py`, output in
`results/investigations/lr_suggested_fix.txt`, in the
attached project) reproduces the closed form to 1e-10 at strike 200, 120 (puts)
and 50, 80 (calls), 1 and 7 days, sigma 1%, 1001 and 4005 steps, except at
strike 200/50, one day, 1001 steps, where exp(-x) underflows.

## Test case

```cpp
// sketch for test-suite/europeanoption.cpp
// spot 100, strike 80, r = q = 0, sigma 0.01, one day, 1001 steps:
// BinomialLRVanillaEngine NPV for the call should equal AnalyticEuropeanEngine (20.0)
// to 1e-10, and the put with strike 120 should equal 20.0 as well.
```

## What we checked before writing this

- Source of v1.43 and v1.44, and master as of 2026-10-15
  (`binomialtree.cpp`, `binomialdistribution.hpp`, `binomialengine.hpp`).
- Pull request #2729 (merged 2026-09-07): its description covers the case
  where pu rounds to 1 and reports 176 Leisen-Reimer cases that throw on its
  grid (moneyness 0.2 to 5.0, vol 0.01 to 0.50, one day to ten years, 801
  steps). Pull request #2811 (merged 2026-09-23) reports greeks as not
  provided when nodes coincide.
- GitHub searches: issues "Leisen Reimer" and "invalid up probability", pull
  requests "Peizer": no existing proposal for computing pd from PP(-z) found.
  GitHub search results pages may not list everything, so this should be
  re-checked before posting.
