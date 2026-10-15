"""Diagnostics for disagreements whose mechanism has been confirmed (see docs/findings.md)."""

from __future__ import annotations

import math

LR_MECHANISM = "LR up probability rounds to 0 (finding 8, mechanism confirmed)"


def lr_up_probability_v143(S: float, K: float, T: float, r: float, q: float, sigma: float, steps: int) -> float:
    """Up probability of QuantLib 1.43's Leisen-Reimer tree, recomputed with its formulas.

    LeisenReimer constructor (ql/methods/lattices/binomialtree.cpp) and
    PeizerPrattMethod2Inversion (ql/math/distributions/binomialdistribution.hpp):
    pu = 0.5 + sign(z) sqrt(0.25 (1 - exp(-x))), which is exactly 0 for z < 0 once
    exp(-x) is below about 1.1e-16.
    """
    n = steps if steps % 2 else steps + 1
    drift_per_step = (r - q - 0.5 * sigma * sigma) * T / n
    variance = sigma * sigma * T
    z = (math.log(S / K) + drift_per_step * n) / math.sqrt(variance)
    x = (z / (n + 1.0 / 3.0 + 0.1 / (n + 1.0))) ** 2 * (n + 1.0 / 6.0)
    return 0.5 + (1 if z > 0 else -1) * math.sqrt(0.25 * (1.0 - math.exp(-x)))


def lr_collapsed(option: str, S: float, K: float, T: float, r: float, q: float, sigma: float, steps: int) -> bool:
    """True when the 1.43 LR tree has pu == 0 for a put: every node price is NaN and the
    payoff max(0.0, NaN) evaluates to 0, so the engine returns 0 (docs/findings.md, finding 8)."""
    return option == "put" and lr_up_probability_v143(S, K, T, r, q, sigma, steps) == 0.0


def lr_collapse_threshold(steps: int) -> float:
    """Smallest |d2| at which the 1.43 inversion returns exactly 0 (bisection)."""
    n = steps if steps % 2 else steps + 1

    def zero(z: float) -> bool:
        x = (z / (n + 1.0 / 3.0 + 0.1 / (n + 1.0))) ** 2 * (n + 1.0 / 6.0)
        return 0.5 - math.sqrt(0.25 * (1.0 - math.exp(-x))) == 0.0

    lo, hi = 1.0, 1e4
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if zero(mid) else (mid, hi)
    return hi
