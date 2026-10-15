"""Check a suggested change to the Leisen-Reimer parameters on the finding-8 points.

QuantLib 1.44 computes pu = PP(d2), up = ermqdt * PP(d2 + s) / pu and
down = (ermqdt - pu * up) / (1 - pu), and rejects pu outside (0, 1). When pu
rounds to 1 (deep in-the-money call) or to 0 the tree is rejected. The
suggestion: compute the down-side quantities from the mirrored small branch,
pd = PP(-d2) and down = ermqdt * PP(-(d2 + s)) / pd, which is the same value in
exact arithmetic (PP(-z) = 1 - PP(z) for the Peizer-Pratt method 2 inversion)
but has no cancellation. A tree is then valid while both pu and pd are
positive. This script prices European options on such a tree in NumPy and
compares with the closed form.

Usage: python scripts/lr_suggested_fix.py
"""

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pricing_crosscheck.reference import closed_form as cf  # noqa: E402


def pp2_small(z: float, n: int) -> float:
    """min(PP(z), 1 - PP(z)) without cancellation (QuantLib 1.44 small branch), for z <= 0."""
    x = (z / (n + 1.0 / 3.0 + 0.1 / (n + 1.0))) ** 2 * (n + 1.0 / 6.0)
    e = math.exp(-x)
    return 0.5 * e / (1.0 + math.sqrt(1.0 - e))


def pp2(z: float, n: int) -> tuple[float, float]:
    """(PP(z), 1 - PP(z)), each side from the small branch where it is the small side."""
    small = pp2_small(-abs(z), n)
    return (1.0 - small, small) if z > 0 else (small, 1.0 - small)


def lr_price(S, K, T, r, q, sigma, option, steps):
    n = steps if steps % 2 else steps + 1
    dt = T / n
    drift = (r - q - 0.5 * sigma * sigma) * dt
    var = sigma * sigma * T
    ermqdt = math.exp(drift + 0.5 * var / n)
    d2 = (math.log(S / K) + drift * n) / math.sqrt(var)
    pu, pd = pp2(d2, n)
    pdash_u, pdash_d = pp2(d2 + math.sqrt(var), n)
    if not (pu > 0.0 and pd > 0.0):
        return float("nan"), pu, pd
    up = ermqdt * pdash_u / pu
    down = ermqdt * pdash_d / pd
    j = np.arange(n + 1)
    with np.errstate(over="ignore", under="ignore"):
        spot = S * np.exp(j * math.log(up) + (n - j) * math.log(down))
    phi = 1.0 if option == "call" else -1.0
    v = np.maximum(phi * (spot - K), 0.0)
    disc = math.exp(-r * dt)
    for _ in range(n):
        v = disc * (pd * v[:-1] + pu * v[1:])
    return float(v[0]), pu, pd


def main() -> None:
    cases = [("put", 200.0, 1), ("put", 200.0, 7), ("put", 120.0, 1), ("put", 120.0, 7),
             ("call", 50.0, 1), ("call", 50.0, 7), ("call", 80.0, 1), ("call", 80.0, 7)]
    print("option strike days steps | closed form | suggested LR | pu | pd")
    for option, K, D in cases:
        for steps in (1001, 4005):
            T = D / 365.0
            ref = cf.european(100.0, K, T, 0.0, 0.0, 0.01, option)
            v, pu, pd = lr_price(100.0, K, T, 0.0, 0.0, 0.01, option, steps)
            print(f"{option:4s} {K:6g} {D:3d} {steps:5d} | {ref:.10f} | {v:.10f} | {pu:.3g} | {pd:.3g}")


if __name__ == "__main__":
    main()
