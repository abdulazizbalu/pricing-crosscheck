"""Cox-Ross-Rubinstein binomial tree for European and American vanilla options."""

from __future__ import annotations

import math

import numpy as np


def crr(
    S: float, K: float, T: float, r: float, q: float, sigma: float, option: str, steps: int, american: bool = False
) -> float:
    if option not in ("call", "put"):
        raise ValueError(option)
    phi = 1.0 if option == "call" else -1.0
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    growth = math.exp((r - q) * dt)
    p = (growth - d) / (u - d)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"CRR probability {p} outside [0, 1]: sigma*sqrt(dt) too small for the drift")
    disc = math.exp(-r * dt)
    pu, pd = disc * p, disc * (1.0 - p)
    log_u = math.log(u)
    j = np.arange(steps + 1, dtype=float)
    values = np.maximum(phi * (S * np.exp((2.0 * j - steps) * log_u) - K), 0.0)
    for i in range(steps - 1, -1, -1):
        values = pu * values[1:] + pd * values[:-1]
        if american:
            spots = S * np.exp((2.0 * j[: i + 1] - i) * log_u)
            np.maximum(values, phi * (spots - K), out=values)
    return float(values[0])


def crr_averaged(
    S: float, K: float, T: float, r: float, q: float, sigma: float, option: str, steps: int, american: bool = False
) -> float:
    """Average of N and N + 1 steps, which damps the odd-even oscillation of CRR."""
    return 0.5 * (
        crr(S, K, T, r, q, sigma, option, steps, american) + crr(S, K, T, r, q, sigma, option, steps + 1, american)
    )
