"""Seeded Monte Carlo under Black-Scholes-Merton (numpy PCG64).

European and digital payoffs use exact lognormal sampling at maturity.
Barriers use exact sampling on a time grid plus the Brownian-bridge
probability that the log-price crossed the barrier between grid dates, which
for constant parameters gives an unbiased estimator of the continuously
monitored price. Antithetic pairs; the standard error is computed from the
pair means.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

CHUNK_PATHS = 8192  # paths per chunk (antithetic pairs = CHUNK_PATHS / 2)


@dataclass
class McResult:
    value: float
    std_error: float
    samples: int


def _rng(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed))


def _summarise(pair_means: np.ndarray, samples: int) -> McResult:
    n = len(pair_means)
    return McResult(float(pair_means.mean()), float(pair_means.std(ddof=1) / math.sqrt(n)), samples)


def _check_samples(samples: int) -> int:
    if samples < 4 or samples % 2:
        raise ValueError("samples must be an even number >= 4 (antithetic pairs)")
    return samples // 2


def terminal(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    option: str,
    samples: int,
    seed: int,
    payoff: str = "vanilla",
) -> McResult:
    """European vanilla or cash-or-nothing digital (pays 1)."""
    pairs = _check_samples(samples)
    phi = 1.0 if option == "call" else -1.0
    z = _rng(seed).standard_normal(pairs)
    drift = (r - q - 0.5 * sigma * sigma) * T
    sd = sigma * math.sqrt(T)
    df = math.exp(-r * T)
    s_plus = S * np.exp(drift + sd * z)
    s_minus = S * np.exp(drift - sd * z)
    if payoff == "digital":
        pay = lambda s: (phi * (s - K) > 0).astype(float)  # noqa: E731
    else:
        pay = lambda s: np.maximum(phi * (s - K), 0.0)  # noqa: E731
    pair_means = 0.5 * df * (pay(s_plus) + pay(s_minus))
    return _summarise(pair_means, samples)


def barrier(
    S: float,
    K: float,
    H: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    option: str,
    barrier_type: str,
    samples: int,
    seed: int,
    monitoring_steps: int,
    bridge: bool = True,
) -> McResult:
    """Single barrier, rebate 0. With ``bridge`` the estimator targets continuous monitoring;
    without it, the barrier is checked only on the ``monitoring_steps`` dates."""
    pairs = _check_samples(samples)
    phi = 1.0 if option == "call" else -1.0
    down = barrier_type.startswith("down")
    knock_in = barrier_type.endswith("-in")
    dt = T / monitoring_steps
    drift = (r - q - 0.5 * sigma * sigma) * dt
    sd = sigma * math.sqrt(dt)
    var = sigma * sigma * dt
    log_h = math.log(H)
    df = math.exp(-r * T)
    rng = _rng(seed)
    out = np.empty(pairs)
    done = 0
    chunk_pairs = CHUNK_PATHS // 2
    while done < pairs:
        n = min(chunk_pairs, pairs - done)
        z = rng.standard_normal((n, monitoring_steps))
        vals = []
        for sign in (1.0, -1.0):
            incr = drift + sign * sd * z
            x = math.log(S) + np.cumsum(incr, axis=1)
            prev = np.concatenate([np.full((n, 1), math.log(S)), x[:, :-1]], axis=1)
            dist_prev = (prev - log_h) if down else (log_h - prev)
            dist_next = (x - log_h) if down else (log_h - x)
            crossed_at_dates = np.any(dist_next <= 0.0, axis=1)
            if bridge:
                with np.errstate(over="ignore"):
                    p_cross = np.exp(-2.0 * np.clip(dist_prev, 0, None) * np.clip(dist_next, 0, None) / var)
                p_cross = np.where((dist_prev <= 0) | (dist_next <= 0), 1.0, p_cross)
                survive = np.prod(1.0 - p_cross, axis=1)
            else:
                survive = (~crossed_at_dates).astype(float)
            pay = np.maximum(phi * (np.exp(x[:, -1]) - K), 0.0)
            vals.append(df * pay * ((1.0 - survive) if knock_in else survive))
        out[done : done + n] = 0.5 * (vals[0] + vals[1])
        done += n
    return _summarise(out, samples)
