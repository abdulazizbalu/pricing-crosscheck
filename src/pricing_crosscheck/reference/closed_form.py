"""Closed-form Black-Scholes-Merton prices and Greeks.

Conventions: continuous rate ``r`` and dividend yield ``q``; ``T`` in years;
vega per unit volatility; theta = dV/dt in calendar time per year (= -dV/dT).
"""

from __future__ import annotations

import math

from scipy.special import ndtr

SQRT_2PI = math.sqrt(2.0 * math.pi)


def _sign(option: str) -> int:
    if option == "call":
        return 1
    if option == "put":
        return -1
    raise ValueError(f"option must be 'call' or 'put', got {option!r}")


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI


def norm_cdf(x: float) -> float:
    return float(ndtr(x))


def _d1_d2(S: float, K: float, T: float, r: float, q: float, sigma: float) -> tuple[float, float]:
    sd = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sd
    return d1, d1 - sd


def european(S: float, K: float, T: float, r: float, q: float, sigma: float, option: str) -> float:
    phi = _sign(option)
    if T <= 0.0:
        return max(phi * (S - K), 0.0)
    df_r, df_q = math.exp(-r * T), math.exp(-q * T)
    if sigma * math.sqrt(T) == 0.0:
        return df_r * max(phi * (S * df_q / df_r - K), 0.0)
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    return phi * (S * df_q * norm_cdf(phi * d1) - K * df_r * norm_cdf(phi * d2))


def european_greeks(S: float, K: float, T: float, r: float, q: float, sigma: float, option: str) -> dict[str, float]:
    phi = _sign(option)
    df_r, df_q = math.exp(-r * T), math.exp(-q * T)
    sqrt_t = math.sqrt(T)
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    pdf1 = norm_pdf(d1)
    delta = phi * df_q * norm_cdf(phi * d1)
    gamma = df_q * pdf1 / (S * sigma * sqrt_t)
    vega = S * df_q * pdf1 * sqrt_t
    theta = (
        -S * df_q * pdf1 * sigma / (2.0 * sqrt_t)
        - phi * r * K * df_r * norm_cdf(phi * d2)
        + phi * q * S * df_q * norm_cdf(phi * d1)
    )
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}


def digital(S: float, K: float, T: float, r: float, q: float, sigma: float, option: str, cash: float = 1.0) -> float:
    """Cash-or-nothing: pays ``cash`` at T if S_T > K (call) or S_T < K (put)."""
    phi = _sign(option)
    if T <= 0.0:
        return cash * (1.0 if phi * (S - K) > 0 else 0.0)
    df_r = math.exp(-r * T)
    if sigma * math.sqrt(T) == 0.0:
        forward = S * math.exp((r - q) * T)
        return cash * df_r * (1.0 if phi * (forward - K) > 0 else 0.0)
    _, d2 = _d1_d2(S, K, T, r, q, sigma)
    return cash * df_r * norm_cdf(phi * d2)


def digital_greeks(
    S: float, K: float, T: float, r: float, q: float, sigma: float, option: str, cash: float = 1.0
) -> dict[str, float]:
    phi = _sign(option)
    df_r = math.exp(-r * T)
    sqrt_t = math.sqrt(T)
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    pdf2 = norm_pdf(d2)
    value = cash * df_r * norm_cdf(phi * d2)
    delta = phi * cash * df_r * pdf2 / (S * sigma * sqrt_t)
    gamma = -phi * cash * df_r * pdf2 * d1 / (S * S * sigma * sigma * T)
    vega = -phi * cash * df_r * pdf2 * d1 / sigma
    m = r - q - 0.5 * sigma * sigma
    dd2_dT = m / (sigma * sqrt_t) - d2 / (2.0 * T)
    theta = r * value - phi * cash * df_r * pdf2 * dd2_dT
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}


def barrier(
    S: float, K: float, H: float, T: float, r: float, q: float, sigma: float, option: str, barrier_type: str
) -> float:
    """Single barrier, continuous monitoring, rebate 0 (Reiner and Rubinstein 1991).

    ``barrier_type`` is one of down-and-out, down-and-in, up-and-out, up-and-in.
    The spot must be on the live side of the barrier.
    """
    phi = _sign(option)
    down = barrier_type.startswith("down")
    knock_in = barrier_type.endswith("-in")
    if barrier_type not in ("down-and-out", "down-and-in", "up-and-out", "up-and-in"):
        raise ValueError(f"unknown barrier type {barrier_type!r}")
    if (down and S <= H) or (not down and S >= H):
        raise ValueError("spot must be strictly on the live side of the barrier")
    eta = 1 if down else -1
    b = r - q
    sd = sigma * math.sqrt(T)
    mu = (b - 0.5 * sigma * sigma) / (sigma * sigma)
    df_r = math.exp(-r * T)
    df_q = math.exp((b - r) * T)
    x1 = math.log(S / K) / sd + (1 + mu) * sd
    x2 = math.log(S / H) / sd + (1 + mu) * sd
    y1 = math.log(H * H / (S * K)) / sd + (1 + mu) * sd
    y2 = math.log(H / S) / sd + (1 + mu) * sd
    hs = H / S
    pw1 = hs ** (2 * (mu + 1))
    pw0 = hs ** (2 * mu)
    A = phi * S * df_q * norm_cdf(phi * x1) - phi * K * df_r * norm_cdf(phi * x1 - phi * sd)
    B = phi * S * df_q * norm_cdf(phi * x2) - phi * K * df_r * norm_cdf(phi * x2 - phi * sd)
    C = phi * S * df_q * pw1 * norm_cdf(eta * y1) - phi * K * df_r * pw0 * norm_cdf(eta * y1 - eta * sd)
    D = phi * S * df_q * pw1 * norm_cdf(eta * y2) - phi * K * df_r * pw0 * norm_cdf(eta * y2 - eta * sd)
    above = K > H  # strike above barrier
    call = phi == 1
    if knock_in:
        if down and call:
            return C if above else A - B + D
        if not down and call:
            return A if above else B - C + D
        if down and not call:
            return B - C + D if above else A
        return A - B + D if above else C
    if down and call:
        return A - C if above else B - D
    if not down and call:
        return 0.0 if above else A - B + C - D
    if down and not call:
        return A - B + C - D if above else 0.0
    return B - D if above else A - C
