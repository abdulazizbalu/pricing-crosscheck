"""Crank-Nicolson finite differences for Black-Scholes-Merton.

Vanilla, digital and American options are solved in the log-forward variable
y = ln S + (r - q - sigma^2/2) tau, in which the PDE has no first-order term:
    V_tau = 0.5 sigma^2 V_yy - r V.
Barrier (knock-out) options are solved in x = ln S with the drift term, so
that the barrier stays on a grid node:
    V_tau = 0.5 sigma^2 V_xx + (r - q - sigma^2/2) V_x - r V.

Time stepping is Crank-Nicolson with a Rannacher start (each of the first
``rannacher_steps`` steps is replaced by two implicit Euler half-steps).
Payoffs are cell-averaged over [x_i - dx/2, x_i + dx/2], the strike sits on
a grid node, and the value at the spot is obtained by cubic interpolation.
American exercise uses a penalty iteration at every time step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.linalg import solve_banded

N_SD = 8.0  # half-width of the domain in standard deviations
PENALTY = 1e8  # enforcement error ~ multiplier / PENALTY; larger values lose the multiplier sign to rounding
MAX_PENALTY_ITERS = 100
ACTIVATION_TOL = 1e-13  # above solve rounding (~1e-15 relative), below multiplier / PENALTY in the exercise region


@dataclass
class PdeResult:
    value: float
    nodes: int
    steps: int
    penalty_iterations: int = 0


def _cell_average(x: np.ndarray, dx: float, K: float, option: str, payoff: str) -> np.ndarray:
    """Exact average of the payoff of S = exp(x) over each cell."""
    a, b = x - 0.5 * dx, x + 0.5 * dx
    k = math.log(K)
    if payoff == "digital":
        if option == "call":
            frac = (b - np.clip(np.maximum(a, k), a, b)) / dx
        else:
            frac = (np.clip(np.minimum(b, k), a, b) - a) / dx
        return frac
    if option == "call":
        lo = np.maximum(a, k)
        mask = b > k
        out = np.zeros_like(x)
        out[mask] = (np.exp(b[mask]) - np.exp(lo[mask]) - K * (b[mask] - lo[mask])) / dx
        return out
    hi = np.minimum(b, k)
    mask = a < k
    out = np.zeros_like(x)
    out[mask] = (K * (hi[mask] - a[mask]) - (np.exp(hi[mask]) - np.exp(a[mask]))) / dx
    return out


def _payoff(S: np.ndarray | float, K: float, option: str, payoff: str):
    phi = 1.0 if option == "call" else -1.0
    if payoff == "digital":
        return np.where(phi * (S - K) > 0, 1.0, 0.0)
    return np.maximum(phi * (S - K), 0.0)


def _interp_cubic(x: np.ndarray, v: np.ndarray, x0: float) -> float:
    i = int(np.searchsorted(x, x0))
    lo = min(max(i - 2, 0), len(x) - 4)
    xs, vs = x[lo : lo + 4], v[lo : lo + 4]
    total = 0.0
    for j in range(4):
        w = 1.0
        for m in range(4):
            if m != j:
                w *= (x0 - xs[m]) / (xs[j] - xs[m])
        total += w * vs[j]
    return float(total)


def _theta_step(
    v: np.ndarray,
    dt: float,
    theta: float,
    lower: float,
    upper: float,
    sub: float,
    diag: float,
    sup: float,
    obstacle: np.ndarray | None,
) -> tuple[np.ndarray, int]:
    """One theta-scheme step with Dirichlet values at both ends; returns iterations."""
    n = len(v)
    rhs = v.copy()
    expl = 1.0 - theta
    if expl:
        rhs[1:-1] = v[1:-1] + expl * dt * (sub * v[:-2] + diag * v[1:-1] + sup * v[2:])
    rhs[0], rhs[-1] = lower, upper
    ab = np.zeros((3, n))
    ab[0, 2:] = -theta * dt * sup
    ab[1, 1:-1] = 1.0 - theta * dt * diag
    ab[2, :-2] = -theta * dt * sub
    ab[1, 0] = ab[1, -1] = 1.0
    if obstacle is None:
        return solve_banded((1, 1), ab, rhs, overwrite_ab=True, overwrite_b=True, check_finite=False), 1
    # Penalty iteration (Forsyth and Vetzal 2002). Exercise can only be optimal
    # where the payoff is positive (V >= 0 always), so other nodes never enter
    # the active set. A node enters the active set only if its value is below
    # the obstacle by more than ACTIVATION_TOL (relative): where the
    # continuation value equals the payoff up to rounding over a whole region
    # (e.g. r = q = 0, where early exercise is never strictly optimal) the
    # sign of value - obstacle is rounding noise and would otherwise make the
    # active set wander. Iteration stops when the active set repeats; if it
    # repeats after a cycle (a multiplier that is zero up to rounding), the
    # result is projected onto the obstacle, a change at rounding level.
    candidate = obstacle > 0.0
    candidate[0] = candidate[-1] = False
    margin = ACTIVATION_TOL * np.maximum(1.0, np.abs(obstacle))
    active = candidate & (v < obstacle - margin)
    seen = {active.tobytes()}
    for it in range(1, MAX_PENALTY_ITERS + 1):
        abp = ab.copy()
        abp[1, active] += PENALTY
        b = rhs + np.where(active, PENALTY * obstacle, 0.0)
        new = solve_banded((1, 1), abp, b, check_finite=False)
        new_active = candidate & (new < obstacle - margin)
        key = new_active.tobytes()
        if key in seen:
            return np.maximum(new, obstacle), it
        seen.add(key)
        active = new_active
    raise RuntimeError("penalty iteration did not converge")


def _march(
    v0: np.ndarray,
    T: float,
    steps: int,
    rannacher_steps: int,
    coeffs: tuple[float, float, float],
    boundary: Callable[[float], tuple[float, float]],
    obstacle: Callable[[float], np.ndarray] | None,
) -> tuple[np.ndarray, int]:
    """March from tau = 0 to T in uniform steps.

    With an early-exercise constraint the time error of this scheme is first
    order in practice (Forsyth and Vetzal obtain second order only with
    adaptive time steps); ``ref_pde`` therefore reports u = |V3 - V2|, which
    for first-order convergence is about the error of V3.
    """
    sub, diag, sup = coeffs
    taus = T * np.arange(steps + 1) / steps
    v = v0
    iters = 0
    for n in range(steps):
        dt = float(taus[n + 1] - taus[n])
        tau = float(taus[n])
        if n < rannacher_steps:
            parts = [(0.5 * dt, 1.0), (0.5 * dt, 1.0)]
        else:
            parts = [(dt, 0.5)]
        for h, th in parts:
            tau += h
            lower, upper = boundary(tau)
            obs = obstacle(tau) if obstacle is not None else None
            v, k = _theta_step(v, h, th, lower, upper, sub, diag, sup, obs)
            iters += k
    return v, iters


def _aligned_grid(lo: float, hi: float, nodes: int, anchor: float | None) -> np.ndarray:
    """Uniform grid with ``nodes`` points covering about [lo, hi], with ``anchor`` on a node."""
    dx = (hi - lo) / (nodes - 1)
    if anchor is not None and lo < anchor < hi:
        j = round((anchor - lo) / dx)
        lo = anchor - j * dx
    return lo + dx * np.arange(nodes)


def vanilla(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    option: str,
    nodes: int,
    steps: int,
    rannacher_steps: int = 2,
    american: bool = False,
    payoff: str = "vanilla",
    align_strike: bool = True,
    smooth_payoff: bool = True,
) -> PdeResult:
    """European/American vanilla or European cash-or-nothing digital (pays 1)."""
    if american and payoff == "digital":
        raise ValueError("American digital is out of scope")
    m = r - q - 0.5 * sigma * sigma
    y0 = math.log(S) + m * T
    width = N_SD * sigma * math.sqrt(T)
    y = _aligned_grid(y0 - width, y0 + width, nodes, math.log(K) if align_strike else None)
    dx = y[1] - y[0]
    if smooth_payoff:
        v0 = _cell_average(y, dx, K, option, payoff)
    else:
        v0 = _payoff(np.exp(y), K, option, payoff)
    a = 0.5 * sigma * sigma / (dx * dx)
    coeffs = (a, -2.0 * a - r, a)
    s2 = 0.5 * sigma * sigma

    def far_value(tau: float, yy: float) -> float:
        # Zero-volatility limit around the forward; exact for payoffs linear on the
        # far side of the strike, which is where the boundaries are (N_SD deviations).
        fwd = math.exp(yy + s2 * tau)
        val = math.exp(-r * tau) * float(_payoff(fwd, K, option, payoff))
        if american:
            spot = math.exp(yy - m * tau)
            val = max(val, float(_payoff(spot, K, option, payoff)))
        return val

    def boundary(tau: float) -> tuple[float, float]:
        return far_value(tau, y[0]), far_value(tau, y[-1])

    obstacle = None
    if american:
        def obstacle(tau: float) -> np.ndarray:
            return _payoff(np.exp(y - m * tau), K, option, payoff)

    v, iters = _march(v0, T, steps, rannacher_steps, coeffs, boundary, obstacle)
    return PdeResult(_interp_cubic(y, v, y0), nodes, steps, iters)


def knock_out(
    S: float,
    K: float,
    H: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    option: str,
    down: bool,
    nodes: int,
    steps: int,
    rannacher_steps: int = 2,
) -> PdeResult:
    """Continuously monitored knock-out with rebate 0; barrier on the boundary node."""
    m = r - q - 0.5 * sigma * sigma
    x0, h = math.log(S), math.log(H)
    reach = N_SD * sigma * math.sqrt(T) + abs(m) * T
    far = x0 + reach if down else x0 - reach
    lo, hi = (h, far) if down else (far, h)
    k = math.log(K)
    dx = (hi - lo) / (nodes - 1)
    if lo < k < hi:
        # keep the barrier on the boundary node and put the strike on a node too
        cells = max(1, round(abs(k - h) / dx))
        dx = abs(k - h) / cells
        count = max(nodes, int(math.ceil((hi - lo) / dx)) + 1)
    else:
        count = nodes
    x = (h + dx * np.arange(count)) if down else (h - dx * np.arange(count))[::-1]
    v0 = _cell_average(x, dx, K, option, "vanilla")
    if down:
        v0[0] = 0.0
    else:
        v0[-1] = 0.0
    a = 0.5 * sigma * sigma / (dx * dx)
    c = m / (2.0 * dx)
    coeffs = (a - c, -2.0 * a - r, a + c)
    phi = 1.0 if option == "call" else -1.0

    def far_value(tau: float, xx: float) -> float:
        fwd = math.exp(xx + (r - q) * tau)
        return math.exp(-r * tau) * max(phi * (fwd - K), 0.0)

    def boundary(tau: float) -> tuple[float, float]:
        if down:
            return 0.0, far_value(tau, x[-1])
        return far_value(tau, x[0]), 0.0

    v, iters = _march(v0, T, steps, rannacher_steps, coeffs, boundary, None)
    return PdeResult(_interp_cubic(x, v, x0), len(x), steps, iters)
