"""Validation of the closed forms (docs/protocol.md section 7) before they serve as references."""

import math

import numpy as np
import pytest
from scipy import integrate

from pricing_crosscheck import spec
from pricing_crosscheck.reference import closed_form as cf

POINTS = spec.vanilla_points("full")
CALLS = [p for p in POINTS if p.option == "call"]


def _args(p):
    return p.spot, p.strike, p.T, p.rate, p.div, p.vol


@pytest.mark.parametrize("p", CALLS, ids=lambda p: p.point_id)
def test_put_call_parity(p):
    S, K, T, r, q, v = _args(p)
    lhs = cf.european(S, K, T, r, q, v, "call") - cf.european(S, K, T, r, q, v, "put")
    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert abs(lhs - rhs) <= 1e-12 * max(S, K)


@pytest.mark.parametrize("p", CALLS, ids=lambda p: p.point_id)
def test_digital_call_plus_put_is_discount_factor(p):
    S, K, T, r, q, v = _args(p)
    total = cf.digital(S, K, T, r, q, v, "call") + cf.digital(S, K, T, r, q, v, "put")
    assert abs(total - math.exp(-r * T)) <= 1e-14


@pytest.mark.parametrize("p", CALLS[::7], ids=lambda p: p.point_id)
def test_digital_is_minus_strike_derivative_of_call(p):
    S, K, T, r, q, v = _args(p)
    h = 1e-2 * K * v * math.sqrt(T)  # small relative to the width of the density

    def d(hh):
        return -(cf.european(S, K + hh, T, r, q, v, "call") - cf.european(S, K - hh, T, r, q, v, "call")) / (2 * hh)

    richardson = (4 * d(h / 2) - d(h)) / 3
    assert abs(richardson - cf.digital(S, K, T, r, q, v, "call")) <= 1e-6


@pytest.mark.parametrize("option", ["call", "put"])
@pytest.mark.parametrize("K", [80.0, 100.0, 120.0])
def test_limits(option, K):
    S, r, q = 100.0, 0.05, 0.03
    fwd = S * math.exp((r - q) * 1.0)
    phi = 1 if option == "call" else -1
    # sigma -> 0: discounted intrinsic value of the forward
    assert cf.european(S, K, 1.0, r, q, 1e-9, option) == pytest.approx(
        math.exp(-r) * max(phi * (fwd - K), 0.0), abs=1e-9
    )
    # T -> 0: payoff; at the money the time value is ~0.4 S sigma sqrt(T)
    t = 1e-12
    bound = 0.4 * S * 0.3 * math.sqrt(t) * 1.01 + 1e-12
    assert abs(cf.european(S, K, t, r, q, 0.3, option) - max(phi * (S - K), 0.0)) <= bound
    assert cf.european(S, K, 0.0, r, q, 0.3, option) == max(phi * (S - K), 0.0)


def test_deep_in_and_out_of_the_money():
    S, T, r, q, v = 100.0, 1.0, 0.05, 0.03, 0.3
    deep_itm = cf.european(S, 1.0, T, r, q, v, "call")
    assert deep_itm == pytest.approx(S * math.exp(-q * T) - 1.0 * math.exp(-r * T), rel=1e-14)
    assert cf.european(S, 1e4, T, r, q, v, "call") < 1e-30
    assert cf.european(S, 1.0, T, r, q, v, "put") < 1e-30


def _integrate(S, K, T, r, q, v, payoff):
    sd = v * math.sqrt(T)
    drift = math.log(S) + (r - q - 0.5 * v * v) * T
    z_k = (math.log(K) - drift) / sd

    def f(z):
        return payoff(math.exp(drift + sd * z)) * math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)

    lo, hi = -40.0, 40.0
    pts = [z for z in [z_k] if lo < z < hi]
    total = 0.0
    edges = [lo, *pts, hi]
    for a, b in zip(edges, edges[1:]):
        total += integrate.quad(f, a, b, epsabs=1e-14, epsrel=1e-12, limit=400)[0]
    return math.exp(-r * T) * total


@pytest.mark.parametrize("p", POINTS[::5], ids=lambda p: p.point_id)
def test_against_numerical_integration(p):
    S, K, T, r, q, v = _args(p)
    phi = 1 if p.option == "call" else -1
    vanilla = _integrate(S, K, T, r, q, v, lambda s: max(phi * (s - K), 0.0))
    assert cf.european(S, K, T, r, q, v, p.option) == pytest.approx(vanilla, rel=1e-8, abs=1e-10)
    dig = _integrate(S, K, T, r, q, v, lambda s: 1.0 if phi * (s - K) > 0 else 0.0)
    assert cf.digital(S, K, T, r, q, v, p.option) == pytest.approx(dig, rel=1e-8, abs=1e-10)


def _fd_greeks(price, S, K, T, r, q, v, option):
    def f(s=S, t=T, vol=v):
        return price(s, K, t, r, q, vol, option)

    def rich(g, h):
        return (4 * g(h / 2) - g(h)) / 3

    hs, ht, hv = 1e-3 * S, 1e-3 * T, 1e-3 * v
    delta = rich(lambda h: (f(s=S + h) - f(s=S - h)) / (2 * h), hs)
    gamma = rich(lambda h: (f(s=S + h) - 2 * f() + f(s=S - h)) / (h * h), hs)
    vega = rich(lambda h: (f(vol=v + h) - f(vol=v - h)) / (2 * h), hv)
    theta = -rich(lambda h: (f(t=T + h) - f(t=T - h)) / (2 * h), ht)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}


GREEK_POINTS = [p for p in POINTS if p.vol >= 0.1 and p.days >= 7][::3]


@pytest.mark.parametrize("p", GREEK_POINTS, ids=lambda p: p.point_id)
@pytest.mark.parametrize("kind", ["european", "digital"])
def test_greeks_against_finite_differences(p, kind):
    price = cf.european if kind == "european" else cf.digital
    greeks = cf.european_greeks if kind == "european" else cf.digital_greeks
    S, K, T, r, q, v = _args(p)
    exact = greeks(S, K, T, r, q, v, p.option)
    approx = _fd_greeks(price, S, K, T, r, q, v, p.option)
    for name in exact:
        scale = max(abs(exact[name]), 1e-6 if name == "gamma" else 1e-4)
        assert abs(exact[name] - approx[name]) <= 1e-5 * scale + 1e-9, name


@pytest.mark.parametrize("p", POINTS, ids=lambda p: p.point_id)
@pytest.mark.parametrize("kind", ["european", "digital"])
def test_black_scholes_pde_identity(p, kind):
    price = cf.european if kind == "european" else cf.digital
    greeks = cf.european_greeks if kind == "european" else cf.digital_greeks
    S, K, T, r, q, v = _args(p)
    g = greeks(S, K, T, r, q, v, p.option)
    V = price(S, K, T, r, q, v, p.option)
    terms = [g["theta"], 0.5 * v * v * S * S * g["gamma"], (r - q) * S * g["delta"], -r * V]
    residual = sum(terms)
    assert abs(residual) <= 1e-10 * max(1.0, max(abs(t) for t in terms))


BARRIERS = spec.barrier_points("full")


@pytest.mark.parametrize("p", [b for b in BARRIERS if not b.is_knock_in], ids=lambda p: p.point_id)
def test_in_out_parity(p):
    S, K, H, T, r, q, v = p.spot, p.strike, p.barrier, p.T, p.rate, p.div, p.vol
    ko = cf.barrier(S, K, H, T, r, q, v, p.option, p.barrier_type)
    ki = cf.barrier(S, K, H, T, r, q, v, p.option, p.barrier_type.replace("-out", "-in"))
    assert abs(ko + ki - cf.european(S, K, T, r, q, v, p.option)) <= 1e-10 * max(1.0, K)
    assert ko >= -1e-12 and ki >= -1e-12


@pytest.mark.parametrize("option", ["call", "put"])
@pytest.mark.parametrize("down", [True, False])
def test_barrier_limits(option, down):
    S, K, T, r, q, v = 100.0, 100.0, 1.0, 0.05, 0.02, 0.3
    vanilla = cf.european(S, K, T, r, q, v, option)
    far = 1e-6 if down else 1e6
    prefix = "down" if down else "up"
    assert cf.barrier(S, K, far, T, r, q, v, option, f"{prefix}-and-out") == pytest.approx(vanilla, rel=1e-12)
    assert abs(cf.barrier(S, K, far, T, r, q, v, option, f"{prefix}-and-in")) < 1e-12
    near = S * (1 - 1e-10) if down else S * (1 + 1e-10)
    assert abs(cf.barrier(S, K, near, T, r, q, v, option, f"{prefix}-and-out")) < 1e-6
    assert cf.barrier(S, K, near, T, r, q, v, option, f"{prefix}-and-in") == pytest.approx(vanilla, abs=1e-6)


def test_worthless_barriers():
    S, T, r, q, v = 100.0, 1.0, 0.05, 0.02, 0.3
    assert cf.barrier(S, 100.0, 110.0, T, r, q, v, "call", "up-and-out") >= 0  # H > K: alive
    assert cf.barrier(S, 115.0, 110.0, T, r, q, v, "call", "up-and-out") == 0.0
    assert cf.barrier(S, 85.0, 90.0, T, r, q, v, "put", "down-and-out") == 0.0


@pytest.mark.parametrize("p", [b for b in BARRIERS if b.barrier_type == "down-and-out" and b.option == "call"
                               and b.barrier <= b.strike], ids=lambda p: p.point_id)
def test_down_and_out_call_method_of_images(p):
    S, K, H, T, r, q, v = p.spot, p.strike, p.barrier, p.T, p.rate, p.div, p.vol
    image = cf.european(S, K, T, r, q, v, "call") - (S / H) ** (1 - 2 * (r - q) / (v * v)) * cf.european(
        H * H / S, K, T, r, q, v, "call"
    )
    assert cf.barrier(S, K, H, T, r, q, v, "call", "down-and-out") == pytest.approx(image, rel=1e-9, abs=1e-11)


@pytest.mark.parametrize("p", [b for b in BARRIERS if b.barrier_type == "up-and-out" and b.option == "put"
                               and b.barrier >= b.strike], ids=lambda p: p.point_id)
def test_up_and_out_put_method_of_images(p):
    S, K, H, T, r, q, v = p.spot, p.strike, p.barrier, p.T, p.rate, p.div, p.vol
    image = cf.european(S, K, T, r, q, v, "put") - (S / H) ** (1 - 2 * (r - q) / (v * v)) * cf.european(
        H * H / S, K, T, r, q, v, "put"
    )
    assert cf.barrier(S, K, H, T, r, q, v, "put", "up-and-out") == pytest.approx(image, rel=1e-9, abs=1e-11)


def test_greek_signs():
    for p in POINTS:
        S, K, T, r, q, v = _args(p)
        g = cf.european_greeks(S, K, T, r, q, v, p.option)
        assert g["gamma"] >= 0 and g["vega"] >= 0
        assert (0 <= g["delta"] <= 1) if p.option == "call" else (-1 <= g["delta"] <= 0)
    assert np.isfinite(cf.norm_cdf(-40.0))


def test_references_do_not_use_quantlib():
    from pathlib import Path

    import pricing_crosscheck.reference as ref_pkg

    for path in Path(ref_pkg.__file__).parent.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "import QuantLib" not in text and "from QuantLib" not in text, path.name
