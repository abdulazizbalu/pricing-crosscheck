"""Convergence of our numerical references to the closed forms (docs/protocol.md section 7)."""

import math

import numpy as np
import pytest

from pricing_crosscheck import spec
from pricing_crosscheck.convergence import fitted_order
from pricing_crosscheck.reference import closed_form as cf
from pricing_crosscheck.reference import mc, pde, tree

SPEC = spec.load()
CONV = SPEC["convergence"]
VAL = SPEC["validation"]
POINTS = CONV["points"]
BARRIER_POINTS = CONV["barrier_points"]
OPTIONS = ["call", "put"]


def _args(p):
    return 100.0, p["strike"], p["days"] / 365.0, p["rate"], p["div"], p["vol"]


def _assert_second_order(nodes, errors, ref, instrument):
    fit = fitted_order(nodes, errors, ref)
    lo, hi = VAL["pde_order_range"]
    if fit.measurable:
        assert lo <= fit.order <= hi, (fit, errors)
    else:  # protocol section 7 fallback, reported in the measurements
        tol = spec.tolerance("numerical", ref, instrument)
        assert all(abs(e) <= tol for n, e in zip(nodes, errors) if n >= VAL["fallback_min_resolution"])


def _even_tree_steps():
    return [n for n in CONV["tree_steps"] if n % 2 == 0]


@pytest.mark.parametrize("p", POINTS, ids=lambda p: p["id"])
@pytest.mark.parametrize("option", OPTIONS)
def test_crr_european_first_order(p, option):
    S, K, T, r, q, v = _args(p)
    ref = cf.european(S, K, T, r, q, v, option)
    steps = _even_tree_steps()
    errors = [tree.crr(S, K, T, r, q, v, option, n) - ref for n in steps]
    fit = fitted_order(steps, errors, ref)
    assert fit.order >= VAL["crr_min_order"]
    earlier = max(abs(e) * n for e, n in zip(errors[:-1], steps[:-1]))
    assert abs(errors[-1]) * steps[-1] <= VAL["crr_scaled_error_factor"] * earlier


@pytest.mark.parametrize("p", POINTS, ids=lambda p: p["id"])
@pytest.mark.parametrize("option", OPTIONS)
@pytest.mark.parametrize("payoff", ["vanilla", "digital"])
def test_pde_second_order(p, option, payoff):
    S, K, T, r, q, v = _args(p)
    price = cf.european if payoff == "vanilla" else cf.digital
    ref = price(S, K, T, r, q, v, option)
    nodes = CONV["pde_nodes"]
    errors = [pde.vanilla(S, K, T, r, q, v, option, n, n, payoff=payoff).value - ref for n in nodes]
    _assert_second_order(nodes, errors, ref, "european" if payoff == "vanilla" else "digital")


@pytest.mark.parametrize("p", BARRIER_POINTS, ids=lambda p: p["id"])
def test_pde_knock_out_second_order(p):
    S, K, T, r, q, v = _args(p)
    H, opt, kind = p["barrier"], p["option"], p["barrier_type"]
    ref = cf.barrier(S, K, H, T, r, q, v, opt, kind)
    nodes = CONV["pde_nodes"]
    runs = [pde.knock_out(S, K, H, T, r, q, v, opt, kind.startswith("down"), n, n) for n in nodes]
    # Order is fitted on the node counts actually used (strike alignment adjusts them).
    _assert_second_order([x.nodes for x in runs], [x.value - ref for x in runs], ref, "barrier")


def test_pde_misaligned_strike_loses_order():
    """Control experiment: without strike alignment and payoff smoothing the digital is first order."""
    S, K, T, r, q, v = 100.0, 100.0, 1.0, 0.05, 0.03, 0.3
    ref = cf.digital(S, K, T, r, q, v, "call")
    nodes = CONV["pde_nodes"]
    errors = [
        pde.vanilla(S, K, T, r, q, v, "call", n, n, payoff="digital", align_strike=False, smooth_payoff=False).value
        - ref
        for n in nodes
    ]
    assert fitted_order(nodes, errors, ref).order < VAL["control_max_order_unaligned_digital"]


@pytest.mark.parametrize("p", POINTS, ids=lambda p: p["id"])
@pytest.mark.parametrize("payoff", ["vanilla", "digital"])
def test_mc_unbiased_with_root_n_error(p, payoff):
    S, K, T, r, q, v = _args(p)
    price = cf.european if payoff == "vanilla" else cf.digital
    for option in OPTIONS:
        ref = price(S, K, T, r, q, v, option)
        sizes = [2**k for k in CONV["mc_samples_log2"]]
        res = [mc.terminal(S, K, T, r, q, v, option, n, spec.seed_for(payoff, f"{p['id']}|{option}", "own_mc", spec.run_tag(samples=n)), payoff)
               for n in sizes]
        for x in res:
            assert abs(x.value - ref) <= 4.0 * x.std_error
        last = CONV["fit_last_levels"]
        slope = np.polyfit(np.log(sizes[-last:]), np.log([x.std_error for x in res][-last:]), 1)[0]
        lo, hi = VAL["mc_se_slope_range"]
        assert lo <= slope <= hi


@pytest.mark.parametrize("p", BARRIER_POINTS, ids=lambda p: p["id"])
@pytest.mark.parametrize("knock_in", [False, True])
def test_mc_barrier_with_bridge_unbiased(p, knock_in):
    S, K, T, r, q, v = _args(p)
    H, opt = p["barrier"], p["option"]
    kind = p["barrier_type"].replace("-out", "-in") if knock_in else p["barrier_type"]
    ref = cf.barrier(S, K, H, T, r, q, v, opt, kind)
    for per_year in CONV["mc_monitoring_steps_per_year"]:
        steps = max(1, round(per_year * T))
        seed = spec.seed_for("barrier", f"{p['id']}|{kind}", "own_mc", spec.run_tag(samples=65536, steps_per_year=per_year))
        x = mc.barrier(S, K, H, T, r, q, v, opt, kind, 65536, seed, steps, bridge=True)
        assert abs(x.value - ref) <= 4.0 * x.std_error, (per_year, x, ref)


def test_mc_barrier_without_bridge_is_biased():
    """Control experiment: discrete monitoring without the bridge overprices a knock-out."""
    p = BARRIER_POINTS[0]
    S, K, T, r, q, v = _args(p)
    ref = cf.barrier(S, K, p["barrier"], T, r, q, v, p["option"], p["barrier_type"])
    x = mc.barrier(S, K, p["barrier"], T, r, q, v, p["option"], p["barrier_type"], 65536, 11, 12, bridge=False)
    assert x.value - ref > 4.0 * x.std_error


def _ref_pde(S, K, T, r, q, v, option):
    levels = SPEC["engines"]["american"]["ref_pde"]["levels"]
    vals = [pde.vanilla(S, K, T, r, q, v, option, m, n, american=True).value for m, n in levels]
    return vals[-1], abs(vals[-1] - vals[-2])


@pytest.mark.parametrize("p", POINTS, ids=lambda p: p["id"])
@pytest.mark.parametrize("option", OPTIONS)
def test_american_reference_converged_and_matches_tree(p, option):
    S, K, T, r, q, v = _args(p)
    value, u = _ref_pde(S, K, T, r, q, v, option)
    tol = spec.tolerance("numerical", value, "american")
    frac = SPEC["tolerances"]["reference_uncertainty_fraction"]
    assert u <= frac * tol
    assert abs(tree.crr_averaged(S, K, T, r, q, v, option, 6400, american=True) - value) <= tol
    assert value >= cf.european(S, K, T, r, q, v, option) - tol


@pytest.mark.parametrize("r", [0.0, 0.05])
def test_american_call_without_dividends_equals_european(r):
    S, K, T, q, v = 100.0, 100.0, 1.0, 0.0, 0.3
    value, _ = _ref_pde(S, K, T, r, q, v, "call")
    assert abs(value - cf.european(S, K, T, r, q, v, "call")) <= spec.tolerance("numerical", value, "american")


@pytest.mark.parametrize("r", [0.0, -0.02])
def test_american_put_with_nonpositive_rate_equals_european(r):
    S, K, T, q, v = 100.0, 100.0, 1.0, 0.03, 0.3
    value, _ = _ref_pde(S, K, T, r, q, v, "put")
    assert abs(value - cf.european(S, K, T, r, q, v, "put")) <= spec.tolerance("numerical", value, "american")


def test_crr_rejects_invalid_probability():
    with pytest.raises(ValueError):
        tree.crr(100.0, 100.0, 10.0, 0.05, 0.0, 0.001, "call", 10)


def test_mc_is_deterministic():
    a = mc.terminal(100.0, 100.0, 1.0, 0.05, 0.0, 0.3, "call", 1024, 7)
    b = mc.terminal(100.0, 100.0, 1.0, 0.05, 0.0, 0.3, "call", 1024, 7)
    assert a == b
    assert not math.isnan(a.std_error)


@pytest.mark.parametrize("option,K", [("call", 80.0), ("put", 100.0), ("put", 120.0)])
def test_american_reference_degenerate_zero_rates(option, K):
    """r = q = 0: early exercise is never strictly optimal, so value - payoff is at
    rounding level over a whole region; the penalty iteration must still terminate
    and reproduce the European price."""
    S, T, v = 100.0, 7 / 365, 0.1
    value, _ = _ref_pde(S, K, T, 0.0, 0.0, v, option)
    assert abs(value - cf.european(S, K, T, 0.0, 0.0, v, option)) <= spec.tolerance("numerical", value, "american")


# Points where an earlier version of the penalty iteration failed to terminate
# (independent review of the references): r = q = 0, deep in the money short-dated
# calls, long-dated high-vol puts, negative rates.
REGRESSION_AMERICAN = [
    ("put", 100.0, 7, 0.3, 0.0, 0.0), ("call", 50.0, 1, 0.01, 0.05, 0.03), ("call", 50.0, 1, 0.3, 0.05, 0.03),
    ("put", 50.0, 3650, 1.0, 0.0, 0.03), ("put", 200.0, 3650, 1.0, 0.0, 0.03), ("call", 100.0, 365, 0.3, -0.02, 0.0),
]


@pytest.mark.parametrize("option,K,D,v,r,q", REGRESSION_AMERICAN)
def test_american_reference_regression_points(option, K, D, v, r, q):
    value, u = _ref_pde(100.0, K, D / 365, r, q, v, option)
    euro = cf.european(100.0, K, D / 365, r, q, v, option)
    assert math.isfinite(value) and value >= euro - spec.tolerance("numerical", value, "american")


def test_pde_call_long_dated_high_vol():
    """Calls go through put-call symmetry; the log-forward grid error for S-linear
    payoffs at large sigma^2 T (independent review) must not appear."""
    a = (100.0, 50.0, 10.0, -0.02, 0.0, 1.0, "call")
    assert abs(pde.vanilla(*a, 400, 400).value - cf.european(*a)) <= spec.tolerance("numerical", cf.european(*a))


# Barrier cases using the B and D terms of the closed form (strike on the other side of the barrier).
BD_POINTS = [
    ("down-and-out", "call", 95.0, 90.0), ("up-and-out", "put", 105.0, 110.0),
    ("down-and-out", "put", 90.0, 95.0), ("up-and-out", "call", 110.0, 105.0),
]


@pytest.mark.parametrize("kind,opt,H,K", BD_POINTS)
def test_closed_form_barrier_b_d_terms_against_pde(kind, opt, H, K):
    S, T, r, q, v = 100.0, 1.0, 0.05, 0.02, 0.3
    ref = cf.barrier(S, K, H, T, r, q, v, opt, kind)
    nodes = CONV["pde_nodes"]
    runs = [pde.knock_out(S, K, H, T, r, q, v, opt, kind.startswith("down"), n, n) for n in nodes]
    # Order is fitted on the node counts actually used (strike alignment adjusts them).
    _assert_second_order([x.nodes for x in runs], [x.value - ref for x in runs], ref, "barrier")


@pytest.mark.parametrize("kind,opt,H,K", BD_POINTS)
def test_closed_form_barrier_b_d_terms_against_mc(kind, opt, H, K):
    S, T, r, q, v = 100.0, 1.0, 0.05, 0.02, 0.3
    for k in (kind, kind.replace("-out", "-in")):
        ref = cf.barrier(S, K, H, T, r, q, v, opt, k)
        x = mc.barrier(S, K, H, T, r, q, v, opt, k, 65536, spec.seed_for("barrier", f"bd|{k}|{opt}|{H}|{K}", "own_mc"),
                       365)
        assert abs(x.value - ref) <= 4.0 * x.std_error


def test_american_long_maturity_against_extrapolated_tree():
    """Long maturity: compare with Richardson-extrapolated CRR (2 V(2N) - V(N), averaged trees)."""
    a = (100.0, 105.0, 10.0, 0.05, 0.0, 0.1, "put")
    value, _ = _ref_pde(*a)
    rich = 2 * tree.crr_averaged(*a, 12800, american=True) - tree.crr_averaged(*a, 6400, american=True)
    assert abs(value - rich) <= spec.tolerance("numerical", value, "american")
