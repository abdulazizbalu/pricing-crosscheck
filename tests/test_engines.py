"""Engine wrappers, failure handling, determinism and identity bookkeeping (PR scope: engines)."""

import math

import pytest

from pricing_crosscheck import pricing, spec
from pricing_crosscheck.engines import own
from pricing_crosscheck.engines import quantlib as qle
from pricing_crosscheck.identities import identity_table
from pricing_crosscheck.reference import closed_form as cf
from pricing_crosscheck.spec import BarrierPoint, VanillaPoint

P = VanillaPoint("call", 100.0, 365, 0.3, 0.05, 0.03)


def test_quantlib_analytic_matches_closed_form_conventions():
    """Same T, curves and units on both sides: the analytic engine reproduces cf."""
    r = qle.price_vanilla(P, "european", "ql_analytic", want_greeks=True)
    assert r.ok
    assert r.value == pytest.approx(cf.european(100, 100, 1.0, 0.05, 0.03, 0.3, "call"), abs=1e-12)
    g = cf.european_greeks(100, 100, 1.0, 0.05, 0.03, 0.3, "call")
    for name in g:
        assert r.greeks[name] == pytest.approx(g[name], rel=1e-10), name


def test_refused_payoff_is_not_available():
    r = qle.price_vanilla(P, "digital", "ql_crr")
    assert not r.ok and r.status.startswith("not available")
    assert math.isnan(r.value)


def test_monte_carlo_needs_positive_seed():
    with pytest.raises(ValueError):
        qle.price_vanilla(P, "european", "ql_mc", seed=0)


def test_reference_failure_raises():
    bad = VanillaPoint("call", -1.0, 365, 0.3, 0.05, 0.0)
    with pytest.raises((ValueError, RuntimeError)):
        own.price_vanilla(bad, "european", "cf")


def test_own_engine_failure_is_recorded():
    r = own.price_barrier(BarrierPoint("down-and-in", "call", 90.0, 100.0, 30, 0.3, 0.05, 0.02), "own_pde")
    assert r.status.startswith("not available")


def test_mc_seed_rule_and_determinism():
    a = pricing.price("european", P, "ql_mc")
    b = pricing.price("european", P, "ql_mc")
    assert a.value == b.value and a.std_error == b.std_error
    c = pricing.price("european", P, "ql_mc", run="refine=4")
    assert c.value != a.value


def test_parallel_results_do_not_depend_on_jobs():
    tasks = pricing.tasks_for("fast", ["european"])[:8]
    one = pricing.parallel_map(pricing._price_point, tasks, 1)
    two = pricing.parallel_map(pricing._price_point, tasks, 2)
    assert repr(one) == repr(two)  # repr: NaN compares unequal to itself


def test_mc_subsets_have_references():
    for instrument in ("american", "barrier"):
        tasks = pricing.tasks_for("fast", [instrument])
        ref = pricing.REFERENCES[instrument]
        for _, p, engines in tasks:
            if any(e in engines for e in ("ql_mc", "own_mc", "ql_mc_lsm")):
                assert ref in engines, p.point_id


def test_identity_table_on_fast_european_and_digital():
    from pricing_crosscheck.results import Table

    t = Table("prices", pricing.PRICE_COLUMNS)
    for instrument in ("european", "digital"):
        for p in spec.vanilla_points("fast")[:4] + spec.vanilla_points("fast")[48:52]:
            for e in ("cf", "ql_analytic"):
                t.add(**pricing._row(instrument, p, e, pricing.price(instrument, p, e)))
    ids = identity_table(t, "fast")
    parity = [r for r in ids.rows if r["identity"] in ("put_call_parity", "digital_call_plus_put")]
    assert parity and all(r["passed"] for r in parity)
