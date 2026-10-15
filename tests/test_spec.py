import pytest

from pricing_crosscheck import spec


def test_grid_sizes_match_protocol():
    assert len(spec.vanilla_points("full")) == 2 * 840
    assert len(spec.barrier_points("full")) == 8 * 216  # 4 barrier types x call/put x 216
    assert len(spec.vanilla_points("fast")) == 2 * 48


def test_barriers_on_live_side():
    for p in spec.barrier_points("full"):
        assert (p.barrier < p.spot) if p.is_down else (p.barrier > p.spot)


def test_point_ids_unique():
    ids = [p.point_id for p in spec.vanilla_points("full")]
    assert len(ids) == len(set(ids))
    ids = [p.point_id for p in spec.barrier_points("full")]
    assert len(ids) == len(set(ids))


def test_seeds_deterministic_and_valid():
    s = spec.seed_for("european", "call|K=100", "own_mc", "n=1024")
    assert s == spec.seed_for("european", "call|K=100", "own_mc", "n=1024")
    assert s != spec.seed_for("european", "call|K=100", "ql_mc", "n=1024")
    assert s != spec.seed_for("digital", "call|K=100", "own_mc", "n=1024")
    assert s != spec.seed_for("european", "call|K=100", "own_mc", "n=4096")
    assert 1 <= s <= 2147483646


def test_tolerance_classes():
    assert spec.tolerance("numerical", 5.0) == 1e-3
    assert spec.tolerance("numerical", 100.0) == 1e-2
    assert spec.tolerance("numerical", 0.5, "digital") == 1e-4 * 0.5
    assert spec.tolerance("numerical", 0.01, "digital") == 1e-5
    assert spec.tolerance("exact", 0.5) == 1e-9
    assert spec.greek_tolerance("delta", 0.5, "european") == pytest.approx(5e-4)
    assert spec.greek_tolerance("gamma", 0.0, "digital") == 1e-7


def test_unknown_profile_rejected():
    import pytest

    with pytest.raises(ValueError):
        spec.vanilla_points("Fast")


def test_values_stated_in_protocol():
    """docs/protocol.md quotes these numbers; keep the two in sync."""
    s = spec.load()
    tol = s["tolerances"]
    assert tol["exact"] == {"abs": 1e-9, "rel": 1e-9}
    assert tol["numerical"] == {"abs_per_unit_notional": 1e-5, "rel": 1e-4}
    assert tol["notional"] == {"european": 100.0, "american": 100.0, "barrier": 100.0, "digital": 1.0}
    assert tol["mc_sigmas"] == 4.0 and tol["reference_uncertainty_fraction"] == 0.1
    assert tol["greeks"]["abs_per_unit_notional"] == {"delta": 1e-6, "gamma": 1e-7, "vega": 1e-5, "theta": 1e-5}
    assert s["engines"]["barrier"]["ql_mc"]["isBiased"] is False
    assert s["refinement"]["min_local_order"] == 0.5
    assert s["engines"]["american"]["ref_pde"]["levels"] == [[800, 400], [1600, 800], [3200, 1600]]
    assert s["seeds"]["base"] == 20250115
    assert s["market"]["valuation_date"] == "2025-01-15" and s["market"]["spot"] == 100.0


def test_run_tag_and_near_barrier_label():
    assert spec.run_tag(samples=1024) == "n=1024"
    assert spec.run_tag(samples=4, steps_per_year=12, refine=2) == "n=4;steps_per_year=12;refine=2"
    limit = spec.load()["labels"]["near_barrier_max_rel_distance"]
    assert abs(99.0 / 100.0 - 1) <= limit and abs(101.0 / 100.0 - 1) <= limit
    assert abs(98.0 / 100.0 - 1) > limit
