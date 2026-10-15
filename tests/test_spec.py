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
    s = spec.seed_for("call|K=100", "own_mc")
    assert s == spec.seed_for("call|K=100", "own_mc")
    assert s != spec.seed_for("call|K=100", "ql_mc")
    assert 1 <= s <= 2147483646


def test_tolerance_classes():
    assert spec.tolerance("numerical", 5.0) == 1e-3
    assert spec.tolerance("numerical", 100.0) == 1e-2
    assert spec.tolerance("exact", 0.5) == 1e-9


def test_values_stated_in_protocol():
    """docs/protocol.md quotes these numbers; keep the two in sync."""
    s = spec.load()
    tol = s["tolerances"]
    assert tol["exact"] == {"abs": 1e-9, "rel": 1e-9}
    assert tol["numerical"] == {"abs": 1e-3, "rel": 1e-4}
    assert tol["mc_sigmas"] == 4.0 and tol["reference_uncertainty_fraction"] == 0.1
    assert tol["greeks"]["abs"] == {"delta": 1e-4, "gamma": 1e-5, "vega": 1e-3, "theta": 1e-3}
    assert s["engines"]["american"]["ref_pde"]["levels"] == [[800, 400], [1600, 800], [3200, 1600]]
    assert s["seeds"]["base"] == 20250115
    assert s["market"]["valuation_date"] == "2025-01-15" and s["market"]["spot"] == 100.0
