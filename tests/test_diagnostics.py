from pricing_crosscheck import diagnostics as dg


def test_lr_collapse_matches_observed_points():
    # Points where QuantLib 1.43's LR engine returned 0 (results/full/comparisons.csv).
    assert dg.lr_collapsed("put", 100.0, 200.0, 1 / 365, 0.0, 0.0, 0.01, 1001)
    assert dg.lr_collapsed("put", 100.0, 120.0, 1 / 365, 0.0, 0.0, 0.01, 1001)
    # Same point is fine at 4005 steps, where QuantLib also agrees with the closed form.
    assert not dg.lr_collapsed("put", 100.0, 120.0, 1 / 365, 0.0, 0.0, 0.01, 4005)
    # Calls round to pu = 1 instead and are not flagged.
    assert not dg.lr_collapsed("call", 100.0, 50.0, 1 / 365, 0.0, 0.0, 0.01, 1001)
    assert dg.lr_up_probability_v143(100.0, 50.0, 1 / 365, 0.0, 0.0, 0.01, 1001) == 1.0


def test_threshold_grows_like_sqrt_n():
    t1, t4 = dg.lr_collapse_threshold(1001), dg.lr_collapse_threshold(4005)
    assert 190 < t1 < 200 and abs(t4 / t1 - 2.0) < 0.01
