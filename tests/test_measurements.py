"""Bookkeeping of the measurement stages (labels, README block); numbers are checked by the fast CI job."""

from pricing_crosscheck import compare, report
from pricing_crosscheck.results import Table


def test_region_labels():
    row = {"days": 1, "vol": 0.01, "rate": -0.02, "div": 0.0, "strike": 200.0, "barrier": 99.9}
    labels = compare.labels(row)
    assert {"near_expiry", "near_zero_vol", "negative_rate", "deep_moneyness", "near_barrier"} <= set(labels)
    assert compare.labels({"days": 365, "vol": 0.3, "rate": 0.05, "div": 0.0, "strike": 100.0, "barrier": ""}) == []


def test_readme_block_is_replaced(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{report.README_START}\nold\n{report.README_END}\noutro\n", encoding="utf-8")
    summary = Table("summary", compare.SUMMARY_COLUMNS)
    dis = Table("disagreements", ["instrument", "engine", "category"])
    report.update_readme({"summary": summary, "disagreements": dis}, readme)
    text = readme.read_text(encoding="utf-8")
    assert "old" not in text and text.startswith("intro") and text.rstrip().endswith("outro")
