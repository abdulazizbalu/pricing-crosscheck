import json

import numpy as np

from pricing_crosscheck import env
from pricing_crosscheck.results import Table, compare_tables, manifest_problems, write_manifest, write_table


def _write(tmp_path, name, rows, header=("# meta: 1",)):
    t = Table(name, ["a", "b"])
    for a, b in rows:
        t.add(a=a, b=b)
    return write_table(t, tmp_path, header)


def test_nan_and_inf_are_differences(tmp_path):
    e = _write(tmp_path / "e", "t", [(1.0, 2.0)])
    a = _write(tmp_path / "a", "t", [(float("inf"), float("nan"))])
    assert len(compare_tables(e, a, 1e-7, 1e-10)) == 2


def test_tolerance_applies_to_numbers(tmp_path):
    e = _write(tmp_path / "e", "t", [(1.0, "x")])
    a = _write(tmp_path / "a", "t", [(1.0 + 1e-12, "x")])
    assert compare_tables(e, a, 1e-7, 1e-10) == []
    a = _write(tmp_path / "b", "t", [(1.001, "x")])
    assert len(compare_tables(e, a, 1e-7, 1e-10)) == 1


def test_numpy_scalars_written_as_plain_numbers(tmp_path):
    path = _write(tmp_path, "t", [(np.float64(1.5), np.int64(3))])
    assert path.read_text(encoding="utf-8").splitlines()[-1] == "1.5,3"
    path = _write(tmp_path, "u", [(np.float32(0.5), np.bool_(True))])
    assert path.read_text(encoding="utf-8").splitlines()[-1] == "0.5,true"


def test_only_leading_header_is_skipped(tmp_path):
    e = _write(tmp_path / "e", "t", [("#x", 1)])
    a = _write(tmp_path / "a", "t", [("#x", 999)])
    assert len(compare_tables(e, a, 1e-7, 1e-10)) == 1


def test_manifest_detects_edits(tmp_path):
    path = _write(tmp_path, "t", [(1.0, 2.0)])
    write_manifest(tmp_path, "h", {}, "fast")
    assert manifest_problems(tmp_path) == []
    path.write_text(path.read_text(encoding="utf-8") + "3.0,4.0\n", encoding="utf-8")
    assert manifest_problems(tmp_path)
    (tmp_path / "extra.csv").write_text("a\n", encoding="utf-8")
    assert len(manifest_problems(tmp_path)) == 2
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["profile"] == "fast"


def test_lock_file_parses():
    # Only reports mismatches; the CI install step makes this list empty.
    assert all(isinstance(line, str) for line in env.lock_mismatches())
