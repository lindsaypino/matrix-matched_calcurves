"""
Regression tests for specific reported issues in calculate-loq.py.

- Issue #11: a worker crash surfaced only as an opaque BrokenProcessPool and
  produced near-empty output. The tool now supports a serial mode
  (``--n_threads 1``) and falls back to serial on BrokenProcessPool. Here we
  test that serial mode runs and produces output identical to the parallel
  golden snapshot (bootstrap replicates are seeded per-index, so serial and
  parallel are numerically identical).

- Issue #12: warn when the filename/concentration map is blank or has
  unannotated rows.
"""

import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SCRIPT = os.path.join(REPO, "bin", "calculate-loq.py")
CURVE_DATA = os.path.join(REPO, "data", "one_protein.csv")
CONC_MAP = os.path.join(REPO, "data", "filename2samplegroup_map.csv")
GOLDEN = os.path.join(HERE, "data", "golden_figuresofmerit.csv")

NUMERIC_COLS = [
    "LOD", "LOQ", "slope_linear", "intercept_linear", "intercept_noise", "stndev_noise",
]


def _run(args, output_dir):
    """Run calculate-loq.py; return (CompletedProcess). Never raises on nonzero."""
    cmd = [
        sys.executable, SCRIPT, CURVE_DATA,
        args.pop("conc_map", CONC_MAP),
        "--output_path", str(output_dir),
        "--plot", "n",
        "--bootreps", "100",
    ] + args.get("extra", [])
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)


# --- Issue #11: serial mode ---------------------------------------------------
def test_serial_mode_matches_parallel_golden(tmp_path):
    """--n_threads 1 runs in-process and yields the same numbers as the golden."""
    proc = _run({"extra": ["--n_threads", "1"]}, tmp_path)
    assert proc.returncode == 0, f"serial run failed:\n{proc.stderr}"

    actual = pd.read_csv(os.path.join(str(tmp_path), "figuresofmerit.csv"))
    actual = actual.sort_values("peptide").reset_index(drop=True)
    golden = pd.read_csv(GOLDEN).sort_values("peptide").reset_index(drop=True)

    assert list(actual["peptide"]) == list(golden["peptide"])
    for col in NUMERIC_COLS:
        a = actual[col].to_numpy(dtype=float)
        g = golden[col].to_numpy(dtype=float)
        assert np.allclose(a, g, rtol=1e-9, atol=0.0, equal_nan=True), (
            f"serial output for {col} differs from golden"
        )


# --- Issue #12: blank / unannotated concentration map -------------------------
def _write_map(path, rows):
    """Write a filename,concentration CSV from a list of (filename, conc) tuples."""
    lines = ["filename,concentration"] + [f"{f},{c}" for f, c in rows]
    path.write_text("\n".join(lines) + "\n")


def test_blank_concentration_rows_warn(tmp_path):
    """Rows with a blank concentration are reported by filename on stderr."""
    src = pd.read_csv(CONC_MAP)
    rows = list(src.itertuples(index=False, name=None))
    # blank out the concentration on the first two rows
    blanked = [str(rows[0][0]), str(rows[1][0])]
    rows = [(f, "" if i < 2 else c) for i, (f, c) in enumerate(rows)]

    map_path = tmp_path / "map_2blank.csv"
    _write_map(map_path, rows)

    proc = _run({"conc_map": str(map_path)}, tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "blank/unannotated" in proc.stderr
    for fname in blanked:
        assert fname in proc.stderr, f"expected blanked filename {fname} in warning"


def test_empty_map_warns(tmp_path):
    """A header-only (empty) map warns that no curve points will be mapped."""
    map_path = tmp_path / "map_empty.csv"
    map_path.write_text("filename,concentration\n")

    proc = _run({"conc_map": str(map_path)}, tmp_path)
    assert "blank" in proc.stderr.lower() or "missing" in proc.stderr.lower()
