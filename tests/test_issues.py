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

- Issue #15: DIA-NN's report.tsv is long format and only carries a row where the
  precursor was identified, so read_input returned a ragged curve -- the runs a
  peptide dropped out of were absent rather than zero. read_input now completes
  the peptide x run grid, and figuresofmerit.csv reports the distinct level count.
"""

import importlib
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "bin"))

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


# --- Issue #15: long-format DIA-NN report.tsv is not dense --------------------
LEVELS = [1, 0.7, 0.5, 0.3, 0.1, 0.07, 0.05, 0.03, 0.01, 0.007, 0.005, 0.003, 0.001, 0]
REPS = 3
SATURATING = "SATURATINGPEPTIDEK2"   # saturates at the top, drops out at the bottom
COMPLETE = "COMPLETEPEPTIDEK2"       # identified in every run


def _write_diann_report(report_path, map_path, id_floor=0.07, ceiling=3.0e5,
                        slope=1.0e6, seed=0):
    """Write a synthetic long-format diann_report.tsv plus its concentration map.

    SATURATING is only reported at/above `id_floor` -- exactly the DIA-NN behaviour
    that makes the curve ragged. COMPLETE is reported everywhere.
    """
    rng = np.random.default_rng(seed)
    rows, maprows = [], []
    for level in LEVELS:
        for rep in range(1, REPS + 1):
            fname = f"run_{level}_{rep}.mzML"
            maprows.append({"filename": fname, "concentration": level})
            for peptide, saturates in ((SATURATING, True), (COMPLETE, False)):
                area = slope * level + rng.normal(0, 2.0e3)
                if saturates:
                    area = min(area, ceiling + rng.normal(0, 2.0e3))
                    if level < id_floor:
                        continue  # not identified -> DIA-NN emits no row at all
                rows.append({"File.Name": fname,
                             "Precursor.Id": peptide,
                             "Stripped.Sequence": peptide[:-1],
                             "Precursor.Quantity": max(area, 0.0)})

    pd.DataFrame(rows).to_csv(report_path, sep="\t", index=False)
    pd.DataFrame(maprows).to_csv(map_path, index=False)
    return len(maprows)


@pytest.fixture
def calc():
    return importlib.import_module("calculate-loq")


def test_diann_report_is_densified(tmp_path, calc):
    """Every peptide spans every mapped run; unidentified runs read as area 0."""
    report = tmp_path / "diann_report.tsv"
    conc_map = tmp_path / "conc_map.csv"
    n_runs = _write_diann_report(report, conc_map)

    df = calc.read_input(str(report), str(conc_map))

    for peptide in (SATURATING, COMPLETE):
        sub = df[df["peptide"] == peptide]
        assert len(sub) == n_runs, f"{peptide}: {len(sub)} rows, expected {n_runs}"
        assert sub["curvepoint"].nunique() == len(LEVELS)

    # the dropped-out low-concentration runs come back as zeros, not as gaps
    dropped = df[(df["peptide"] == SATURATING) & (df["curvepoint"] < 0.07)]
    assert not dropped.empty
    assert (dropped["area"] == 0).all()

    # ...and the peptide that was identified everywhere keeps its real areas
    # (0.01 rather than the bottom level, where simulated noise can legitimately
    # clamp a reported area to 0 and make the check ambiguous)
    low = df[(df["peptide"] == COMPLETE) & (df["curvepoint"] == 0.01)]
    assert (low["area"] > 0).all()


def test_densified_report_recovers_uloq(tmp_path, calc):
    """With the grid completed, the saturating peptide gets a finite LOD and ULOQ.

    Read ragged (as the reader used to), the same curve has too few distinct
    levels for the 'auto' model to keep a saturation segment, so ULOQ is inf.
    """
    report = tmp_path / "diann_report.tsv"
    conc_map = tmp_path / "conc_map.csv"
    _write_diann_report(report, conc_map)

    df = calc.read_input(str(report), str(conc_map))
    sub = df[df["peptide"] == SATURATING].sort_values("curvepoint")

    def fit(frame):
        x = np.asarray(frame["curvepoint"], dtype=float)
        y = np.asarray(frame["area"], dtype=float)
        result, _ = calc.fit_by_lmfit_yang(x, y, "auto", min_saturation_points=3)
        slope = result.params["a"].value
        intercept = result.params["b"].value
        c_high = result.params["c_high"].value
        lod = calc.calculate_lod(
            np.asarray([0.0, result.params["c"].value, slope, intercept]),
            frame, 2.0, 2, 3, x, "auto")[0]
        uloq = calc.calculate_uloq(slope, intercept, c_high, frame, 2.0, 3)
        return lod, uloq

    lod, uloq = fit(sub)
    assert np.isfinite(uloq), "densified curve should still resolve a ULOQ"
    assert np.isfinite(lod), "densified curve should still resolve an LOD"

    # the ragged view of the very same data resolves neither
    ragged = sub[sub["area"] > 0]
    ragged_lod, ragged_uloq = fit(ragged)
    assert np.isinf(ragged_uloq)
    assert np.isinf(ragged_lod)


# --- Issue #16: every input normalized alike; bootstrap must not depend on order --
def test_reader_output_is_canonically_sorted(calc, tmp_path):
    """read_input returns rows in canonical (peptide, curvepoint, area) order."""
    report = tmp_path / "diann_report.tsv"
    conc_map = tmp_path / "conc_map.csv"
    _write_diann_report(report, conc_map)

    df = calc.read_input(str(report), str(conc_map))
    expected = df.sort_values(["peptide"] + calc.SORT_KEYS, kind="mergesort")
    assert df.equals(expected.reset_index(drop=True))


def test_figures_of_merit_are_row_order_independent(calc):
    """Shuffling a peptide's rows must not change any figure of merit.

    The bootstrap resamples by position (df.sample draws positional indices), so
    before the canonical sort a permuted frame produced a different resample and
    moved the LOQ -- by up to 240% on this dataset.
    """
    df = calc.read_input(CURVE_DATA, CONC_MAP)
    rng = np.random.default_rng(0)

    # a peptide whose LOQ was among the least stable under permutation
    peptide = "GEGFMVVTATGDNTFVGR"
    base = df[df["peptide"] == peptide]
    assert not base.empty, f"{peptide} missing from the sample dataset"

    def figures(subset):
        row = calc.process_peptide(50, 0.2, None, peptide, "n", 2.0, 2, 1, 2,
                                   subset, "n", "auto")
        return row.iloc[0]

    reference = figures(base)
    for _ in range(3):
        shuffled = base.iloc[rng.permutation(len(base))]
        got = figures(shuffled)
        for col in ("LOD", "LOQ", "ULOQ", "slope_linear", "intercept_linear"):
            assert np.isclose(got[col], reference[col], rtol=1e-12, equal_nan=True), (
                f"{col} changed with row order: {reference[col]} -> {got[col]}"
            )


def test_figuresofmerit_reports_level_count(tmp_path):
    """figuresofmerit.csv carries n_curvepoints so 'no saturation' is separable
    from 'too few levels to look for one'."""
    out = tmp_path / "out"
    out.mkdir()
    proc = _run({"extra": ["--n_threads", "1"]}, out)
    assert proc.returncode == 0, proc.stderr

    fom = pd.read_csv(os.path.join(str(out), "figuresofmerit.csv"))
    assert "n_curvepoints" in fom.columns
    # the sample dataset is a dense 14-point curve for every peptide
    assert (fom["n_curvepoints"] == 14).all()
