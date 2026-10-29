"""
`benchmarks.py` -- run benchmarks on `calculate_loq.py`
"""

import os
import sys

sys.path.append("bin")

import importlib
import numpy as np
import pytest
import subprocess

# every test in this module needs the `benchmark` fixture; skip cleanly instead of
# erroring at setup when pytest-benchmark is not installed
pytest.importorskip("pytest_benchmark")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def mock_dataset():
    return (os.path.join(REPO, "data", "one_protein.csv"),
            os.path.join(REPO, "data", "filename2samplegroup_map.csv"))


def bench_calculate_loq(tmp_path, dataset):
    cmd = [
        sys.executable, os.path.join(REPO, "bin", "calculate-loq.py"),
        "--output_path", tmp_path,
        *dataset
    ]
    subprocess.run(cmd, check=True)


def test_bench_calculate_loq(benchmark, tmp_path, mock_dataset):
    benchmark(bench_calculate_loq, tmp_path, mock_dataset)


def bench_process_nonquant_peptide(calculate_loq, subset):
    result_row = calculate_loq.process_peptide(
        100, #  bootreps
        0.2,  # cv thresh
        None,  # output dir
        subset.iloc[0]["peptide"],
        "n",  # plot_or_not
        2.0,  # std mult
        calculate_loq.DEFAULT_MIN_NOISE_POINTS,
        calculate_loq.DEFAULT_MIN_LINEAR_POINTS,
        calculate_loq.DEFAULT_MIN_SATURATION_POINTS,
        subset,
        "n",  # verbose
        "auto",  # model
    )

    # process_peptide never raises: a failed call comes back as a row with LOD = inf
    # and the exception in 'notes', which would pass the LOD check below. Require a
    # real fit first (this is how the 8-argument call stayed hidden, issue #17).
    note = result_row.iloc[0]["notes"]
    assert note in ("", "below_min_detections"), f"process_peptide failed: {note}"
    assert np.isinf(result_row.iloc[0]["LOD"]), "Got LOD for non-quantitative peptide!!"


def test_bench_process_nonquant_peptide(benchmark, mock_dataset):
    calculate_loq = importlib.import_module("calculate-loq")

    df = calculate_loq.read_input(*mock_dataset)

    # Carefully chosen to have no LoD
    peptide = "YGLNQMADEKESLVVK"

    benchmark(bench_process_nonquant_peptide, calculate_loq, df[df["peptide"] == peptide])