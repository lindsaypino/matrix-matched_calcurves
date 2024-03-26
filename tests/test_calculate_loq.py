"""
`benchmarks.py` -- run benchmarks on `calculate_loq.py`
"""

import sys

sys.path.append("bin")

import importlib
import numpy as np
import pytest
import subprocess


@pytest.fixture
def mock_dataset():
    return ("data/one_protein.csv", "data/filename2samplegroup_map.csv")


def bench_calculate_loq(tmp_path, dataset):
    cmd = [
        "python", "bin/calculate-loq.py",
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
        False,  # plot_or_not
        2.0,  # std mult
        subset,
        False
    )

    assert np.isinf(result_row.iloc[0]["LOD"]), "Got LOD for non-quantitative peptide!!"


def test_bench_process_nonquant_peptide(benchmark, mock_dataset):
    calculate_loq = importlib.import_module("calculate-loq")

    df = calculate_loq.read_input(*mock_dataset)

    # Carefully chosen to have no LoD
    peptide = "RGEGFMVVTATGDNTFVGR"

    benchmark(bench_process_nonquant_peptide, calculate_loq, df[df["peptide"] == peptide])