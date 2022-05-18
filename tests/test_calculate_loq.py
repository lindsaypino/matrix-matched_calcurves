"""
`benchmarks.py` -- run benchmarks on `calculate_loq.py`
"""

import subprocess


def bench_calculate_loq(tmp_path):
    cmd = [
        "python", "bin/calculate-loq.py",
        "--output_path", tmp_path,
        "data/one_protein.csv", "data/filename2samplegroup_map.csv"
    ]
    subprocess.run(cmd, check=True)


def test_bench_calculate_loq(benchmark, tmp_path):
    benchmark(bench_calculate_loq, tmp_path)