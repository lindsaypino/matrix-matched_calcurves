"""
`benchmarks.py` -- run benchmarks on `calculate_loq.py`
"""


def bench_calculate_loq(tmp_path):
    raise RuntimeError("TODO")


def test_bench_calculate_loq(benchmark, tmp_path):
    benchmark(bench_calculate_loq, tmp_path)