# Benchmarking against the 2021 (Pino 2020) method

`bin/calculate-loq_2021diann.py` is a **frozen** copy of the calibration-curve
code as it stood in 2021, matching the LOD/LOQ computation described in the Pino
2020 paper (with a minor addition to read a DIA-NN input file). It exists so
that differences between the current tool and the original method can be
explained: run both on the same data and compare the figures of merit.

## Do not modernize this script

The script's value is faithful reproduction of the *old* numbers, so its logic
must not change. Critically, it also must not be run against modern libraries,
because they change its results **silently**:

- **pandas >= 3.0 (copy-on-write):** calls like
  `df['area'].fillna(0, inplace=True)` become no-ops. The intended NaN -> 0
  replacement never happens, and the leftover NaNs propagate into
  `std_noise`, `LOD`, and `LOQ`. The script still exits 0 and writes output, so
  the corruption is invisible. (Observed: 27 peptides from `data/one_protein.csv`
  produced 31 NaN cells on pandas 3.0 vs 0 on the pinned stack.)
- **pandas >= 2.0:** removed APIs used elsewhere in the 2021 code path.
- **matplotlib >= 3.6:** dropped the `seaborn-whitegrid` style name the script
  requests.

Patching these would mean editing the frozen logic and defeats the purpose. Fix
the *environment* instead.

## How to run the benchmark

Use the pinned 2021-era stack in `requirements-2021.txt`. The easiest way is the
dedicated Docker image:

```bash
docker build -f Dockerfile.benchmark -t calcurves-benchmark:2021 .

docker run --rm --user $(id -u):$(id -g) -v `pwd`:`pwd` -w `pwd` \
    calcurves-benchmark:2021 data/one_protein.csv data/filename2samplegroup_map.csv
```

Or, without Docker, install the pins into an isolated Python 3.9 environment:

```bash
python3.9 -m venv .venv-benchmark
.venv-benchmark/bin/pip install -r requirements-2021.txt
.venv-benchmark/bin/python bin/calculate-loq_2021diann.py \
    data/one_protein.csv data/filename2samplegroup_map.csv
```

Compare the resulting `figuresofmerit.csv` against the output of the current
tool (`bin/calculate-loq.py`) to attribute LOD/LOQ differences to method changes
(e.g. sample vs population noise standard deviation, the corrected LOQ formula)
rather than environment artifacts.
