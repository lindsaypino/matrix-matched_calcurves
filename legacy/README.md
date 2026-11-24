# legacy/

Scripts that are no longer maintained. They are kept for the record and are not tested,
not run in CI, and not kept in step with `bin/calculate-loq.py`. Do not use them for new
analyses.

## `loq_by_cv.py`

**Retired 2026-11-24, moved here from `bin/`.**

**What it does.** Reads an EncyclopeDIA `*.elib.peptides.txt` or a Skyline export, takes
the coefficient of variation (sample SD / mean) of the replicate areas at each curve
point, and reports as the LOQ the lowest curve point above zero with a CV of 20% or less.
It writes `loqsbycv.csv` and `peptidecvs.csv`, plus one plot per peptide unless
`--plot n`. It has no LOD, no bootstrap, no detection gate and no `notes` column, and it
does not require the CV to stay under 20% above the reported point.

**Why it was retired.**

- The method work ended in 2018-2019. After that it had only compatibility edits
  (pandas and seaborn in 2024, a pandas and performance pass in 2026). Nothing in the
  repository called it, and there were no tests for it.
- `bin/calculate-loq.py` supersedes it: it reads the LOQ off a bootstrapped CV curve
  with an interpolated crossing (matrix-matched_calcurves#21) and shares one reader for
  every input format.
- Bringing it up to date would be a separate effort: the shared readers, tests, and a
  decision on how a per-level CV threshold should relate to the bootstrapped readout.

**State when retired.** On `data/one_protein.csv` with `--plot n` it ran without error
(pandas 3.1.1) and gave an LOQ for 18 of 27 peptides. The plotting path and the Skyline
path were not checked. The Skyline path still uses `rename(..., inplace=True)`, which
pandas 3.1 deprecates.

## Not in this folder

`bin/calculate-loq_2021diann.py` is also frozen, but it stays in `bin/` because
`doc/BENCHMARK.md` runs it as the 2021 baseline.
