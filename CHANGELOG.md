# Changelog

The baseline for comparison is the frozen 2020/2021 published method in
[`bin/calculate-loq_2021diann.py`](bin/calculate-loq_2021diann.py) (Pino 2020). To
attribute numeric differences to method rather than environment, run both tools on the
same data in their respective environments; see [`doc/BENCHMARK.md`](doc/BENCHMARK.md).
This project is not formally versioned, so entries are grouped by theme rather than by
release tag.

## Unreleased

### Added — detection gate (`--min_detections`)

A new `--min_detections` option (default 3) refuses a finite LOD/LOQ for a peptide that
was actually reported at fewer than that many distinct dilution levels.

- **Why.** Missing cells are backfilled with zero so lmfit has a value at every run. A
  search engine that reports only where a precursor is identified (DIA-NN) leaves the
  low-concentration cells empty, so for a peptide seen at only one or two levels the fit
  is drawn through backfilled zeros and returns a confident LOD pinned to a dilution
  step. That is a fabricated limit, not a measured one.
- **What it counts.** The number of dilution levels with a real, non-null reported value,
  counted in `_normalize_input` *before* the zero-fill (the search engine's own detection
  call). Three levels = two to fit the linear segment plus one to anchor the noise.
- **Effect.** Peptides below the threshold get non-finite LOD/LOQ/ULOQ and the note
  `below_min_detections`; the bootstrap is skipped for them. Peptides at or above the
  threshold are unchanged (bit-identical) — the gate only removes fabricated rows. Dense
  inputs (EncyclopeDIA integrates the window at every run, so every level is reported)
  never trip the gate. Set `--min_detections 0` to disable.

### Fixed — Spectronaut reader; round-trip tests for three readers

`tests/test_readers.py` writes `data/one_protein.csv` as a Skyline export, a DIA-NN
`*.pr_matrix.tsv` and a Spectronaut report, and checks that each reads back to the frame
the EncyclopeDIA file gives. The Skyline and `pr_matrix` readers passed as they were. The
Spectronaut reader had two defects:

- `EG.PrecursorId` (`_SEQUENCE_.charge`) was split at the first dot, so a modification
  written as a mass (`C[+57.0214635]`) gave a broken peptide name. It is now split at the
  last dot. Spectronaut normally writes modifications by name, which has no dot.
- Two helper columns went into the melt as if they were runs. The concentration map
  removed those rows, but `area` came back as `object`, not `float64`. On the sample data
  every figure of merit and note was the same as the EncyclopeDIA run all the same.

### Fixed — pandas `inplace=` deprecation in the input reader

pandas 3.1 deprecates the `inplace` keyword (PDEP-8), and pandas 4 will remove it.
`read_input` now assigns the result of `drop` and `rename` back instead of using
`inplace=True`, in the EncyclopeDIA, Skyline, DIA-NN report and Spectronaut branches.
Output is unchanged; the golden characterization test still passes. The frozen 2021
script is left as it is.

### Changed — LOQ readout (matrix-matched_calcurves#21)

How `calculate_loq` reads the LOQ off the bootstrap CV curve was rewritten. LOD and ULOQ
are untouched.

- **Interpolated crossing instead of grid snapping.** The LOQ is now the concentration
  where the bootstrap CV curve crosses below the threshold, interpolated between the two
  grid points that bracket the crossing, rather than the lowest grid point that happens
  to sit under the threshold. Snapping made the reported value depend on the arbitrary
  grid point count (a median ~5%, up to ~16%, shift between a 100- and 400-point grid);
  interpolation removes that dependence.
- **Readout grid matched to the dilution design.** A new `detect_spacing` helper
  classifies the design as log- or linear-spaced from the regularity of the curve
  points' ratios versus differences. A log design gets a geometric grid (`geomspace`,
  fine resolution down where the LOQ sits) and log-x interpolation; a linear design gets
  the uniform grid (`linspace`) and linear-x interpolation. The old code always used a
  uniform grid.
- **The tool no longer invents an LOQ.** When the CV stays above the threshold across the
  whole range there is no LOQ; the tool reports it as non-finite with the note
  `loq_no_crossing` rather than snapping to the lowest grid point. When the CV is already
  below the threshold at the bottom of the range (quantifiable down to detection), the
  LOQ is reported as the LOD with the note `loq_at_lod`. Ordinary interpolated crossings
  keep an empty note (fit OK).
- **Effect on the sample dataset** (`data/one_protein.csv`, 27 peptides): 24 LOQs move,
  median ~15% lower; the peptides the old uniform grid floor-pinned drop by roughly half
  (e.g. `VVEILQNR` 0.0129 → 0.0058). Two peptides that previously reported a fabricated
  LOQ now correctly report none. The golden characterization snapshot was regenerated to
  re-pin this behaviour.

### Changed — fit failures in `notes` are stable codes (matrix-matched_calcurves#24)

A failed fit used to write the raw exception message to `notes`, so the text changed
with scipy and lmfit versions. Known failures now get a short code: `no_usable_points`,
`fit_underdetermined` (too few points for the model) and `fit_singular` (a
linear-algebra error in the solve). Any other failure is written as
`fit_error: <ExceptionType>: <message>`, truncated to 300 characters. The full message
still goes to stderr. Figures of merit are unchanged; only the `notes` text of failed
peptides changes.

## Method changes from the Pino 2020 baseline

Verified against the frozen script. `doc/BENCHMARK.md` and the frozen script are the
empirical source of truth for exact numeric attribution; intervening changes (multiplier
handling, piecewise edge cases, refactors) are in the git history.

- **LOD noise standard deviation.** The 2020 method used the population standard
  deviation (`np.std`, ddof=0); the current tool uses the sample standard deviation
  (ddof=1). This shifts every LOD slightly.
- **LOQ readout.** See *Unreleased* above. The 2020 method snapped the LOQ to the lowest
  point of a uniform grid whose CV fell under the threshold, with no interpolation, no
  spacing awareness, and no explicit no-crossing outcome.
- **Saturation and ULOQ.** The 2020 method fit only a noise + linear model and reported
  LOD and LOQ. The current `--model auto` additionally tests each peptide for a
  high-signal saturation plateau (trilinear fit selected by AIC) and reports a ULOQ when
  one is found.
- **Deterministic bootstrap.** The 2020 method resampled with `df.sample(replace=True)`
  and no seed, so results depended on global RNG state. The current tool seeds every
  bootstrap replicate from its own `SeedSequence`, so output is reproducible and
  independent of the worker-process count.
- **Input handling.** The current reader auto-detects EncyclopeDIA `.elib.peptides.txt`,
  Skyline exports, DIA-NN `diann_report.tsv`, DIA-NN `.pr_matrix.tsv`, and Spectronaut;
  it fills runs a peptide dropped out of with zero areas (keeping the noise plateau
  dense) and sorts rows canonically by `(curvepoint, area)` so the LOQ no longer depends
  on the order the reader emitted rows.
- **Output schema.** The current `figuresofmerit.csv` adds `n_curvepoints` and a `notes`
  column (fit-failure and LOQ-outcome codes, see above), writes rows incrementally, and
  keeps a row with non-finite figures of merit for a peptide whose fit fails. The 2020
  output had neither column.
