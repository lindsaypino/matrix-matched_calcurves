# TODO

Open questions about `calculate-loq` that aren't tracked as GitHub issues yet.

## Decide whether the `notes` column stays in figuresofmerit.csv

Added by `2205392` ("Never drop a peptide: record fit failures as noted rows"), which
changed the output schema from

```
peptide,LOD,LOQ,ULOQ,slope_linear,intercept_linear,intercept_noise,stndev_noise
```

to that plus `n_curvepoints` and `notes`.

The question is whether `notes` earns its place in the output, or whether the
never-drop-a-peptide guarantee is better expressed some other way (a sentinel in an
existing column, a separate sidecar file, a log line). Worth deciding before the
schema is baked into published supplemental tables.

Things to weigh:

- **What actually goes in it.** On the two EncyclopeDIA datasets re-run on 2026-08-12
  (Exploris 62,760 peptides; IL15 539) the column was empty for every single row. If
  it is usually empty, is a column the right shape for it?
- **It is a free-text column in a numeric table.** Anything consuming the CSV has to
  cope with a string field that may contain commas/quotes. Consider a small
  closed vocabulary of codes if it stays.
- **Downstream readers.** `2026_calcurve_paper` reads these CSVs with
  `pandas.read_csv` by column name, so extra columns are harmless there — but the
  manuscript ships these files as supplemental tables, where a column that is empty
  in every row invites reviewer questions.
- **`n_curvepoints` is a separate call.** It arrived in the same schema change but is
  a populated numeric field, so it stands on its own merits.

Raised while regenerating the paper's figures of merit; see
`2026_calcurve_paper/docs/fom_provenance.md` for the run that surfaced it.

## Decide whether to change how the LOQ is read off the CV curve

Two changes to `calculate_loq` would alter reported LOQ values. Nothing else on this
page does, so these are the only items that need settling before figures of merit are
regenerated for publication. They are coupled and should be decided together.

**1. Grid spacing.** The bootstrap CV is read on `np.linspace(LOD, max_x, 100)` -- a
uniform grid -- while the dilution series is log-spaced. The two disagree at exactly
the end where the LOQ lives. On `data/one_protein.csv`, for 16 of 26 peptides with a
finite LOD the *first* grid point above the LOD already steps over at least one
measured level; `VVEILQNR` steps over 4 of the 12 levels above its LOD.

**2. The crossing rule.** The LOQ is the lowest grid *point* whose CV is under
threshold, so when the CV never exceeds the threshold the rule silently returns the
lowest value the grid can express -- a number that looks like a measured LOQ but is
`LOD + one grid step`. There is no way for the output to say "the CV never got that
bad in the measured range".

Both were scored against a simulated ground truth, so the true LOQ was known:

- A log grid roughly halves the bias of genuine crossings versus the uniform grid
  (+56% vs +93%) and cuts the dependence on the arbitrary point count from a median
  15.9% to 2.6%. Reading only at measured levels is *worse* than either (+111%) --
  8-11 points above the LOD is too coarse to locate the crossing.
- In a scenario where the truth has no crossing at all, the current rule reports an
  LOQ in **100%** of experiments on every grid, and the fabricated value is itself a
  grid artifact: it halves from 0.0154 to 0.0078 when the grid goes from 100 to 400
  points. An interpolated crossing with an explicit no-crossing outcome fabricates in
  0-1%.
- The cost of the explicit outcome is silence: where a crossing really does exist it
  declines to report for 22% of experiments on a log grid, 34% on a uniform one.

Things to weigh:

- **This changes published numbers.** Both the values and, for some peptides, whether
  an LOQ is reported at all.
- **An explicit outcome is a schema question**, and overlaps the `notes` decision
  above -- "no crossing in range" needs somewhere to live.
- **Even the best combination is not accurate.** Log spacing plus interpolation plus
  an explicit outcome still leaves the LOQ biased about +50% high with an
  interquartile range near 100% of its own value across replicate experiments. These
  changes remove artifacts; they do not make the LOQ a precise number. What limits it
  is the design.

Full numbers, caveats and reproduction recipes are in
`2026_calcurve_paper/docs/loq_grid_and_resampling_note.md`; the figure is
`SUPP_loq_readout` from `figures/supp_loq_readout.py` in that repo.

## Settled: do not change the bootstrap resampling scheme

Recorded so this is not reopened. `_bootstrap_once` uses case resampling -- rows drawn
across the whole curve -- which lets a replicate lose every measurement of a
concentration level (on a 14x3 curve, half of all replicates lose at least one level,
and the blank vanishes from about 5%). That looks like a defect, and stratified, wild
and Bayesian resampling were all implemented and evaluated as replacements.

Scored against a simulated ground truth, where the real sampling variability is known,
case resampling is the **best calibrated of the four**: bootstrap CV divided by true CV
was 0.96-0.97 across two error models, against 0.80 stratified, 0.86 wild and 0.85
Bayesian. The alternatives all *understate* uncertainty by 15-20%, which would report
the assay as more precise than it is. The likely mechanism is small-sample bootstrap
bias -- with 3 replicates per level, within-stratum resampling cannot express the true
variance.

A `--bootstrap {case,stratified}` flag was implemented during this evaluation and
deliberately reverted; the tool should keep case resampling only. One practical note if
it is ever revisited: stratified resampling cannot work on a curve with a single
measurement per level, because drawing within a one-row level returns that row, every
replicate reproduces the curve exactly, the CV is identically zero and the LOQ
collapses onto the LOD.

Figure: `SUPP_bootstrap_calibration` from `figures/supp_bootstrap_calibration.py` in
`2026_calcurve_paper`.

## Small defects found but not filed as issues

None of these change any reported figure of merit.

- **Unrecognized input filetype raises `UnboundLocalError`.** `read_input` ends with
  `return _normalize_input(df_long, col_conc_map)` and no `else` branch, so a header
  matching none of the readers fails on an undefined name instead of saying the
  filetype was not recognized. Pre-existing (it did the same with `df_melted`), and
  now easy to fix cleanly since every branch funnels through one tail.
- **`build_plots` resamples without a seed.** The CV scatter in the bottom subplot
  calls `df.sample(n=len(df), replace=True)` with no `random_state`, so plotted CVs
  differ between runs on identical input. Plots only -- `figuresofmerit.csv` is
  unaffected.
- **The fit weight cap of 1000 is an undocumented magic number.** Weights are
  `min(1/sqrt(x), 1000)`; the `1/sqrt(x)` part is an ordinary proportional-error
  model, but the cap is what keeps a 0-concentration point finite and it gives blank
  replicates roughly a million times the squared leverage of a top-of-curve point.
  Changing it 1000 -> 100 moves the typical LOD by 0.49% (max 2.65%), so results do
  not hinge on it -- but it deserves a comment saying it is a floor-guard rather than
  a derived value. See `2026_calcurve_paper/docs/fit_weighting_note.md`.
- **`tests/test_calculate_loq.py::bench_process_nonquant_peptide` is stale.** It calls
  `process_peptide` with 8 positional arguments; the signature has needed 12 since the
  min-points parameters were added, so it would `TypeError`. It is masked because
  `pytest-benchmark` is declared in `tests/conda-env.txt` but not installed in the
  local venv, so both benchmark tests error at setup instead of running. That is the
  two errors on every `pytest` run.
- **No round-trip test for the reader.** Re-expressing `data/one_protein.csv` as a
  ragged long-format `diann_report.tsv` and checking it reproduces the golden snapshot
  is a sharper regression test than the synthetic cases in `tests/test_issues.py`, since
  the wide-format reading of real data is the ground truth. Verified by hand when #15
  was fixed (all 27 peptides matched to within 1e-9) but never committed as a test.

## Downstream copies of the bootstrap still seed one global RNG

Found 2026-09-03 in a separate session on a different dataset. `calculate-loq.py` is
not affected; this is recorded here because the tool is the reference the copies are
checked against, and because the same defect survives in the frozen 2021 script in
this repo.

**Symptom.** In a protein-level derivative of this tool
(`scripts/calculate-loq-protein-level.py`, in a downstream repository), the LOQ and
`avg_cv` are not reproducible across input sets. A 2-protein subset run to generate
figures reproduced LOD, `slope_linear`, `intercept_linear` and `r2_linear` from the
full 6,759-protein run exactly, but not the LOQ:

| Protein | LOQ, 2-protein run | LOQ, full 6,759 run |
|---------|-------------------:|--------------------:|
| FOXO1   | 0.1238             | 0.1335              |
| PAX3    | 0.1285             | 0.1563              |

**Cause.** That script sets one global `np.random.seed(8888)` at import and the
bootstrap draws with `df.sample()` off the global RNG. The stream a given protein sees
therefore depends on how many proteins were processed before it, so the answer changes
with the set and its order. Everything that does not touch the bootstrap (LOD, the
linear fit) is unaffected, which is exactly the split observed.

`bin/calculate-loq.py` does not have this problem: `_bootstrap_once` seeds each
replicate from its own `SeedSequence` (see the note near the imports and the
`default_rng(SeedSequence(seed))` call), and a 6-precursor subset reproduced the full
run exactly once run coverage was matched.

**Same pattern in this repo.** `bin/calculate-loq_2021diann.py` has the identical
construction: `np.random.seed(8888)` / `random.seed(8888)` at module level and
`df.sample(n=len(df), replace=True)` with no `random_state` in `bootstrap_once`. It is
a frozen copy kept for the v1-vs-v2 benchmark, so it should probably not be changed,
but any comparison against it should note that its LOQs are order- and subset-dependent
and are only reproducible for a full run over the same input in the same order.

**To do.**

- Fix the protein-level script the same way as `_bootstrap_once`: derive one
  `SeedSequence` per (peptide or protein, replicate) and pass a `Generator` or
  `random_state` into `df.sample`. Re-run the 2-protein subset and confirm the LOQ and
  `avg_cv` match the full run to the printed precision.
- Add a regression test here that runs `calculate-loq.py` on a subset of
  `data/one_protein.csv` and on the full file, and asserts the shared peptides match.
  This is the property the downstream copy broke, and nothing in `tests/` pins it.
- Decide whether the README paragraph on the 2021 script should state the caveat above.
