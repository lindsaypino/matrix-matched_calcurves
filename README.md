**USAGE.**

`calculate-loq  <curve_data> <filename_concentration_map>`


**DESCRIPTION.**

`calculate-loq` fits a piecewise prediction interval model to calibration
curve data. A horizontal "noise segment" is fit to model the background
signal from a blank or any concentration point below the LOD;
a positive-sloped "signal segment" is fit to model the linear range
from points with signal above the noise. The data is bootstrapped to
determine a mean and standard deviation for quantities above the LOD,
which are then used to calculate a coefficient of variation and
therefore an LOQ.

By default (`--model auto`) the fit also tests, per peptide, whether the top
of the curve bends over into a high-signal saturation plateau. When it does,
a trilinear (noise + linear + saturation) model is selected by AIC and an
upper limit of quantitation (ULOQ) is reported in addition to the LOD and
LOQ; curves that stay linear at the top are fit with the bilinear
(noise + linear) model and have no ULOQ.


**INPUT REQUIREMENTS.**

- `curve_data` – a quantitative report from any of the supported search
engines. The filetype is auto-detected from the header:
  - EncyclopeDIA `*.elib.peptides.txt` (peptides as rows, runs as columns)
  - Skyline `*.csv` custom export (must include `Peptide Sequence`,
  `File Name`, and `Total Area Fragment`)
  - DIA-NN `diann_report.tsv` (recommended DIA-NN input)
  - DIA-NN `*.pr_matrix.tsv` (supported, but the tool will warn and
  recommend `diann_report.tsv` instead)
  - Spectronaut export (with `PEP.StrippedSequence`)

  Every format is normalized identically once it has been read, so the same
  experiment gives the same answer whichever report it came from:

  - **Dense.** Each peptide spans every measured run. Long-format reports (DIA-NN
  `diann_report.tsv`, and Skyline exports that omit undetected peptides) carry a
  row only where the peptide was identified; the runs it dropped out of are filled
  in as zero areas, which is what the wide-format matrices already contain. This
  keeps the noise plateau the LOD is fit from.
  - **Canonically ordered.** Rows are sorted by `(curvepoint, area)`. The
  bootstrap resamples by row position, so without a canonical order the LOQ
  depended on whatever order the reader happened to emit — by up to 240% on the
  sample dataset. Sorting on `curvepoint` alone is not enough, because replicates
  tie on it.
  - Only runs listed in the concentration map are kept, and a run absent from the
  report entirely is never invented as an all-zero column.

- `filename_concentration_map` - a csv with two columns named `filename`
and `concentration`, one row per run, mapping each filename to the
concentration point it represents. Rows with a blank/unannotated
`concentration` are skipped with a warning that lists the affected
filenames.


**OUTPUT.**

The program writes files to the current working directory by default (use
`--output_path` to change this). The following files will be created:

- `figuresofmerit.csv` – one row per peptide with columns `peptide`, `LOD`,
`LOQ`, `ULOQ`, `slope_linear`, `intercept_linear`, `intercept_noise`,
`stndev_noise`, `n_curvepoints`, and `notes`. `ULOQ` is non-finite for peptides
fit with the bilinear model (no saturation); `n_curvepoints` is the number of
distinct concentration levels the peptide was fit over, which separates a
genuine "no saturation" result from a curve that had too few levels for the
`auto` model to look for one. Rows are written incrementally as each peptide
finishes, so partial results survive an interrupted run. A peptide whose fit
fails is never dropped: it still gets a row with non-finite figures of merit
and the error message recorded in the `notes` column.

- `*.png` – (optional) plots of each peptide calibration curve with the
fitted piecewise linear regression.

**FIGURES OF MERIT.**

The three limits below are given at the default settings (`--std_mult 2`,
`--cv_thresh 0.2`, `--bootreps 100`, `--min_noise_points 2`,
`--min_linear_points 1`, `--min_saturation_points 2`, `--model auto`). In the
formulas, `m_lin` and `b_lin` are the slope and the intercept of the linear
segment, `b_noise` is the noise intercept, `c_high` is the saturation ceiling,
and `s_noise` / `s_sat` are the sample standard deviations (ddof=1) of the areas
in the noise segment and in the saturation plateau.

| | LOD | LLOQ (the `LOQ` column) | ULOQ |
|---|---|---|---|
| **Definition at the defaults** | The concentration at which the fitted linear segment gets to the noise intercept plus 2 noise standard deviations. | The lowest concentration above the LOD at which the bootstrap CV is less than 0.20 (20%). | The concentration at which the linear segment comes to 2 saturation standard deviations below the saturation ceiling. |
| **Formula** | `LOD = (b_noise + 2*s_noise - b_lin) / m_lin` | `LLOQ = min{ x : CV(x) < 0.20 }`, where `CV(x) = std/mean` of the 100 bootstrap fits evaluated at `x` | `ULOQ = (c_high - 2*s_sat - b_lin) / m_lin` |
| **Where it is measured** | The intersection of the noise and linear segments | A 100-point grid from the LOD up to `min(ULOQ, max curvepoint)` | The saturation onset, `(c_high - b_lin) / m_lin`, backed off by the plateau noise |
| **Data support necessary** | At least 2 distinct curve points below the LOD, and at least 1 at or above it | A non-empty bootstrap summary, and at least one grid point below the CV threshold | A trilinear fit selected by AIC, and at least 2 distinct curve points in the plateau |
| **Reported as non-finite when** | `m_lin <= 0` (noise only), or a support rule above fails | No grid point meets the CV threshold; or the LLOQ is at the top of the grid; or the LLOQ is <= 0; or the LOD is non-finite | The fit is bilinear (`c_high` is infinite); too few plateau points; `ULOQ <= 0`; or `ULOQ <= LOD` |
| **Code** | `calculate_lod` | `calculate_loq` | `calculate_uloq` |

Two notes about the defaults:

- With `--model auto`, most curves get a bilinear fit, so `ULOQ` is non-finite
for them. The `n_curvepoints` column separates "no saturation" from "too few
levels to test for saturation".
- The `piecewise` model uses different LOD edge cases: it compares the LOD
against the second-lowest curve point and the maximum curve point, instead of
counting the distinct points on each side. The table above describes the
default `auto` path.

**OPTIONS.**

- `--std_mult`, default=2, type=float,
'specify a multiplier of the standard deviation of the noise for
determining limit of detection (LOD)'

- `--cv_thresh`, default=0.2, type=float,
'specify a coefficient of variation threshold for determining limit of
quantitation (LOQ) (Note: this should be a decimal, not a percentage,
e.g. 20% CV threshold should be input as 0.2)'

- `--bootreps`, default=100, type=int,
'specify a number of times to bootstrap the data (Note: this must be an
integer, e.g. to resample the data 100 times, the parameter value
should be input as 100'

- `--min_noise_points`, default=2, type=int,
'the minimum number of curve points required below the LOD for a fit to
be considered valid'

- `--min_linear_points`, default=1, type=int,
'the minimum number of curve points required above the LOD for a fit to
be considered valid'

- `--min_saturation_points`, default=2, type=int,
'minimum curve points in the high-signal saturation plateau required for the
`auto` model to adopt a trilinear (noise + linear + saturation) fit and report
a ULOQ; raise to be more conservative about calling saturation'

- `--model`, default=`auto`, choices=[`auto`, `piecewise`, `bilinear`, `trilinear`],
'which curve model to fit: `auto` (default) picks bilinear (noise + linear)
vs trilinear (noise + linear + saturation) per peptide by AIC, adding a
saturation ceiling / ULOQ only when the top of the curve bends over;
`piecewise` is the original legacy-init bilinear fit; `bilinear` forces the
improved-init noise+linear fit; `trilinear` forces the noise+linear+saturation
fit'

- `--multiplier_file`, type=str,
'use a single-point multiplier associated with the curve data peptides'

- `--output_path`, default=os.getcwd(), type=str,
'specify an output path for figures of merit and plots'

- `--plot`, default='y', type=str,
'yes/no (y/n) to create individual calibration curve plots for each
peptide'

- `--verbose`, default='n', type=str,
'output a detailed summary of the bootstrapping step'

- `--n_threads`, default=`cpu_count - 2`, type=int,
'number of worker processes for parallel peptide processing. Set to `-1`
to use all CPUs, or `1` to run serially in-process (see PARALLELISM below)'


**PARALLELISM.**

Peptides are fit in parallel across worker processes. Results are
deterministic and independent of the number of workers: each bootstrap
replicate is seeded from its own `SeedSequence`, so a run with `--n_threads 1`
produces output identical to a parallel run.

If a worker process is terminated abruptly (e.g. out-of-memory, or a native
crash in numpy/scipy/lmfit/matplotlib), the tool prints a warning and
automatically falls back to serial processing so the run still completes and
the underlying error is shown. To debug such a crash directly, rerun with
`--n_threads 1`, which skips the process pool and reports the failing peptide
with a full traceback.


**EXAMPLE.**

```python
python bin\calculate-loq.py data\one_protein.csv data\filename2samplegroup_map.csv --multiplier_file data\multiplier_file.csv
```

**DOCKER.**

To build with Docker: `docker build -t matrix-matched_calcurves:latest .`

To run:

```bash
docker run --rm --user $(id -u):$(id -g) -v `pwd`:`pwd` -w `pwd` matrix-matched_calcurves:latest <curve_data> <filename_concentraion_map>
```

**BENCHMARK.**

`bin/calculate-loq_2021diann.py` is a frozen copy of the 2021 (Pino 2020)
method, kept for comparing old vs new LOD/LOQ calculations. It must be run in
its pinned 2021-era environment (`requirements-2021.txt` / `Dockerfile.benchmark`)
— newer pandas/matplotlib silently change its results. See
[doc/BENCHMARK.md](doc/BENCHMARK.md).
