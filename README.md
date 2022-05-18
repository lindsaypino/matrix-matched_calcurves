## Usage

`calculate-loq  <curve_data> <filename_concentration_map>`


## Description

`calculate-loq` fits a piecewise prediction interval model to calibration
curve data. A horizontal "noise segment" is fit to model the background
signal from a blank or any concentration point below the LOD;
a positive-sloped "signal segment" is fit to model the linear range
from points with signal above the noise. The data is bootstrapped to
determine a mean and standard deviation for quantities above the LOD,
which are then used to calculate a coefficient of variation and
therefore an LOQ.


## Input Requirements

- `curve_data` – either the Encyclopedia `*.elib.peptides.txt` file or a
Skyline `\*.csv` custom export with peptides as rows, concentration
points as columns, and areas as values

- `filename_concentration_map` - a csv containing each filename as a row
with its corresponding concentration point in a second column.


## Output

The program writes files to the folder curvefits-output by default.
The following files will be created:

- `figures_of_merit.csv` – a file containing the peptides in one column
and their calculated LOQ in another.

- `*.png` – (optional) plots of each peptide calibration curve with the
fitted piecewise linear regression.

## Options

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

- `--multiplier_file`, type=str,
'use a single-point multiplier associated with the curve data peptides'

- `--output_path`, default=os.getcwd(), type=str,
'specify an output path for figures of merit and plots'

- `--plot`, default='y', type=str,
'yes/no (y/n) to create individual calibration curve plots for each
peptide'

- `--verbose`, default='n', type=str,
'output a detailed summary of the bootstrapping step'


## Example

```python
python bin\calculate-loq.py data\one_protein.csv data\filename2concentration.csv --multiplier_file data\multiplier_file.csv
```

## For developers

### Setting up an environment

Use `conda` from anaconda/miniconda.

```shell
conda create -n matrix-matched-calcurves python=3.8
conda activate matrix-matched-calcurves
conda install --file requirements.txt
```