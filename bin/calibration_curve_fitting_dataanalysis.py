# ### Background.
# 
# I have developed and tested a piecewise linear model for sparse calibration curve data.
# The model consists of two regimes: the noise regime and the linear regime.
# Because label-free DIA calibration curve data is so sparse (70% zeroes) and considering
# the canonical shape of a calibration curve, the noise regime is initialized with a
# slope of zero and an intercept of zero. The linear regime is initialized with a slope
# and intercept learned from the two highest concentration points. The model produces the
# intersection between the noise regime and the linear regime of a calibration curve,
# and creates plots to visualize the model fits and intersection value.
#

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# detect whether the file is Enyclopedia output or Skyline report, then read it in appropriately
def read_input(filename, col_conc_map_file):

    header_line = open(filename, 'r').readline()

    # if numFragments is a column, it's an Encyclopedia file
    if 'numFragments' in header_line:
        sys.stdout.write("Input identified as EncyclopeDIA *.elib.peptides.txt filetype.\n")

        # read in the table
        df = pd.read_table(filename, sep=None, engine="python")

        # (map back to proteins later if necessary)
        #protpep_map = df[['Protein', 'Peptide']]

        # make a quantitative dataframe with just the curve points and the peptides
        df = df.drop(['numFragments', 'Protein'], 1)

        # read in the filename-to-concentration map
        col_conc_map = pd.read_csv(col_conc_map_file, sep=',', engine="python")

        # add a mapping to preserve the peptide column
        if 'Peptide' not in col_conc_map:
            col_conc_map.loc[len(col_conc_map) + 1] = ['Peptide', 'Peptide']

        # map the filenames to concentrations
        df = df.rename(columns=col_conc_map.set_index('filename')['concentration'])

        # melt the dataframe down
        df_melted = pd.melt(df, id_vars=['Peptide'])
        df_melted.columns = ['peptide', 'curvepoint', 'area']

        # convert the curve points to numbers so that they sort correctly
        df_melted['curvepoint'] = pd.to_numeric(df_melted['curvepoint'])

    else:
        sys.stdout.write("Input identified as Skyline export filetype. Not supported (yet!)\n")
        return -1 # exit out for now since I haven't built this for Skyline files yet
        #df = pd.read_csv(filename, engine='python')

    return df_melted

# define each of the linear segments and do a piecewise fit
def two_lines(x, a, b, c, d):

    # slope of the noise (a) should be zero
    a = 0
    noise = a * x + b
    linear = c * x + d

    return np.maximum(noise, linear)

# find the slope between the top two points
def initialize_slope(subsetdf):

    # find the mean response area for each curve point
    mean_y = subsetdf.groupby('curvepoint')['area'].mean()

    # find the top point and second-top point of the curve
    conc_list = list(pd.to_numeric(subsetdf['curvepoint'].drop_duplicates()))
    top_point = max(conc_list)
    conc_list.remove(top_point)
    second_top = max(conc_list)

    # using the means, calculate a slope (y1-y2/x1-x2)
    slope = (mean_y[str(second_top)] - mean_y[str(top_point)]) / (second_top - top_point)

    return slope

# find the y-intercept of the linear regime based on the top two points of the curve
def initialize_linearintercept(subsetdf):

    # find the mean response area for each curve point
    mean_y = subsetdf.groupby('curvepoint')['area'].mean()

    # find the top point and second-top point of the curve
    conc_list = list(pd.to_numeric(subsetdf['curvepoint'].drop_duplicates()))
    top_point = max(conc_list)
    conc_list.remove(top_point)
    second_top = max(conc_list)

    # using the means, calculate a slope (y1-y2/x1-x2)
    slope = (mean_y[str(second_top)] - mean_y[str(top_point)]) / (second_top - top_point)

    # find the y intercept using this slope (b = y-mx) and the top point
    intercept = mean_y[str(top_point)] - (slope * top_point)

    return intercept

# find the y-intercept of the noise regime based on the bottom point of the curve
def initialize_noiseintercept(subsetdf):

    # find the mean response area for each curve point
    mean_y = subsetdf.groupby('curvepoint')['area'].mean()

    # find the top point and second-top point of the curve
    conc_list = list(pd.to_numeric(subsetdf['curvepoint'].drop_duplicates()))
    bottom_point = min(conc_list)

    # find the y intercept using this slope (b = y-mx) and the top point
    intercept = mean_y[str(bottom_point)] - (0 * bottom_point)

    return intercept


# set the project directory to the current directory
project_dir = os.getcwd()

# parse args?
parser = argparse.ArgumentParser(
    description="A piecewise linear model for calibration curve data.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('curve_data', type=str,
                    help='a matrix containing peptides and their quantitative values across each curve point')
parser.add_argument('filename_concentration_map', type=str,
                    help='a comma-delimited file containing maps between filenames and the concentration point '
                         'they represent')
parser.add_argument('--plot', default=True, type=bool,
                    help='create individual calibration curve plots for each peptide')

parser.parse_args()

# read in the command line arguments
#raw_file = sys.argv[1]
#col_conc_map_file = sys.argv[2]

'''
project_dir = "G:/My Drive/00_UW_GS/proj/CalibratedQuant_DIA-MS/csf/"
raw_file = os.path.join(project_dir,
                        "data/20180112_CSFCurves_quant.elib.peptides.txt")
col_conc_map_file = os.path.join(project_dir,
                                 "data/20180112_CSFCurves_map.csv")
'''

# read in the data
quant_df_melted = read_input(raw_file, col_conc_map_file)

# sort the dataframe with x values in strictly ascending order.
quant_df_melted = quant_df_melted.sort_values(by='curvepoint', ascending=True)

### loop over all the peptides

# initialize empty data frame to store results
peptidecrossover = pd.DataFrame(columns=['peptide', 'LOD'])

# initialize a counter for sanity
counter = 1

# and awwaayyyyy we go~
for peptide in quant_df_melted['peptide'].unique():

    # print a small sanity-check counter to know whether the code's running or not
    if counter % len(quant_df_melted['peptide'].unique()) == 0:
        pctprogress = ((counter * 1.00) / (len(quant_df_melted['peptide'].unique()) * 1.00)) * 100.00
        sys.stderr.write('progress: %.2f \n' % round(pctprogress, 2))
    counter += 1

    # subset the dataframe for that peptide
    subset = quant_df_melted.loc[(quant_df_melted['peptide'] == peptide)]

    # create the x and y arrays
    x = np.array(subset['curvepoint'], dtype=float)
    y = np.array(subset['area'], dtype=float)

    # back to string
    ## TODO REPLACE WITH .iloc
    subset['curvepoint'] = subset['curvepoint'].astype(str)

    # use non-linear least squares to fit the two functions (noise and linear) to data.
    pw0 = (0, initialize_noiseintercept(subset),
           initialize_slope(subset), initialize_linearintercept(subset))

    try:
        pw, cov = curve_fit(two_lines, x, y, pw0)

        # find the crossover point, defined by the intersection of the noise and linear regime
        crossover = (pw[3] - pw[1]) / (pw[0] - pw[2])

        # if the crossover point is greater than the top point of the curve or is a negative number,
        # then replace it with a value (Inf or NaN) indicating such
        if crossover > max(x):
            crossover = float('Inf')
        elif crossover < 0:
            crossover = float('nan')

        # make a plot of the curve points and the fit, in both linear and log space
        plt.figure(figsize=(10, 5))

        # left side: plot the linear scale
        plt.subplot(1, 2, 1)
        plt.plot(x, y, 'o', x, two_lines(x, *pw), '-')
        plt.axvline(x=crossover, color='r')
        plt.title(peptide)
        plt.figtext(0.1, 0.87, crossover, wrap=True, fontsize=12)
        plt.xlabel("curve point")
        plt.ylabel("area")
        # right side: plot exact same thing but with log scaled x axis
        plt.subplot(1, 2, 2)
        plt.plot(x, y, 'o', x, two_lines(x, *pw), '-')
        plt.axvline(x=crossover, color='r')
        plt.xscale('log')
        plt.title(peptide)
        plt.figtext(0.6, 0.87, crossover, wrap=True, fontsize=12)
        plt.xlabel("curve point (log-space)")
        plt.ylabel("area")
        # clean up the borders with '*.tight_layout()' and save the plot under the peptide's name
        plt.tight_layout()
        plt.savefig(('C:/Users/Lindsay/Desktop/scratch/' + peptide + '.png'))
        plt.close()

    except:
        sys.stderr.write("Peptide %s could not be fit." % peptide)
        crossover = 'no fit'

    # make a dataframe row with the peptide and it's crossover point
    new_row = [peptide, crossover]
    new_df_row = pd.DataFrame([new_row], columns=['peptide', 'LOD'])
    peptidecrossover = peptidecrossover.append(new_df_row)

peptidecrossover.to_csv(path_or_buf='C:/Users/Lindsay/Desktop/scratch/figuresofmerit.csv', index=False)


