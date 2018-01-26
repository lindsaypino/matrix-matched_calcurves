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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def init_argparse():

    parser = argparse.ArgumentParser(
        description="blablablabla~~",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-quantitative_matrix', type=os.path.abspath)

    return parser

# parse arguments and check for appropriate inputs
def validate_args(args):

    if args.count_peptides_in_lib != None and not os.path.exists(args.count_peptides_in_lib):
        return 0
    # check arguments here

    return 1

# read in file
def read_input(filename):
    sys.stderr.write("Reading in file %s...\n" % filename)
    df = pd.read_table(filename, sep=None, engine="python")
    sys.stderr.write("File read-in complete.\n")
    return df

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

    # using the means, calculate a slope (y1-y2/x1-x2)
    # dont need to do this for the noise, since slope is assumed 0
    #slope = (mean_y[str(second_bottom)] - mean_y[str(bottom_point)]) / (second_bottom - bottom_point)

    # find the y intercept using this slope (b = y-mx) and the top point
    intercept = mean_y[str(bottom_point)] - (0 * bottom_point)

    return intercept


# set the project directory
project_dir = "G:/My Drive/00_UW_GS/proj/CalibratedQuant_DIA-MS/csf/"
raw_file = os.path.join(project_dir,
                        "data/20180112_CSFCurves_quant.elib.peptides.txt")

# read in the data
raw_df = read_input(raw_file)

# make a quantitative dataframe with just the curve points and the peptides (map back to proteins later if necessary)
protpep_map = raw_df[['Protein', 'Peptide']]
quant_df = raw_df.drop(['numFragments', 'Protein'], 1)

# map the file names to concentrations
## TODO this should be a parameter option or required in the input matrix, opposed to being hard coded in here
newcols = ['Peptide',
           '100','100','100','1000','1000','1000','1000','1000','1000','1000','1000','700','700','700',
           '500','500','500','300','300','300','70','70','70','50','50','50','30','30','30',
           '10','10','10','0','0','0']
quant_df.columns = newcols

# melt the dataframe down
quant_df_melted = pd.melt(quant_df, id_vars=['Peptide'])
quant_df_melted.columns = ['peptide', 'curvepoint', 'area']

# convert the curve points to numbers so that they sort correctly
quant_df_melted['curvepoint'] = pd.to_numeric(quant_df_melted['curvepoint'])

# sort the dataframe with x values in strictly ascending order.
quant_df_melted = quant_df_melted.sort_values(by='curvepoint', ascending=True)

### loop over all the peptides

# initialize empty data frame to store results
column_names = ['peptide', 'crossover']
peptidecrossover = pd.DataFrame(columns=column_names)

# initialize a counter for sanity
counter = 1

# and awwaayyyyy we go~
for peptide in quant_df_melted['peptide'].unique():

    # print a small sanity-check counter to know whether the code's running or not
    if counter % 1000 == 0:
        pctprogress = ((counter * 1.00) / (len(quant_df_melted['peptide'].unique()) * 1.00)) * 100.00
        sys.stderr.write('progress: %.2f \n' % round(pctprogress, 2))
    counter += 1

    # subset the dataframe for that peptide
    subset = quant_df_melted.loc[(quant_df_melted['peptide'] == peptide)]

    # create the x and y arrays
    x = np.array(subset['curvepoint'], dtype=float)
    y = np.array(subset['area'], dtype=float)

    # back to string
    subset['curvepoint'] = subset['curvepoint'].astype(str)

    # use non-linear least squares to fit the two functions (noise and linear) to data.
    pw0 = (0, initialize_noiseintercept(subset),
           initialize_slope(subset), initialize_linearintercept(subset))
    pw, cov = curve_fit(two_lines, x, y, pw0)

    # find the crossover point, defined by the intersection of the noise and linear regime
    crossover = (pw[3] - pw[1]) / (pw[0] - pw[2])

    # if the crossover point is greater than the top point of the curve or is a negative number,
    # then replace it with a value (Inf or NaN) indicating such
    if crossover > max(x):
        crossover = float('Inf')
    elif crossover < 0:
        crossover = float('nan')

    # make a dataframe row with the peptide and it's crossover point
    new_row = [peptide, crossover]
    new_df_row = pd.DataFrame([new_row], columns=column_names)
    peptidecrossover = peptidecrossover.append(new_df_row)

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
    plt.savefig(('C:/Users/lpino/Desktop/scratch/' + peptide + '.png'))
    plt.close()

peptidecrossover.to_csv(path_or_buf='C:/Users/lpino/Desktop/scratch/peptidecrossovers.csv', index=False)


