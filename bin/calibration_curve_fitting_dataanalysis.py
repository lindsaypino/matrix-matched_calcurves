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

import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import curve_fit
from scipy import stats
from tqdm import tqdm
from matplotlib.ticker import ScalarFormatter
import argparse


# detect whether the file is Enyclopedia output or Skyline report, then read it in appropriately
## TODO: incorporate an optional multiplier file
def read_input(filename, col_conc_map_file):

    header_line = open(filename, 'r').readline()

    # if numFragments is a column, it's an Encyclopedia file
    if 'numFragments' in header_line:
        sys.stdout.write("Input identified as EncyclopeDIA *.elib.peptides.txt filetype.\n")

        # read in the table
        df = pd.read_table(filename, sep=None, engine="python")

        # (map back to proteins later if necessary)
        #protpep_map = df[['Protein', 'peptide']]

        # make a quantitative dataframe with just the curve points and the peptides
        df = df.drop(['numFragments', 'Protein'], 1)

        # read in the filename-to-concentration map
        col_conc_map = pd.read_csv(col_conc_map_file, sep=',', engine="python")

        # add a mapping to preserve the peptide column
        ## i definitely fucked this up
        '''if 'peptide' not in col_conc_map:
            col_conc_map.loc[len(col_conc_map) + 1] = ['peptide', 'peptide']'''

        # map the filenames to concentrations
        df = df.rename(columns=col_conc_map.set_index('filename')['concentration'])

        # rename the peptide column
        df = df.rename(columns={'Peptide': 'peptide'})

        # melt the dataframe down
        df_melted = pd.melt(df, id_vars=['peptide'])
        df_melted.columns = ['peptide', 'curvepoint', 'area']

        # convert the curve points to numbers so that they sort correctly
        df_melted['curvepoint'] = pd.to_numeric(df_melted['curvepoint'])

    else:
        sys.stdout.write("Input identified as Skyline export filetype. \n")

        # read in the table
        df_melted = pd.read_csv(filename, sep=None, engine="python")

        ##################################################
        ## TODO clean up the dataset-specific stuff below
        # remove replicates 4 and 5 from the curve data
        df_melted = df_melted[df_melted['replicate'] < 4]
        # for now, remove all peptides for which there isn't a ghaemma protein coPI_lineares per cell value
        df_melted = df_melted.loc[-df_melted['ghaemma_protein_cpc'].isnull()]
        ##################################################

        ## TODO: REQUIRE COLUMN NAMING SCHEME
        df_melted.rename(columns={'curvepoint_cpc':'curvepoint'}, inplace=True)
        df_melted.rename(columns={'tic_normalized_area': 'area'}, inplace=True)
        df_melted.rename(columns={'precursor': 'peptide'}, inplace=True)

        # convert the curve points to numbers so that they sort correctly
        df_melted['curvepoint'] = pd.to_numeric(df_melted['curvepoint'])

    return df_melted

# define each of the linear segments and do a PI_linearecewise fit
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

    #if not min(conc_list):
    #    print subsetdf['peptide'].drop_duplicates()

    bottom_point = min(conc_list)

    # find the y intercept using this slope (b = y-mx) and the top point
    intercept = mean_y[str(bottom_point)] - (0 * bottom_point)
    return intercept

# fit just a first degree polynomial
def fit_one_segment(x, slope, intercept):
    segment = slope * x + intercept
    return segment

# plot a line given a slope and intercept
def add_line_to_plot(slope, intercept, setstyle='-', setcolor='k'):
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, linestyle=setstyle, color=setcolor)



# set the project directory to the current directory
project_dir = os.getcwd()

#'''
# parse args?
parser = argparse.ArgumentParser(
    description="A prediction interval-based model for calibration curve data.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('curve_data', type=str,
                    help='a matrix containing peptides and their quantitative values across each curve point')
parser.add_argument('filename_concentration_map', type=str,
                    help='a comma-delimited file containing maps between filenames and the concentration point '
                         'they represent')
parser.add_argument('--plot', default=True, type=bool,
                    help='create individual calibration curve plots for each peptide')

parser.parse_args()
#'''

# read in the command line arguments
raw_file = sys.argv[1]
col_conc_map_file = sys.argv[2]

'''
project_dir = "G:/My Drive/00_UW_GS/proj/CalibratedQuant_DIA-MS/data/"
raw_file =  os.path.join(project_dir,
                         "../results/20170901_yeastcalcurve_skylinequant_cleaned.csv")
col_conc_map_file = os.path.join(project_dir,
                                 "filename2samplegroup_map.csv")'''

# read in the data
quant_df_melted = read_input(raw_file, col_conc_map_file)


##
## ITERATE OVER PEPTIDES
##

# initialize empty data frame to store results
peptidecrossover = pd.DataFrame(columns=['peptide', 'LOD', 'LOQ'])
peptide_nan = 0

# and awwaayyyyy we go~
for peptide in tqdm(quant_df_melted['peptide'].unique()):
#for peptide in ['FTPTSNFGQSIIK_2']:

    # subset the dataframe for that peptide
    subset = quant_df_melted.loc[(quant_df_melted['peptide'] == peptide)]

    # if the peptide is nan, skip it and move on to the next peptide
    if subset.empty:
        peptide_nan += 1
        continue

    # sort the dataframe with x values in strictly ascending order.
    subset = subset.sort_values(by='curvepoint', ascending=True)

    # create the x and y arrays
    x = np.array(subset['curvepoint'], dtype=float)
    y = np.array(subset['area'], dtype=float)

    # back to string
    ## TODO REPLACE WITH .iloc
    subset['curvepoint'] = subset['curvepoint'].astype(str)

    # use non-linear least squares to fit the two functions (noise and linear) to data.
    # pw vector = (m_noise, b_noise, m_linear, b_linear)
    pw0 = (0, initialize_noiseintercept(subset),
           initialize_slope(subset), initialize_linearintercept(subset))

    try:  # I think I can take this exception handling out? idk, check if there's 'no fit' I guess
        pw, cov = curve_fit(two_lines, x, y, pw0)
        slope_noise = pw[0]
        intercept_noise = pw[1]
        slope_linear = pw[2]
        intercept_linear = pw[3]
    except:
        sys.stderr.write("Peptide %s could not be curve_fit." % peptide)
        slope_noise = np.nan
        intercept_noise = np.nan
        slope_linear = np.nan
        intercept_linear = np.nan

    if slope_linear < 0:
        crossover = float('Inf')
    else:
        # find the crossover point, defined by the intersection of the noise and linear regime
        crossover = (intercept_linear - intercept_noise) / (slope_noise - slope_linear)

    # if the crossover point is greater than the top point of the curve or is a negative number,
    # then replace it with a value (Inf or NaN) indicating such
    if crossover > max(x):
        crossover = float('Inf')
    elif crossover < 0:
        crossover = float('nan')

    ##
    ## PREDICTION BAND
    ##

    ## NOISE SEGMENT PREDICTION BAND
    subset_noise = subset.loc[(subset['curvepoint'].astype(float) < crossover)]
    # if there is no noise portion, just set intersect_PI_linear to negative infinite
    if not subset_noise.size:
        intersect_PI_linear = float('-Inf')
        x2_noise = np.nan
        y2_noise = np.nan
        PI_noise = np.nan
    else:
        x_noise = np.array(subset_noise['curvepoint'], dtype=float)
        y_noise = np.array(subset_noise['area'], dtype=float)

        # calculate model statistics
        n = y_noise.size  # number of observations in the linear range
        m = pw.size / 2  # number of parameters (half of which are linear range)
        DF = n - m  # degrees of freedom
        t = scipy.stats.t.ppf(0.95, n - m)  # used for CI and PI_linear bands

        pred_y_noise = fit_one_segment(x_noise, slope_noise, intercept_noise)

        # estimate the error in the data/model
        resid_noise = y_noise - pred_y_noise
        s_err_noise = np.sqrt(np.sum(resid_noise ** 2) / (DF))  # standard deviation of the error

        # draw the prediction intervals using a range of x and y values
        x2_noise = np.linspace(np.min(x), crossover, 100)
        y2_noise = np.linspace(np.min(pred_y_noise), np.max(pred_y_noise), 100)
        PI_noise = max(t * s_err_noise * np.sqrt(
            1 + 1 / n + (x2_noise - np.mean(x_noise)) ** 2 / np.sum((x_noise - np.mean(x_noise)) ** 2)))

    ## LINEAR SEGMENT PREDICTION BAND
    subset_linear = subset.loc[(subset['curvepoint'].astype(float) >= crossover)]
    # if there is no linear portion, just set intersect_PI_linear to infinite
    if not subset_linear.size:
        intersect_PI_linear = float('Inf')
        x2_linear = np.nan
        y2_linear = np.nan
        PI_linear = np.nan
    else:
        x_linear = np.array(subset_linear['curvepoint'], dtype=float)
        y_linear = np.array(subset_linear['area'], dtype=float)

        # calculate model statistics
        n = y_linear.size  # number of observations in the linear range
        m = pw.size / 2  # number of parameters (half of which are linear range)
        DF = n - m  # degrees of freedom
        t = scipy.stats.t.ppf(0.95, n - m)  # used for CI and PI_linear bands

        pred_y_linear = fit_one_segment(x_linear, slope_linear, intercept_linear)

        # estimate the error in the data/model
        resid_linear = y_linear - pred_y_linear
        s_err_linear = np.sqrt(np.sum(resid_linear ** 2) / (DF))  # standard deviation of the error

        # draw the prediction intervals using a range of x and y values
        x2_linear = np.linspace(crossover, np.max(x), 100)
        y2_linear = np.linspace(np.min(pred_y_linear), np.max(pred_y_linear), 100)
        PI_linear = max(t * s_err_linear * np.sqrt(
            1 + 1 / n + (x2_linear - np.mean(x_linear)) ** 2 / np.sum((x_linear - np.mean(x_linear)) ** 2)))

    ##
    ## LOQ BY PREDICTION BAND & PIECEWISE LINEAR MODEL INTERSECTIONS
    ##

    # find the intersection of the lower PI_linear and the upper PI_noise
    intersect_PI_linear = (intercept_linear - intercept_noise - PI_linear - PI_noise) / (slope_noise - slope_linear)

    # if the crossover point is greater than the top point of the curve or is a negative number,
    # then replace it with a value (Inf or NaN) indicating such
    if intersect_PI_linear > max(x):
        intersect_PI_linear = float('Inf')
    elif intersect_PI_linear < 0:
        intersect_PI_linear = float('nan')

    ##
    ## PRETTY PLOTS
    ##

    # make a plot of the curve points and the fit, in both linear and log space
    plt.figure(figsize=(10, 5))

    # left hand plot: linear scale x axis
    plt.subplot(1, 2, 1)
    plt.plot(x, y, 'o')
    add_line_to_plot(slope_noise, intercept_noise, '-', 'g')
    if slope_linear > 0:
        add_line_to_plot(slope_linear, intercept_linear, '-', 'g')
    plt.axvline(x=crossover, color='m', label=('LOD= %.3e' % crossover))
    plt.axvline(x=intersect_PI_linear, color='c', label=('LOQ= %.3e' % intersect_PI_linear))
    plt.title(peptide)
    plt.xlabel("curve point")
    plt.ylabel("area")

    # force axis ticks to be scientific notation so the plot is prettier
    ####### TEST #######
    # plt.ticklabel_format(style='sci', axis='x')
    # returns  AttributeError: This method only works with the ScalarFormatter.
    ####################
    #if not np.isnan(PI_noise):
    # Add the prediction intervals on the left hand plot
    add_line_to_plot(slope_noise, intercept_noise + PI_noise, '--', setcolor='0.5')
    add_line_to_plot(slope_noise, intercept_noise - PI_noise, '--', setcolor='0.5')
    #if not np.isnan(PI_linear):
    add_line_to_plot(slope_linear, intercept_linear + PI_linear, '--', setcolor='0.5')
    add_line_to_plot(slope_linear, intercept_linear - PI_linear, '--', setcolor='0.5')

    # right hand plot: log scale x axis
    plt.subplot(1, 2, 2)
    plt.plot(x, y, 'o')
    add_line_to_plot(slope_noise, intercept_noise, '-', 'g')
    if slope_linear > 0:
        add_line_to_plot(slope_linear, intercept_linear, '-', 'g')
    plt.axvline(x=crossover, color='m', label=('LOD= %.3e' % crossover))
    plt.axvline(x=intersect_PI_linear, color='c', label=('LOQ= %.3e' % intersect_PI_linear))

    #if not np.isnan(PI_noise):
    # Add the prediction intervals on the left hand plot
    add_line_to_plot(slope_noise, intercept_noise + PI_noise, '--', setcolor='0.5')
    add_line_to_plot(slope_noise, intercept_noise - PI_noise, '--', setcolor='0.5')
    #if not np.isnan(PI_linear):
    add_line_to_plot(slope_linear, intercept_linear + PI_linear, '--', setcolor='0.5')
    add_line_to_plot(slope_linear, intercept_linear - PI_linear, '--', setcolor='0.5')

    plt.xscale('log')
    plt.title(peptide)
    plt.xlabel("curve point (log-space)")
    plt.ylabel("area")

    # force axis ticks to be scientific notation so the plot is prettier
    ## TEST
    # plt.ticklabel_format(style='sci', axis='x')
    # returns  AttributeError: This method only works with the ScalarFormatter.
    ##

    # add legend with LOD and LOQ values
    legend = plt.legend(loc=9, bbox_to_anchor=(0, -0.21, 1., .102), ncol=2)

    # save the figure
    plt.savefig(('C:/Users/Lindsay/Desktop/scratch/' + peptide + '.png'),
                bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.close()

    # make a dataframe row with the peptide and it's crossover point
    new_row = [peptide, crossover, intersect_PI_linear]
    new_df_row = pd.DataFrame([new_row], columns=['peptide', 'LOD', 'LOQ'])
    peptidecrossover = peptidecrossover.append(new_df_row)

peptidecrossover.to_csv(path_or_buf='C:/Users/Lindsay/Desktop/scratch/figuresofmerit.csv', index=False)
print "fyi: there were ", peptide_nan, "NaN peptides in the data"
