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


# detect whether the file is Encyclopedia output or Skyline report, then read it in appropriately
def read_input(filename, col_conc_map_file):

    header_line = open(filename, 'r').readline()

    # if numFragments is a column, it's an Encyclopedia file
    if 'numFragments' in header_line:
        sys.stdout.write("Input identified as EncyclopeDIA *.elib.peptides.txt filetype.\n")

        df = pd.read_table(filename, sep=None, engine="python")  # read in the table
        df = df.drop(['numFragments', 'Protein'], 1)  # make a quantitative df with just curve points and peptides
        col_conc_map = pd.read_csv(col_conc_map_file, sep=',', engine="python")
        df = df.rename(columns=col_conc_map.set_index('filename')['concentration'])  # map filenames to concentrations
        df = df.rename(columns={'Peptide': 'peptide'})  # rename the peptide column
        df_melted = pd.melt(df, id_vars=['peptide'])  # melt the dataframe down
        df_melted.columns = ['peptide', 'curvepoint', 'area']

    else:
        sys.stdout.write("Input identified as Skyline export filetype. \n")

        df_melted = pd.read_csv(filename, sep=None, engine="python")  # read in the csv
        # TODO: REQUIRE COLUMN NAMING SCHEME (SampleGroup, Total Area Fragment, Peptide Sequence)
        df_melted.rename(columns={'SampleGroup': 'curvepoint'}, inplace=True)
        df_melted.rename(columns={'Total Area Fragment': 'area'}, inplace=True)
        df_melted.rename(columns={'Peptide Sequence': 'peptide'}, inplace=True)

        df_melted['area'].fillna(0, inplace=True)  # replace NA with 0

    # convert the curve points to numbers so that they sort correctly
    df_melted['curvepoint'] = pd.to_numeric(df_melted['curvepoint'])

    return df_melted

# associates a multiplier value to the curvepoint a la single-point calibration
def associate_multiplier(df, multiplier_file):
    mutliplier_df = pd.read_csv(multiplier_file, sep=None, engine="python")  # read in the multiplier file

    # merge the multiplier with the data frame
    merged_df = pd.merge(df, mutliplier_df, on="peptide", how="inner")
    merged_df['curvepoint_multiplied'] = merged_df['curvepoint'] * merged_df['multiplier']
    multiplied_df = merged_df[['peptide', 'curvepoint_multiplied', 'area']]
    multiplied_df.columns = ['peptide', 'curvepoint', 'area']

    return multiplied_df

# define each of the linear segments and do a piecewise fit
def two_lines(x, a, b, c, d):
    a = 0  # slope of the noise (a) should be zero
    noise = a * x + b
    linear = c * x + d
    return np.maximum(noise, linear)

# fit just a first degree polynomial
def fit_one_segment(x, m, b):
    segment = m * x + b
    return segment

# establish initial model fitting parameters based on the data
def initialize_params(subsetdf):

    mean_y = subsetdf.groupby('curvepoint')['area'].mean()  # find the mean response area for each curve point

    # find the top point, second-top point, and bottom points of the curve data
    conc_list = list(pd.to_numeric(subsetdf['curvepoint'].drop_duplicates()))
    top_point = max(conc_list)
    conc_list.remove(top_point)
    second_top = max(conc_list)
    bottom_point = min(conc_list)

    # assume that slope of the noise will be zero
    noise_slope = 0
    # using the means, calculate a slope (y1-y2/x1-x2)
    linear_slope = (mean_y[str(second_top)] - mean_y[str(top_point)]) / (second_top - top_point)
    # find the y1 intercept using noise_slope=0 and the bottom point
    noise_intercept = mean_y[str(bottom_point)] - (noise_slope * bottom_point)
    # find the y2 intercept using linear slope (b = y-mx) and the top point
    linear_intercept = mean_y[str(top_point)] - (linear_slope * top_point)

    return noise_slope, noise_intercept, linear_slope, linear_intercept

# determine the maximum prediction band (unweighted)
def max_prediction_interval(df, crossover, slope, intercept, num_params):
    if not df.size:  # if there's no data for this segment, don't bother with a prediction interval
        pred_int = np.nan
    else:
        x = np.array(df['curvepoint'], dtype=float)
        y = np.array(df['area'], dtype=float)

        # calculate model statistics
        n_obs = y.size  # number of observations in the linear range
        deg_freedom = n_obs - num_params  # degrees of freedom
        t = scipy.stats.t.ppf(0.95, n_obs - num_params)  # used for CI and PI_linear bands

        pred_y = fit_one_segment(x, slope, intercept)

        # estimate the error in the data/model
        resid = y - pred_y
        s_err = np.sqrt(np.sum(resid ** 2) / deg_freedom)  # standard deviation of the error

        # draw the prediction intervals using a range of x and y values
        x2 = np.linspace(crossover, np.max(x), 100)
        pred_int = max(t * s_err * np.sqrt(
            1 + 1 / n_obs + (x2 - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2)))
    return pred_int

# find the crossover point of the piecewise fit, defined by the intersection of the noise and linear regime
def calculate_LOD(b_noise, b_linear, m_noise, m_linear, x):
    if m_linear < 0:
        LOD = float('Inf')
    else:
        LOD = (b_linear - b_noise) / (m_noise - m_linear)

    # consider some special edge cases
    if LOD > max(x):  # if the crossover point is greater than the top point of the curve or is a negative number,
        LOD = float('Inf')  # then replace it with a value (Inf or NaN) indicating such
    elif LOD < 0:  # if the LOQ is for some reason below zero
        LOD= float('nan')  # then replace it with NaN
    return LOD

# find the intersection of the lower PI_linear and the upper PI_noise
def calculate_LOQ(b_noise, b_linear, m_noise, m_linear, x):
    LOQ = (intercept_linear - intercept_noise - PI_linear - PI_noise) / (slope_noise - slope_linear)

    # consider some special edge cases
    if LOQ > max(x):  # if the crossover point is greater than the top point of the curve or is a negative number,
        LOQ = float('Inf')  # then replace it with a value (Inf or NaN) indicating such
    elif LOQ < 0:  # if the LOQ is for some reason below zero
        LOQ = float('nan')  # then replace it with NaN
    return LOQ

# plot a line given a slope and intercept
def add_line_to_plot(slope, intercept, scale, setstyle='-', setcolor='k'):
    axes = plt.gca()
    xlims = np.array(axes.get_xlim())
    x_vals = np.arange(xlims[0], xlims[1], ((xlims[1]-xlims[0])/100))
    y_vals = intercept + slope * x_vals
    if scale == 'semilogx':
        plt.semilogx(x_vals, y_vals, linestyle=setstyle, color=setcolor)
    else:
        plt.plot(x_vals, y_vals, linestyle=setstyle, color=setcolor)

# create plots of the curve points, the segment fits, and the LOD/LOQ values
def build_plots(x, y, intercept_noise, slope_noise, intercept_linear, slope_linear, crossover, intersect_PI_linear):
    # MAGIC MATPLOTLIB BELOW. NO TOUCH.
    plt.figure(figsize=(10, 5))

    ###
    ### left hand plot: linear scale x axis
    plt.subplot(1, 2, 1)
    plt.plot(x, y, 'o')  # scatterplot of the data
    add_line_to_plot(slope_noise, intercept_noise, 'linear', '-', 'g')  # add noise segment line
    if slope_linear > 0:  # add linear segment line
        add_line_to_plot(slope_linear, intercept_linear, 'linear', '-', 'g')
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
    # Add the prediction intervals on the left hand plot
    add_line_to_plot(slope_noise, intercept_noise + PI_noise, 'linear', '--', setcolor='0.5')
    add_line_to_plot(slope_noise, intercept_noise - PI_noise, 'linear', '--', setcolor='0.5')
    add_line_to_plot(slope_linear, intercept_linear + PI_linear, 'linear', '--', setcolor='0.5')
    add_line_to_plot(slope_linear, intercept_linear - PI_linear, 'linear', '--', setcolor='0.5')

    ###
    ### right hand plot: log scale x axis (semilog x)
    plt.subplot(1, 2, 2)
    plt.semilogx(x, y, 'o')
    add_line_to_plot(slope_noise, intercept_noise, 'semilogx', '-', 'g')
    if slope_linear > 0:
        add_line_to_plot(slope_linear, intercept_linear, 'semilogx', '-', 'g')
    plt.axvline(x=crossover, color='m', label=('LOD= %.3e' % crossover))
    plt.axvline(x=intersect_PI_linear, color='c', label=('LOQ= %.3e' % intersect_PI_linear))

    # Add the prediction intervals on the right hand plot
    add_line_to_plot(slope_noise, intercept_noise + PI_noise, 'semilogx', '--', setcolor='0.5')
    add_line_to_plot(slope_noise, intercept_noise - PI_noise, 'semilogx', '--', setcolor='0.5')
    add_line_to_plot(slope_linear, intercept_linear + PI_linear, 'semilogx', '--', setcolor='0.5')
    add_line_to_plot(slope_linear, intercept_linear - PI_linear, 'semilogx', '--', setcolor='0.5')

    plt.title(peptide)
    plt.xlabel("curve point (log10)")
    plt.ylabel("area")

    # force axis ticks to be scientific notation so the plot is prettier
    ## TEST
    # plt.ticklabel_format(style='sci', axis='x')
    # returns  AttributeError: This method only works with the ScalarFormatter.
    ##

    # add legend with LOD and LOQ values
    legend = plt.legend(loc=9, bbox_to_anchor=(0, -0.21, 1., .102), ncol=2)

    # save the figure
    plt.savefig(os.path.join(output_dir, peptide + '.png'),
                bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":

    # usage statement and input descriptions
    parser = argparse.ArgumentParser(
        description="A prediction interval-based model for fitting calibration curve data. Takes calibration curve \
                    measurements as input, and returns the Limit of Detection (LOD) and Limit of Quantitation (LOQ) for \
                    each peptide measured in the calibration curve.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('curve_data', type=str,
                        help='a matrix containing peptides and their quantitative values across each curve point (currently\
                                supporting Encyclopedia *.elib.peptides.txt quant reports and Skyline export reports)')
    parser.add_argument('filename_concentration_map', type=str,
                        help='a comma-delimited file containing maps between filenames and the concentration point \
                                they represent (two columns named "filename" and "concentration")')
    parser.add_argument('--multiplier_file', type=str,
                        help='use a single-point multiplier associated with the curve data peptides')
    parser.add_argument('--output_path', default=os.getcwd(), type=str,
                        help='specify an output path for figures of merit and plots (default=current directory)')
    parser.add_argument('--plot', default=True, type=bool,
                        help='create individual calibration curve plots for each peptide (default=True)')

    # parse arguments from command line
    args = parser.parse_args()
    raw_file = args.curve_data
    col_conc_map_file = args.filename_concentration_map
    multiplier_file = args.multiplier_file
    output_dir = args.output_path
    plot_or_not = args.plot

    # read in the data
    quant_df_melted = read_input(raw_file, col_conc_map_file)

    # associate multiplier with the curvepoint ratio (if there is a multiplier provided)
    if multiplier_file:
        quant_df_melted = associate_multiplier(quant_df_melted, multiplier_file)

    # initialize empty data frame to store results
    peptidecrossover = pd.DataFrame(columns=['peptide', 'LOD', 'LOQ'])
    peptide_nan = 0

    # and awwaayyyyy we go~
    for peptide in tqdm(quant_df_melted['peptide'].unique()):

        subset = quant_df_melted.loc[(quant_df_melted['peptide'] == peptide)]  # subset the dataframe for that peptide

        if subset.empty:  # if the peptide is nan, skip it and move on to the next peptide
            peptide_nan += 1
            continue

        # sort the dataframe with x values in strictly ascending order
        subset = subset.sort_values(by='curvepoint', ascending=True)

        # create the x and y arrays
        x = np.array(subset['curvepoint'], dtype=float)
        y = np.array(subset['area'], dtype=float)

        # TODO REPLACE WITH .iloc
        subset['curvepoint'] = subset['curvepoint'].astype(str)  # back to string

        # use non-linear least squares to fit the two functions (noise and linear) to data.
        try:
            model_parameters, cov = curve_fit(two_lines, x, y, initialize_params(subset))
            slope_noise = model_parameters[0]
            intercept_noise = model_parameters[1]
            slope_linear = model_parameters[2]
            intercept_linear = model_parameters[3]
        except:  # catch for when a peptide can't be fit for whatever reason
            sys.stderr.write("Peptide %s could not be curve_fit." % peptide)
            slope_noise = np.nan
            intercept_noise = np.nan
            slope_linear = np.nan
            intercept_linear = np.nan

        # find the crossover point of the piecewise fit, defined by the intersection of the noise and linear regime
        LOD = calculate_LOD(intercept_noise, intercept_linear, slope_noise, slope_linear, x)

        # calculate prediction intervals
        PI_noise = max_prediction_interval(subset.loc[(subset['curvepoint'].astype(float) < LOD)],
                                                 LOD, slope_noise, intercept_noise, (model_parameters.size / 2))
        PI_linear = max_prediction_interval(subset.loc[(subset['curvepoint'].astype(float) >= LOD)],
                                                  LOD, slope_linear, intercept_linear, (model_parameters.size / 2))

        # find the intersection of the lower PI_linear and the upper PI_noise
        LOQ = calculate_LOQ(intercept_noise, intercept_linear, slope_noise, slope_linear, x)

        if plot_or_not == True:
            # make a plot of the curve points and the fit, in both linear and log space
            build_plots(x, y, intercept_noise, slope_noise, intercept_linear, slope_linear, LOD, LOQ)

        # make a dataframe row with the peptide and its crossover point
        new_row = [peptide, LOD, LOQ]
        new_df_row = pd.DataFrame([new_row], columns=['peptide', 'LOD', 'LOQ'])
        peptidecrossover = peptidecrossover.append(new_df_row)

    peptidecrossover.to_csv(path_or_buf=os.path.join(output_dir,'figuresofmerit.csv'),
                            index=False)

    print "fyi: there were ", peptide_nan, "NaN peptides in the data"
