

import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy
from tqdm import tqdm
import argparse
from lmfit import minimize, Minimizer, Parameters, report_fit, Model, conf_interval, printfuncs
from scipy.optimize import curve_fit


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

        # require columns for File Name, Total Area Fragment, Peptide Sequence
        # TODO: option for Total Area Ratio?
        if 'Total Area Fragment' not in header_line \
                or 'Peptide Sequence' not in header_line \
                or 'File Name' not in header_line:
            sys.exit("Skyline export must include Peptide Sequence, File Name, and Total Area Fragment.\n")

        else:
            df_melted = pd.read_csv(filename, sep=None, engine="python")  # read in the csv

            # map filenames to concentrations
            col_conc_map = pd.read_csv(col_conc_map_file, sep=',', engine="python")
            df_melted.rename(columns={'File Name': 'filename'}, inplace=True)
            df_melted = pd.merge(df_melted, col_conc_map, on='filename', how='outer')

            # clean up column names to match downstream convention
            df_melted.rename(columns={'Total Area Fragment': 'area'}, inplace=True)
            df_melted.rename(columns={'Peptide Sequence': 'peptide'}, inplace=True)
            df_melted.rename(columns={'concentration': 'curvepoint'}, inplace=True)

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


# determine the maximum prediction band (unweighted)
def max_prediction_interval(df, LOD, params):

    predint_noise = calculate_PI(df.loc[(df['curvepoint'].astype(float) < LOD)], params[0], params[1])
    predint_linear = calculate_PI(df.loc[(df['curvepoint'].astype(float) > LOD)], params[2], params[3])

    predint_results = [predint_noise, predint_linear]

    return predint_results


def calculate_max_PI(subset_df, slope, intercept):
    if not subset_df.size:  # if there's no data for this segment, don't bother with a prediction interval
        pred_int = np.nan

    else:
        '''
        ## first attempt
        # print list(subset_df['curvepoint'].drop_duplicates())
        x = np.array(subset_df['curvepoint'], dtype=float)
        y = np.array(subset_df['area'], dtype=float)

        n_obs = float(y.size)  # number of observations in the segment
        ss_total = np.sum(y ** 2) - ((1.0 / n_obs) * np.sum(y) ** 2)
        ss_residual = slope * (np.sum(x * y) - ((1.0 / n_obs) * np.sum(x) * np.sum(y)))
        # sys.stderr.write("ss_total = %f | ss_residual = %f\n" % (ss_total, ss_residual))
        ss_error = ss_total - ss_residual  # TODO why is my ss_error negative for "all linear" peptides?
        standard_error = np.sqrt(ss_error / (n_obs - 2.0))
        alpha = 0.68  # one standard deviation = 68; two standard deviations = 95
        t = scipy.stats.t.ppf(((1.0-(alpha/2)) / 2.0), (n_obs - 2.0))

        all_pred_int = []
        for i in set(x):
            this_pred_int = t * standard_error * np.sqrt(
                1.0 + (1.0 / n_obs) + ((i - np.mean(x)) ** 2) / (np.sum(x ** 2) - (1.0 / n_obs) * np.mean(x) ** 2))
            all_pred_int.append(abs(this_pred_int))
            # sys.stderr.write("x value %f = %f prediction interval\n" % (i,this_pred_int))

        # print all_pred_int
        pred_int = max(all_pred_int)
        '''

        '''# new attempt
        x = np.array(subset_df['curvepoint'], dtype=float)
        y = np.array(subset_df['area'], dtype=float)
        n_obs = float(y.size)  # number of observations in the segment
        mean_x = np.mean(x)

        # fit statistics, should be global for this whole section?
        ss_total = np.sum(y ** 2) - ((1.0 / n_obs) * np.sum(y) ** 2)
        ss_residual = slope * (np.sum(x * y) - ((1.0 / n_obs) * np.sum(x) * np.sum(y)))
        # sys.stderr.write("ss_total = %f | ss_residual = %f\n" % (ss_total, ss_residual))
        ss_error = ss_total - ss_residual  # TODO is this ss_error negative for "all linear" peptides?
        standard_error = np.sqrt(ss_error / (n_obs - 2.0))
        # sys.stderr.write("ss_error = %f; standard_error = %f\n" % (ss_error, standard_error))
        alpha = 0.68  # one standard deviation = 68; two standard deviations = 95
        t = scipy.stats.t.ppf(((1.0 - (alpha / 2)) / 2.0), (n_obs - 2.0))

        # generate some new data
        x_i = np.linspace(min(x), max(x), num=10)
        y_i = (slope*x_i) + intercept

        all_pred_int = []
        # iterate over each point (x) to calculate the uncertainty for that point
        for i in subset_df['curvepoint'].drop_duplicates():
            thispoint_df = subset_df.loc[subset_df['curvepoint'] == i]
            x_obs = np.array(thispoint_df['curvepoint'], dtype=float); print x_obs
            y_obs = np.array(thispoint_df['area'], dtype=float)
            n_obs = float(y_obs.size)  # number of y observations for this x value

            # predicted y_0 for this point i == x_0
            x_0 = float(i)  # x value that we're given
            y_0 = (slope*x_0)+intercept  # mean value for y we would predict given that x_0 value


            # prediction interval calculation
            # these np.mean -- should they be the y that the model gives, for this x?
            # mean of x will just be x since it's replicates of the same point??
            # and i == x
            this_pred_int = t*standard_error*np.sqrt(1.0+(1.0/n_obs)+((float(x_0)-mean_x**2)/np.sum((x_0 - x)**2)))
            pred_int_bits = 1.0+(1.0/n_obs)
            pred_int_num = (float(x_0)-mean_x**2)
            pred_int_denom = np.sum((x_0 - x)**2)
            all_pred_int.append(abs(this_pred_int))
            #sys.stderr.write("x value %f: %f pred_int_bits, %f pred_int_num, %f pred_int_denom, %f prediction interval\n" %
            #                 (float(i), pred_int_bits, pred_int_num, pred_int_denom, this_pred_int))'''

        # new attempt
        x = np.array(subset_df['curvepoint'], dtype=float)
        y = np.array(subset_df['area'], dtype=float)
        n_obs = float(y.size)  # number of observations in the segment
        mean_x = np.mean(x)

        # fit statistics, should be global for this whole section?
        ss_total = np.sum(y ** 2) - ((1.0 / n_obs) * np.sum(y) ** 2)
        ss_residual = slope * (np.sum(x * y) - ((1.0 / n_obs) * np.sum(x) * np.sum(y)))
        # sys.stderr.write("ss_total = %f | ss_residual = %f\n" % (ss_total, ss_residual))
        ss_error = ss_total - ss_residual  # TODO is this ss_error negative for "all linear" peptides?
        standard_error = np.sqrt(ss_error / (n_obs - 2.0))
        # sys.stderr.write("ss_error = %f; standard_error = %f\n" % (ss_error, standard_error))
        alpha = 0.68  # one standard deviation = 68; two standard deviations = 95
        t = scipy.stats.t.ppf((1.0 - (alpha / 2)), (n_obs - 2.0))

        # generate some new data
        x_i = np.linspace(min(x), max(x), num=10, dtype=float)
        # make sure to estimate the PI of the intercept for linear segment where x_i might not include 0
        if float(0) not in x_i:
            x_i = np.append(x_i, float(0))
        y_i = (slope * x_i) + intercept

        all_pred_int = [];
        pred_int_sqrt_list = []

        # iterate over each resampled/binned point (x_i) to calculate the uncertainty for that point
        for i in x_i:
            # prediction interval calculation
            # these np.mean -- should they be the y that the model gives, for this x?
            # mean of x will just be x since it's replicates of the same point??
            # and i == x
            this_pred_int = t * standard_error * np.sqrt(
                1.0 + (1.0 / n_obs) + (((float(i) - mean_x) ** 2) / (np.sum(x) - ((1 / n_obs) * (np.sum(x) ** 2)))))
            pred_int_sqrt = np.sqrt(
                1.0 + (1.0 / n_obs) + (((float(i) - mean_x) ** 2) / (np.sum(x) - ((1 / n_obs) * (np.sum(x) ** 2)))))
            pred_int_sqrt_list.append(pred_int_sqrt)
            all_pred_int.append(abs(this_pred_int))
            # sys.stderr.write("x value %f: %f pred_int_bits, %f pred_int_num, %f pred_int_denom, %f prediction interval\n" %
            #                 (float(i), pred_int_bits, pred_int_num, pred_int_denom, this_pred_int))

        # print a bunch of info about the PI calculation
        print "x_i:", x_i
        print "t:", t
        print "standard error:", standard_error
        print "pred_int_sqrt:", pred_int_sqrt_list
        print "prediction intervals:", all_pred_int
        print "predicted y_i:", y_i

        # take the maximum of all the_pred_int
        pred_int = max(all_pred_int)

    return pred_int


def calculate_PI(subset_df, slope, intercept, bins=10):

    if not subset_df.size:  # if there's no data for this segment, don't bother with a prediction interval
        predint_df = pd.DataFrame({'x_i': [np.nan], 'y_i': [np.nan], 'predint': [np.nan], 'Mstatistic': [np.nan]})

    else:

        x = np.array(subset_df['curvepoint'], dtype=float)
        y = np.array(subset_df['area'], dtype=float)
        n_obs = float(y.size)  # number of observations in the segment
        mean_x = np.mean(x)

        # fit statistics, should be global for this whole section?
        ss_total = np.sum(y ** 2) - ((1.0 / n_obs) * np.sum(y) ** 2)
        ss_residual = slope * (np.sum(x * y) - ((1.0 / n_obs) * np.sum(x) * np.sum(y)))
        # sys.stderr.write("ss_total = %f | ss_residual = %f\n" % (ss_total, ss_residual))
        ss_error = ss_total - ss_residual  # TODO is this ss_error negative for "all linear" peptides?
        standard_error = np.sqrt(ss_error / (n_obs - 2.0))
        # sys.stderr.write("ss_error = %f; standard_error = %f\n" % (ss_error, standard_error))
        alpha = 0.68  # one standard deviation = 68; two standard deviations = 95
        t = scipy.stats.t.ppf((1.0 - (alpha / 2)), (n_obs - 2.0))

        # generate some new data
        x_i = np.linspace(min(x), max(x), num=bins, dtype=float)
        # make sure to estimate the PI of the intercept for linear segment where x_i might not include 0
        if float(0) not in x_i:
            x_i = np.append(x_i, float(0))
        y_i = (slope * x_i) + intercept

        all_pred_int = [];
        pred_int_sqrt_list = []

        # iterate over each resampled/binned point (x_i) to calculate the uncertainty for that point
        for i in x_i:
            # prediction interval calculation
            # these np.mean -- should they be the y that the model gives, for this x?
            # mean of x will just be x since it's replicates of the same point??
            # and i == x
            this_pred_int = t * standard_error * np.sqrt(
                1.0 + (1.0 / n_obs) + (((float(i) - mean_x) ** 2) / (np.sum(x) - ((1 / n_obs) * (np.sum(x) ** 2)))))
            pred_int_sqrt = np.sqrt(
                1.0 + (1.0 / n_obs) + (((float(i) - mean_x) ** 2) / (np.sum(x) - ((1 / n_obs) * (np.sum(x) ** 2)))))
            pred_int_sqrt_list.append(pred_int_sqrt)
            all_pred_int.append(abs(this_pred_int))
            # sys.stderr.write("x value %f: %f pred_int_bits, %f pred_int_num, %f pred_int_denom, %f prediction interval\n" %
            #                 (float(i), pred_int_bits, pred_int_num, pred_int_denom, this_pred_int))

        # print a bunch of info about the PI calculation
        #print "x_i:", x_i
        #print "prediction intervals:", all_pred_int
        #print "predicted y_i:", y_i

        # take the maximum of all the_pred_int
        #pred_int = max(all_pred_int)

        predint_df = pd.DataFrame({'x_i': x_i, 'y_i': y_i, 'predint': all_pred_int})

        # add a column for y_{x_0} - PI_{x_0} / y_{x_i}
        predint_df['Mstatistic'] = np.abs((predint_df.loc[predint_df['x_i'] == 0.0, 'predint'].iloc[0] - predint_df.loc[predint_df['x_i'] == 0.0, 'y_i'].iloc[0])/predint_df['y_i'])
        #print predint_df

    return predint_df


# find the intersection of the noise and linear regime
def calculate_LOD(model_params, x):

    m_noise, b_noise, m_linear, b_linear = model_params

    if m_linear < 0:
        LOD = float('Inf')
    else:
        LOD = (b_linear - b_noise) / (m_noise - m_linear)

    curve_points = set(list(x))
    blank_point = min(curve_points)
    curve_points.remove(blank_point)
    lowest_point = min(curve_points)

    # consider some special edge cases
    if LOD > max(x):  # if the intersection is higher than the top point of the curve or is a negative number,
        LOD = float('Inf')  # then replace it with a value (Inf) indicating such
    elif LOD < lowest_point:  # if there's not at least two points below the LOD
        LOD = float('Inf')  # then replace it with a value (Inf) indicating such

    return LOD


# find the intersection of the lower PI_linear and the upper PI_noise
def calculate_LOQ(params, pi_noise, pi_linear, x):
    LOQ = (params[3] - params[1] - pi_linear - pi_noise) / (params[0] - params[2])

    # consider some special edge cases
    if LOQ > max(x):  # if the intersection is greater than the top point of the curve or is a negative number,
        LOQ = float('Inf')  # then replace it with a value (Inf or NaN) indicating such
    '''elif LOQ < 0 or np.isnan(pi_noise):  # if the LOQ is for some reason below zero
        #LOQ = float('nan')  # then replace it with NaN
        curve_points = set(list(x))
        blank_point = min(curve_points)
        curve_points.remove(blank_point)
        lowest_point = min(curve_points)
        LOQ = "<"+str(lowest_point)'''

    return LOQ


# find the intersection of the noise and linear regime
def calculate_fom(model_params, df):

    m_noise, b_noise, m_linear, b_linear = model_params

    # calculate the prediction interval for the noise segment
    intersection = (b_linear - b_noise) / (m_noise - m_linear)
    PI_noise = max(calculate_PI(df.loc[(df['curvepoint'].astype(float) < intersection)],
                                model_params[0], model_params[1])['predint'])

    if m_linear < 0:
        LOD = float('Inf')
    else:
        #LOD = (b_linear - b_noise) / (m_noise - m_linear)  # version prior to 11/26
        LOD = (b_linear - b_noise - PI_noise) / (m_noise - m_linear)

    # LOD edge cases
    curve_points = set(list(df['curvepoint']))
    curve_points.remove(min(curve_points))
    curve_points.remove(max(curve_points)) # now max is 2nd highest point
    if LOD > max(x):  # if the intersection is higher than the top point of the curve or is a negative number,
        fom_results = [float('Inf'), float('Inf'), float('Inf'), float('Inf')]
        return fom_results
    elif LOD < float(min(curve_points)):  # if there's not at least two points below the LOD
        fom_results = [float('Inf'), float('Inf'), float('Inf'), float('Inf')]
        return fom_results

    # calculate the prediction intervals around each segment
    #PI_noise, PI_linear = max_prediction_interval(subset, LOD, model_params)

    # calculate the prediction intervals for X bins over the linear range (default bins=10)
    predint_df = calculate_PI(df.loc[(df['curvepoint'].astype(float) > LOD)],
                                 model_params[2], model_params[3], bins=100)
    PI_linear = max(predint_df['predint'])

    # move down the curve points from highest to lowest, checking the
    # cv at each to find the lowest consecutive point with <= 20% CV
    predint_df = predint_df.sort_values(by='x_i', ascending=True); #print predint_df
    if predint_df.isnull().values.any():
        prev_i = np.nan
    else:
        prev_i = predint_df['x_i'][1]
        for i in predint_df['x_i'][1:].tolist():
            #print predint_df.loc[predint_df['x_i'] == i]
            if predint_df.loc[predint_df['x_i'] == i, 'Mstatistic'].iloc[0] >= np.float(0.2):
                prev_i = i
                continue
            else:
                break
    LOQ = prev_i

    # OLD WAY calculate the LOQ
    #LOQ = (model_params[3] - model_params[1] - PI_linear - PI_noise) / (model_params[0] - model_params[2])

    # LOQ edge cases
    if LOQ > float(max(set(list(df['curvepoint'])))):  # if the intersection is greater than the top point of the curve or is a negative number,
        LOQ = float('Inf')
    elif LOQ < 0:  # no idea why this should ever happen but just in case
        LOQ = float('Inf')
    #elif LOQ > float(max(curve_points)):  # require two curve points above LOQ
    #    LOQ = float('Inf')

    fom_results = [LOD, LOQ, PI_noise, PI_linear]; #print fom_results

    return fom_results


# calculate LOQs from CVs for all peptides using the CV dataframe created above
def calculate_LOQ_byCV(df):
    sys.stderr.write("Calculating LOQ by %CV.\n")

    # calculate %CV (= std/mean) for a given set of data
    def calculate_oneCV(subset_df):
        std = np.std(subset_df['area'])
        mean = np.mean(subset_df['area'])
        if mean != 0:
            cv = float((std / mean) * 100)
        else:
            cv = np.nan  # TODO what does it mean to have a mean = 0?
            # the only precursor with this problem is LTDFENLKNEYSK_2
            # std = 0
            # mean = 0
            # why is this precursor even in the dataset??

        return cv

    # initialize empty data frames to store results
    cv_colnames = ['peptide', 'curvepoint', '%CV']
    peptideCVs = pd.DataFrame(columns=cv_colnames)

    loq_colnames = ['peptide', 'loq']
    peptideLOQs = pd.DataFrame(columns=loq_colnames)

    # iterate through each peptide
    for peptide in tqdm(df.peptide.unique()):

        # subset the dataframe for this precursor
        subset_df = df.loc[(df['peptide'] == peptide)]

        # empty dataframe to store this peptides CVs
        this_peptides_CVs = pd.DataFrame(columns=cv_colnames)

        # subset the subset dataframe for each curve point
        for point in subset_df['curvepoint'].unique():
            # subset the subset dataframe for each curve point
            subsubset_df = subset_df.loc[(subset_df['curvepoint']) == point]

            # calculate the CV for the curve point
            cv = calculate_oneCV(subsubset_df)

            # store the peptide, curve point, & %cv as dataframe
            new_row = [peptide, point, cv]
            new_df_row = pd.DataFrame([new_row], columns=cv_colnames)
            peptideCVs = peptideCVs.append(new_df_row)
            this_peptides_CVs = this_peptides_CVs.append(new_df_row)

        # sort by curvepoint
        this_peptides_CVs = this_peptides_CVs.sort_values(by='curvepoint', ascending=False)
        curvepoints = this_peptides_CVs['curvepoint'].unique().tolist()

        # set the first curvepoint in case it is the one with <=20% CV
        prev_i = max(curvepoints)

        # move down the curve points from highest to lowest, checking the
        # cv at each to find the lowest consecutive point with <= 20% CV
        if this_peptides_CVs.loc[this_peptides_CVs['curvepoint'] == max(curvepoints), '%CV'].iloc[0] > 20:
            loq = np.nan
        else:
            for i in curvepoints:
                if this_peptides_CVs.loc[this_peptides_CVs['curvepoint'] == i, '%CV'].iloc[0] <= 20:
                    prev_i = i
                    continue
                else:
                    loq = prev_i
                    break

        # write the 20% cv loq to dataframe
        new_loq_row = pd.DataFrame([[peptide, loq]], columns=loq_colnames)
        peptideLOQs = peptideLOQs.append(new_loq_row)

        this_peptides_CVs.to_csv(os.path.join(output_dir, "./peptidecvs.csv"), index=False)

    peptideLOQs.to_csv(os.path.join(output_dir, "./loqsbycv.csv"), index=False)

    loq_byCV_df = pd.merge(df, pd.merge(peptideLOQs, peptideCVs, on=['peptide'], how='outer'),
                           on=['peptide', 'curvepoint'], how='outer')

    sys.stdout.write("Done calculating LOQs.\n")

    return loq_byCV_df


# create plots of the curve points, the segment fits, and the LOD/LOQ values
# make one plot in untransformed space, and the second in semilog(x)
def build_plots(x, y, model_results, intersection, intersect_PI_linear):

    plt.figure(figsize=(10, 5))
    plt.suptitle(peptide, fontsize="large")

    slope_noise, intercept_noise, slope_linear, intercept_linear = model_results

    # plot a line given a slope and intercept
    def add_line_to_plot(slope, intercept, scale, setstyle='-', setcolor='k'):
        axes = plt.gca()
        xlims = np.array(axes.get_xlim())
        x_vals = np.arange(xlims[0], xlims[1], ((xlims[1] - xlims[0]) / 100))
        y_vals = intercept + slope * x_vals
        if scale == 'semilogx':
            plt.semilogx(x_vals, y_vals, linestyle=setstyle, color=setcolor)
        elif scale == 'loglog':
            plt.loglog(x_vals, y_vals, linestyle=setstyle, color=setcolor)
        else:
            plt.plot(x_vals, y_vals, linestyle=setstyle, color=setcolor)

    ###
    ### left hand plot: linear scale x axis
    plt.subplot(1, 2, 1)
    plt.plot(x, y, 'o')  # scatterplot of the data
    add_line_to_plot(slope_noise, intercept_noise, 'linear', '-', 'g')  # add noise segment line
    if slope_linear > 0:  # add linear segment line
        add_line_to_plot(slope_linear, intercept_linear, 'linear', '-', 'g')

    if "<" in str(intersection):
        plt.axvline(x=float(intersection[1:]),
                      color='m',
                      label=('LOD < %.3e' % float(intersection[1:])))
    else:
        plt.axvline(x=intersection,
                    color='m',
                    label=('LOD = %.3e' % intersection))

    if "<" in str(intersect_PI_linear):
        plt.axvline(x=float(intersect_PI_linear[1:]),
                            color='c',
                            label=('LOQ < %.3e' % float(intersect_PI_linear[1:])))
    else:
        plt.axvline(x=intersect_PI_linear,
                    color='c',
                    label=('LOQ = %.3e' % intersect_PI_linear))

    #plt.title(peptide, y=1.08)
    plt.xlabel("curve point")
    plt.ylabel("area")

    # force axis ticks to be scientific notation so the plot is prettier
    #plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))

    # Add the prediction intervals on the left hand plot
    add_line_to_plot(slope_noise, intercept_noise + PI_noise, 'linear', '--', setcolor='0.5')
    add_line_to_plot(slope_noise, intercept_noise - PI_noise, 'linear', '--', setcolor='0.5')
    add_line_to_plot(slope_linear, intercept_linear + PI_linear, 'linear', '--', setcolor='0.5')
    add_line_to_plot(slope_linear, intercept_linear - PI_linear, 'linear', '--', setcolor='0.5')

    #plt.xlim(xmin=(-50), xmax=(30500))
    #plt.ylim(ymin=0, ymax=(7.6*10**9))


    ###
    ### right hand plot: log scale x axis (semilog x)
    plt.subplot(1, 2, 2)
    plt.semilogx(x, y, 'o')
    add_line_to_plot(slope_noise, intercept_noise, 'semilogx', '-', 'g')
    if slope_linear > 0:
        add_line_to_plot(slope_linear, intercept_linear, 'semilogx', '-', 'g')

    if "<" in str(intersection):
        plt.axvline(x=float(intersection[1:]),
                      color='m',
                      label=('LOD < %.3e' % float(intersection[1:])))
    else:
        plt.axvline(x=intersection,
                    color='m',
                    label=('LOD = %.3e' % intersection))

    if "<" in str(intersect_PI_linear):
        plt.axvline(x=float(intersect_PI_linear[1:]),
                      color='c',
                      label=('LOQ < %.3e' % float(intersect_PI_linear[1:])))
    else:
        plt.axvline(x=intersect_PI_linear,
                    color='c',
                    label=('LOQ = %.3e' % intersect_PI_linear))

    #plt.title(peptide, y=1.08)
    plt.xlabel("curve point (log10)")
    plt.ylabel("area")

    # force y axis ticks to be scientific notation so the plot is prettier (x is already semilog)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    # Add the prediction intervals on the right hand plot
    add_line_to_plot(slope_noise, intercept_noise + PI_noise, 'semilogx', '--', setcolor='0.5')
    add_line_to_plot(slope_noise, intercept_noise - PI_noise, 'semilogx', '--', setcolor='0.5')
    add_line_to_plot(slope_linear, intercept_linear + PI_linear, 'semilogx', '--', setcolor='0.5')
    add_line_to_plot(slope_linear, intercept_linear - PI_linear, 'semilogx', '--', setcolor='0.5')

    #plt.ylim(ymin=(intercept_noise - (PI_noise+0.25*PI_noise)), ymax=(max(y)+max(y)*0.05))

    # add legend with LOD and LOQ values
    legend = plt.legend(loc=9, bbox_to_anchor=(0, -0.21, 1., .102), ncol=2)
    #plt.show()
    # save the figure
    plt.savefig(os.path.join(output_dir, peptide + '.png'),
                bbox_extra_artists=(legend,), bbox_inches='tight', pad_inches=0.25)
    plt.close()


# create plots of the curve points, the segment fits, and the LOD/LOQ values
# make one plot in untransformed space, and the second in log-log
def build_loglog_plots(x, y, model_results, intersection, intersect_PI_linear):

    plt.figure(figsize=(10, 5))
    plt.suptitle(peptide, fontsize="large")

    slope_noise, intercept_noise, slope_linear, intercept_linear = model_results

    # plot a line given a slope and intercept
    def add_line_to_plot(slope, intercept, scale, setstyle='-', setcolor='k'):
        axes = plt.gca()
        xlims = np.array(axes.get_xlim())
        x_vals = np.arange(xlims[0], xlims[1], ((xlims[1] - xlims[0]) / 100))
        y_vals = intercept + slope * x_vals
        if scale == 'semilogx':
            plt.semilogx(x_vals, y_vals, linestyle=setstyle, color=setcolor)
        elif scale == 'loglog':
            plt.loglog(x_vals, y_vals, linestyle=setstyle, color=setcolor)
        else:
            plt.plot(x_vals, y_vals, linestyle=setstyle, color=setcolor)

    ###
    ### left hand plot: linear scale x axis
    plt.subplot(1, 2, 1)
    plt.plot(x, y, 'o')  # scatterplot of the data
    add_line_to_plot(slope_noise, intercept_noise, 'linear', '-', 'g')  # add noise segment line
    if slope_linear > 0:  # add linear segment line
        add_line_to_plot(slope_linear, intercept_linear, 'linear', '-', 'g')

    plt.axvline(x=intersection,
                color='m',
                label=('LOD = %.3e' % intersection))
    plt.axvline(x=intersect_PI_linear,
                color='c',
                label=('LOQ = %.3e' % intersect_PI_linear))

    #plt.title(peptide, y=1.08)
    plt.xlabel("curve point")
    plt.ylabel("area")

    # force axis ticks to be scientific notation so the plot is prettier
    #plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))

    # Add the prediction intervals on the left hand plot
    add_line_to_plot(slope_noise, intercept_noise + PI_noise, 'linear', '--', setcolor='0.5')
    add_line_to_plot(slope_noise, intercept_noise - PI_noise, 'linear', '--', setcolor='0.5')
    add_line_to_plot(slope_linear, intercept_linear + PI_linear, 'linear', '--', setcolor='0.5')
    add_line_to_plot(slope_linear, intercept_linear - PI_linear, 'linear', '--', setcolor='0.5')

    plt.ylim(ymin=min(y)*0.5, ymax=(max(y)*1.05))
    plt.xlim(xmin=min(x)-np.finfo(np.float).eps, xmax=max(x)*1.1)

    ###
    ### right hand plot: log scale x axis (semilog x)
    plt.subplot(1, 2, 2)
    plt.loglog(x+0.0001, y, 'o')
    add_line_to_plot(slope_noise, intercept_noise, 'loglog', '-', 'g')
    if slope_linear > 0:
        add_line_to_plot(slope_linear, intercept_linear, 'loglog', '-', 'g')

    plt.axvline(x=intersection,
                color='m',
                label=('LOD = %.3e' % intersection))
    plt.axvline(x=intersect_PI_linear,
                color='c',
                label=('LOQ = %.3e' % intersect_PI_linear))

    #plt.title(peptide, y=1.08)
    plt.xlabel("curve point (log10)")
    plt.ylabel("area (log10)")

    # force y axis ticks to be scientific notation so the plot is prettier (x is already semilog)
    #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    # Add the prediction intervals on the right hand plot
    add_line_to_plot(slope_noise, intercept_noise + PI_noise, 'loglog', '--', setcolor='0.5')
    add_line_to_plot(slope_noise, intercept_noise - PI_noise, 'loglog', '--', setcolor='0.5')
    add_line_to_plot(slope_linear, intercept_linear + PI_linear, 'loglog', '--', setcolor='0.5')
    add_line_to_plot(slope_linear, intercept_linear - PI_linear, 'loglog', '--', setcolor='0.5')

    plt.ylim(ymin=min(y)*0.5, ymax=(max(y)*1.05))
    plt.xlim(xmin=min(x)-np.finfo(np.float).eps, xmax=max(x)*1.1)

    # add legend with LOD and LOQ values
    legend = plt.legend(loc=9, bbox_to_anchor=(0, -0.21, 1., .102), ncol=2)
    #plt.show()
    # save the figure
    plt.savefig(os.path.join(output_dir, peptide + '.png'),
                bbox_extra_artists=(legend,), bbox_inches='tight', pad_inches=0.25)
    plt.close()


# create plots of the curve points, the segment fits, and the LOD/LOQ values
# make one plot in untransformed space, and the second zoomed in around the LOD/LOQ
def build_other_plots(x, y, model_results, intersection, intersect_PI_linear):

    plt.figure(figsize=(10, 5))
    plt.suptitle(peptide, fontsize="large")

    slope_noise, intercept_noise, slope_linear, intercept_linear = model_results

    # plot a line given a slope and intercept
    def add_line_to_plot(slope, intercept, scale, setstyle='-', setcolor='k'):
        axes = plt.gca()
        xlims = np.array(axes.get_xlim())
        x_vals = np.arange(xlims[0], xlims[1], ((xlims[1] - xlims[0]) / 100))
        y_vals = intercept + slope * x_vals
        if scale == 'semilogx':
            plt.semilogx(x_vals, y_vals, linestyle=setstyle, color=setcolor)
        elif scale == 'loglog':
            plt.loglog(x_vals, y_vals, linestyle=setstyle, color=setcolor)
        else:
            plt.plot(x_vals, y_vals, linestyle=setstyle, color=setcolor)

    ###
    ### left hand plot: linear scale x axis
    plt.subplot(1, 2, 1)
    plt.plot(x, y, 'o')  # scatterplot of the data
    add_line_to_plot(slope_noise, intercept_noise, 'linear', '-', 'g')  # add noise segment line
    if slope_linear > 0:  # add linear segment line
        add_line_to_plot(slope_linear, intercept_linear, 'linear', '-', 'g')

    plt.axvline(x=intersection,
                color='m',
                label=('LOD = %.3e' % intersection))
    plt.axvline(x=intersect_PI_linear,
                color='c',
                label=('LOQ = %.3e' % intersect_PI_linear))

    #plt.title(peptide, y=1.08)
    plt.xlabel("curve point")
    plt.ylabel("area")

    # force axis ticks to be scientific notation so the plot is prettier
    #plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))

    # Add the prediction intervals on the left hand plot
    add_line_to_plot(slope_noise, intercept_noise + PI_noise, 'linear', '--', setcolor='0.5')
    add_line_to_plot(slope_noise, intercept_noise - PI_noise, 'linear', '--', setcolor='0.5')
    add_line_to_plot(slope_linear, intercept_linear + PI_linear, 'linear', '--', setcolor='0.5')
    add_line_to_plot(slope_linear, intercept_linear - PI_linear, 'linear', '--', setcolor='0.5')

    plt.ylim(ymin=-1.0, ymax=(max(y)*1.01))
    plt.xlim(xmin=min(x)-np.finfo(np.float).eps, xmax=max(x)*1.1)

    ###
    ### right hand plot: zoomed x axis
    plt.subplot(1, 2, 2)
    plt.plot(x, y, 'o')
    add_line_to_plot(slope_noise, intercept_noise, 'linear', '-', 'g')
    if slope_linear > 0:
        add_line_to_plot(slope_linear, intercept_linear, 'linear', '-', 'g')

    plt.axvline(x=intersection,
                color='m',
                label=('LOD = %.3e' % intersection))
    plt.axvline(x=intersect_PI_linear,
                color='c',
                label=('LOQ = %.3e' % intersect_PI_linear))

    #plt.title(peptide, y=1.08)
    plt.xlabel("curve point")
    plt.ylabel("area)")

    # force y axis ticks to be scientific notation so the plot is prettier (x is already semilog)
    #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    # Add the prediction intervals on the right hand plot
    add_line_to_plot(slope_noise, intercept_noise + PI_noise, 'linear', '--', setcolor='0.5')
    add_line_to_plot(slope_noise, intercept_noise - PI_noise, 'linear', '--', setcolor='0.5')
    add_line_to_plot(slope_linear, intercept_linear + PI_linear, 'linear', '--', setcolor='0.5')
    add_line_to_plot(slope_linear, intercept_linear - PI_linear, 'linear', '--', setcolor='0.5')

    plt.ylim(ymin=-1.0)
    plt.xlim(xmin=intersection*0.5, xmax=intersect_PI_linear*1.5)

    # add legend with LOD and LOQ values
    legend = plt.legend(loc=9, bbox_to_anchor=(0, -0.21, 1., .102), ncol=2)
    #plt.show()
    # save the figure
    plt.savefig(os.path.join(output_dir, peptide + '.png'),
                bbox_extra_artists=(legend,), bbox_inches='tight', pad_inches=0.25)
    plt.close()


# my solve for the piecewise fit
def fit_by_scipycurvefit(x, y):

    # define each of the linear segments and do a piecewise fit
    def two_lines(x, a, b, c, d):
        a = 0  # slope of the noise (a) should be zero
        noise = a * x + b
        linear = c * x + d
        return np.maximum(noise, linear)

    # parameter initialization
    def initialize_params(x, y):
        subsetdf = pd.DataFrame({'curvepoint': pd.to_numeric(x), 'area': y})
        mean_y = subsetdf.groupby('curvepoint')['area'].mean()  # find the mean response area for each curve point

        # find the top point, second-top point, and bottom points of the curve data
        # conc_list = list(pd.to_numeric(subsetdf['curvepoint'].drop_duplicates()))
        conc_list = list(set(x))
        top_point = max(conc_list)
        conc_list.remove(top_point)
        second_top = max(conc_list)
        bottom_point = min(conc_list)
        conc_list.remove(bottom_point)
        second_bottom = min(conc_list)
        conc_list.remove(second_bottom)
        third_bottom = min(conc_list)

        '''# take the top 10% points to fit the horizontal line
        conc_list = list(x)
        top10pct = int(.1*len(conc_list))
        # take the bottom 50% to fit the diagonal?
        bottom50pct = int(.5 * len(conc_list))
        noise_intercept = mean_y[bottom50pct]'''

        # using the means, calculate a slope (y1-y2/x1-x2)
        linear_slope = (mean_y[second_top] - mean_y[top_point]) / (second_top - top_point)
        # find the noise intercept using average of bottom three points
        noise_intercept = (mean_y[bottom_point]+mean_y[second_bottom]+mean_y[third_bottom])/3.0
        # find the linear intercept using linear slope (b = y-mx) and the top point
        linear_intercept = mean_y[top_point] - (linear_slope * top_point)
        # noise_slope needs to be zero
        noise_slope = 0.0

        # edge case catch?
        if noise_intercept < linear_intercept:
            noise_intercept = linear_intercept * 1.05
            #sys.stderr.write("\nawh geeze noise_intercept < linear_intercept\n")

        return noise_slope, noise_intercept, linear_slope, linear_intercept

    # use non-linear least squares to fit the two functions (noise and linear) to data.
    try:
        model_parameters, cov = curve_fit(two_lines, x, y, initialize_params(x,y))
        slope_noise = model_parameters[0]
        intercept_noise = model_parameters[1]
        slope_linear = model_parameters[2]
        intercept_linear = model_parameters[3]
    except:  # catch for when a peptide can't be fit for whatever reason
        sys.stderr.write("Peptide %s could not be curve_fit." % peptide)
        model_parameters = [np.nan, np.nan, np.nan, np.nan]
        slope_noise = np.nan
        intercept_noise = np.nan
        slope_linear = np.nan
        intercept_linear = np.nan

    return model_parameters


# yang's solve for the piecewise fit
def fit_by_lmfit(x, y):

    def fcn2min(params, x, data, weight):
        a = params['a'].value
        b = params['b'].value
        c = params['c'].value
        model = np.maximum(c, a*x+b)
        return (model - data) * weight

    # parameter initialization
    def initialize_params(x, y):
        subsetdf = pd.DataFrame({'curvepoint': pd.to_numeric(x), 'area': y})
        mean_y = subsetdf.groupby('curvepoint')['area'].mean()  # find the mean response area for each curve point

        # find the top point, second-top point, and bottom points of the curve data
        # conc_list = list(pd.to_numeric(subsetdf['curvepoint'].drop_duplicates()))
        conc_list = list(set(x))
        top_point = max(conc_list)
        conc_list.remove(top_point)
        second_top = max(conc_list)
        bottom_point = min(conc_list)
        conc_list.remove(bottom_point)
        second_bottom = min(conc_list)
        conc_list.remove(second_bottom)
        third_bottom = min(conc_list)

        '''# take the top 10% points to fit the horizontal line
        conc_list = list(x)
        top10pct = int(.1*len(conc_list))
        # take the bottom 50% to fit the diagonal?
        bottom50pct = int(.5 * len(conc_list))
        noise_intercept = mean_y[bottom50pct]'''

        # using the means, calculate a slope (y1-y2/x1-x2)
        linear_slope = (mean_y[second_top] - mean_y[top_point]) / (second_top - top_point)
        # find the noise intercept using average of bottom three points
        noise_intercept = (mean_y[bottom_point]+mean_y[second_bottom]+mean_y[third_bottom])/3.0
        # find the linear intercept using linear slope (b = y-mx) and the top point
        linear_intercept = mean_y[top_point] - (linear_slope * top_point)

        # edge case catch?
        if noise_intercept < linear_intercept:
            noise_intercept = linear_intercept * 1.05
            #sys.stderr.write("\nawh geeze noise_intercept < linear_intercept\n")

        return linear_slope, linear_intercept, noise_intercept

    params = Parameters()
    initial_a, initial_b, initial_c = initialize_params(x,y)
    initial_cminusb = initial_c - initial_b
    params.add('a', value=initial_a, min=0, vary=True)
    params.add('b', value=initial_b, vary=True)
    params.add('c_minus_b', value=initial_cminusb, min=np.finfo(np.float).eps, vary=True)
    params.add('c', expr='b + c_minus_b')

    weights = np.minimum(1/(np.asarray(np.sqrt(x), dtype=float)+np.finfo(np.float).eps), 1000)  # inverse weights
    #weights =  np.minimum(np.asarray(((x+1)/(x+1)), dtype=float), 1000) # fake weights
    #sys.stderr.write("Min weight %f, max weight %f\n" % (min(weights),max(weights)) )
    #print set(weights)
    minner = Minimizer(fcn2min, params, fcn_args=(x, y, weights ))
    result = minner.minimize()

    #final = y + result.residual
    #print report_fit(result)
    # TODO check if maybe there's someway to get better model parameters

    return result, minner






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
                    help='specify an output path for figures of merit and plots')
parser.add_argument('--plot', default='y', type=str,
                    help='yes/no (y/n) to create individual calibration curve plots for each peptide')

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

# initialize empty data frame to store figures of merit
peptide_fom = pd.DataFrame(columns=['peptide', 'LOD', 'LOQ',
                                    'slope_linear', 'intercept_linear', 'intercept_noise',
                                    'PI_noise', 'PI_linear'])
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

    # set up the model and the parameters
    ## lmfit approach by yang!
    result, min_func = fit_by_lmfit(x, y)
    slope_noise = 0.0
    intercept_noise = result.params['c'].value
    slope_linear = result.params['a'].value
    intercept_linear = result.params['b'].value
    ## my original fit
    #result = fit_by_curvefit(x, y)
    #slope_noise = result[0]
    #intercept_noise = result[1]
    #slope_linear = result[2]
    #intercept_linear = result[3]

    model_parameters = np.asarray([slope_noise, intercept_noise, slope_linear, intercept_linear])

    # find the intersection of the noise and linear segments
    #LOD = calculate_LOD(model_parameters, x)

    # find the intersection of the lower PI_linear and the upper PI_noise
    #LOQ = calculate_LOQ(model_parameters, PI_noise, PI_linear, x)

    LOD, LOQ, PI_noise, PI_linear = calculate_fom(model_parameters, subset)

    if plot_or_not == 'y':
        # make a plot of the curve points and the fit, in both linear and log space
        build_plots(x, y, model_parameters, LOD, LOQ)

    # make a dataframe row with the peptide and its figures of merit
    new_row = [peptide, LOD, LOQ, slope_linear, intercept_linear, intercept_noise, PI_noise, PI_linear]
    new_df_row = pd.DataFrame([new_row], columns=['peptide', 'LOD', 'LOQ',
                                                  'slope_linear', 'intercept_linear', 'intercept_noise',
                                                  'PI_noise', 'PI_linear'])
    peptide_fom = peptide_fom.append(new_df_row)

    #sys.stderr.write("%s LOD=%f, LOQ=%f\n" % (peptide, LOD, LOQ))
    #sys.stderr.write("%s slope_linear=%f, intercept_linear=%f, intercept_noise=%f\n" %
    #                 (peptide, slope_linear, intercept_linear, intercept_noise))
    #sys.stderr.write("%s PI_noise=%f, PI_linear=%f\n" % (peptide, PI_noise, PI_linear))

peptide_fom.to_csv(path_or_buf=os.path.join(output_dir,'figuresofmerit.csv'),
                        index=False)

