

import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy
from tqdm import tqdm
import argparse
from lmfit import Minimizer, Parameters



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


def calculate_PI(subset_df, slope, intercept, bins=10):

    if not subset_df.size:  # if there's no data for this segment, don't bother with a prediction interval
        predint_df = pd.DataFrame({'x_i': [np.nan], 'y_i': [np.nan], 'predint': [np.nan], 'Mstatistic': [np.nan]})

    else:
        x = np.array(subset_df['curvepoint'], dtype=float)
        y = np.array(subset_df['area'], dtype=float)
        n_obs = float(y.size)  # number of observations in the segment
        mean_x = np.mean(x)
        x_weights = 1/(np.asarray(np.sqrt(x), dtype=float)+np.finfo(np.float).eps)
        yhat = (slope*x) + intercept

        # fit statistics, should be global for this whole section?
        #ss_total = np.sum(y**2) - ((1.0/n_obs)*(np.sum(y)**2)); #print "\nTSS: ", ss_total
        ss_total = np.sum((y-np.mean(y))**2)
        #ss_residual = slope * (np.sum(x*y) - ((1.0/n_obs)*(np.sum(x)*np.sum(y)))); #print "RSS: ", ss_residual
        ss_residual = np.sum((yhat-np.mean(y))**2)
        #sys.stderr.write("ss_total = %f | ss_residual = %f\n" % (ss_total, ss_residual))
        ss_expl = ss_total - ss_residual  # TODO is this ss_expl negative for "all linear" peptides?
        standard_error = np.sqrt(ss_expl / (n_obs - 2.0)); #print "ESS: ", ss_expl
        #sys.stderr.write("ss_expl = %f; standard_error = %f\n" % (ss_expl, standard_error))
        alpha = 0.68  # one standard deviation = 68; two standard deviations = 95
        t = scipy.stats.t.ppf((1.0 - (alpha / 2)), (n_obs - 2.0))

        ##
        ## Total Error = bias + 1.65*CV_A
        ##

        # generate some new data
        x_i = np.linspace(min(x), max(x), num=bins, dtype=float)
        # make sure to estimate the PI of the intercept for linear segment where x_i might not include 0
        if float(0) not in x_i:
            x_i = np.append(x_i, float(0))
        y_i = (slope * x_i) + intercept

        all_pred_int = []
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

        predint_df = pd.DataFrame({'x_i': x_i, 'y_i': y_i, 'predint': all_pred_int})

        # add a column for y_{x_0} - PI_{x_0} / y_{x_i}
        predint_df['Mstatistic'] = np.abs((predint_df.loc[predint_df['x_i'] == 0.0, 'predint'].iloc[0] - predint_df.loc[predint_df['x_i'] == 0.0, 'y_i'].iloc[0])/predint_df['y_i'])
        #print predint_df

    return predint_df


# find the intersection of the noise and linear regime
def calculate_fom_SAVE(model_params, df, conf_int):

    m_noise, b_noise, m_linear, b_linear = model_params

    # calculate the prediction interval for the noise segment
    intersection = (b_linear - b_noise) / (m_noise - m_linear)
    PI_noise = max(calculate_PI(df.loc[(df['curvepoint'].astype(float) < intersection)],
                                model_params[0], model_params[1])['predint']); #print "PI_noise: ", PI_noise

    if m_linear < 0:  # catch edge cases where the slope of the linearity
        LOD = float('Inf')
    else:
        LOD = (b_linear - b_noise - PI_noise) / (m_noise - m_linear)

    # LOD edge cases
    curve_points = set(list(df['curvepoint']))
    curve_points.remove(min(curve_points))
    curve_points.remove(max(curve_points))  # now max is 2nd highest point
    if LOD > max(x):  # if the intersection is higher than the top point of the curve or is a negative number,
        fom_results = [float('Inf'), float('Inf'), float('Inf'), float('Inf')]
        return fom_results
    elif LOD < float(min(curve_points)):  # if there's not at least two points below the LOD
        fom_results = [float('Inf'), float('Inf'), float('Inf'), float('Inf')]
        return fom_results

    # calculate the prediction intervals for X bins over the linear range (default bins=10)
    predint_df = calculate_PI(df.loc[(df['curvepoint'].astype(float) > LOD)],
                                 model_params[2], model_params[3], bins=100)
    PI_linear = max(predint_df['predint'])  # we don't use this in post 11/26 version

    # move down the curve points from highest to lowest, checking the
    # cv at each to find the lowest consecutive point with <= 20% CV
    predint_df = predint_df.sort_values(by='x_i', ascending=True); #print predint_df
    predint_df.to_csv(path_or_buf=os.path.join(output_dir, list(set(df['peptide']))[0]+'.csv'),
                        index=False)
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

    # LOQ edge cases
    if LOQ >= float(max(set(list(df['curvepoint'])))):
        LOQ = float('Inf')
    elif LOQ < 0:  # no idea why this should ever happen but just in case
        LOQ = float('Inf')

    fom_results = [LOD, LOQ, PI_noise, PI_linear]; #print fom_results

    return fom_results


# create plots of the curve points, the segment fits, and the LOD/LOQ values
def build_plots_SAVE(x, y, model_results, intersection, intersect_PI_linear):

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
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))

    # Add the prediction intervals on the left hand plot
    add_line_to_plot(slope_noise, intercept_noise + PI_noise, 'linear', '--', setcolor='0.5')
    #add_line_to_plot(slope_noise, intercept_noise - PI_noise, 'linear', '--', setcolor='0.5')
    #add_line_to_plot(slope_linear, intercept_linear + PI_linear, 'linear', '--', setcolor='0.5')
    add_line_to_plot(slope_linear, intercept_linear - PI_linear, 'linear', '--', setcolor='0.5')

    plt.xlim(xmin=min(x)-max(x)*0.01)  # anchor x and y to 0-ish
    plt.ylim(ymin=min(y)-max(y)*0.01, ymax=(max(y))*1.01)


    ###
    ### right hand plot: zoom in on LOD/LOQ scaled x axis
    plt.subplot(1, 2, 2)
    plt.semilogx(x, y, 'o')
    add_line_to_plot(slope_noise, intercept_noise, 'semilogx', '-', 'g')
    if slope_linear > 0:
        add_line_to_plot(slope_linear, intercept_linear, 'semilogx', '-', 'g')

    plt.axvline(x=intersection,
                color='m',
                label=('LOD = %.3e' % intersection))

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
    #add_line_to_plot(slope_noise, intercept_noise - PI_noise, 'linear', '--', setcolor='0.5')
    #add_line_to_plot(slope_linear, intercept_linear + PI_linear, 'linear', '--', setcolor='0.5')
    add_line_to_plot(slope_linear, intercept_linear - PI_linear, 'semilogx', '--', setcolor='0.5')

    #if np.isfinite(LOD) and np.isfinite(LOQ):
    #    plt.xlim(xmin=(LOD*0.7), xmax=(LOQ*1.3))  # anchor x to the interesting bits
    #    plt.ylim(ymin=(-0.0001), ymax=(max(y)))

    plt.xlim(xmin=min(x)-max(x)*0.01)  # anchor x and y to 0-ish
    plt.ylim(ymin=min(y)-max(y)*0.01, ymax=(max(y))*1.01)

    # add legend with LOD and LOQ values
    legend = plt.legend(loc=9, bbox_to_anchor=(0, -0.21, 1., .102), ncol=2)
    #plt.show()
    # save the figure
    plt.savefig(os.path.join(output_dir, peptide + '.png'),
                bbox_extra_artists=(legend,), bbox_inches='tight', pad_inches=0.25)
    plt.close()


# yang's solve for the piecewise fit using lmfit Minimize function
def fit_by_lmfit_yang(x, y):

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

        # using the means, calculate a slope (y1-y2/x1-x2)
        linear_slope = (mean_y[second_top] - mean_y[top_point]) / (second_top - top_point)
        # find the noise intercept using average of bottom three points
        noise_intercept = mean_y[bottom_point]
        # find the linear intercept using linear slope (b = y-mx) and the top point
        linear_intercept = mean_y[top_point] - (linear_slope * top_point)

        # edge case catch?
        if noise_intercept < linear_intercept:
            noise_intercept = linear_intercept * 1.05

        return linear_slope, linear_intercept, noise_intercept

    params = Parameters()
    initial_a, initial_b, initial_c = initialize_params(x,y)
    initial_cminusb = initial_c - initial_b
    params.add('a', value=initial_a, min=0, vary=True)
    params.add('b', value=initial_b, vary=True)
    params.add('c_minus_b', value=initial_cminusb, min=np.finfo(np.float).eps, vary=True)
    params.add('c', expr='b + c_minus_b')

    weights = np.minimum(1/(np.asarray(np.sqrt(x), dtype=float)+np.finfo(np.float).eps), 1000)  # inverse weights
    minner = Minimizer(fcn2min, params, fcn_args=(x, y, weights))
    result = minner.minimize()

    #final = y + result.residual
    #print report_fit(result)

    return result, minner


# find the intersection of the noise and linear regime
def calculate_lod(model_params, df):

    m_noise, b_noise, m_linear, b_linear = model_params

    # calculate the standard deviation for the noise segment
    intersection = (b_linear - b_noise) / (m_noise - m_linear); #print intersection
    std_noise = np.std(df['area'].loc[(df['curvepoint'].astype(float) < intersection)]); #print "var_noise: ", var_noise

    if m_linear < 0:  # catch edge cases where there is only noise in the curve
        LOD = float('Inf')
    else:
        LOD = (b_linear - b_noise - std_noise) / (m_noise - m_linear)

    # LOD edge cases
    curve_points = set(list(df['curvepoint']))
    curve_points.remove(min(curve_points))
    curve_points.remove(max(curve_points))  # now max is 2nd highest point
    if LOD > max(x):  # if the intersection is higher than the top point of the curve or is a negative number,
        lod_results = [float('Inf'), float('Inf')]
    elif LOD < float(min(curve_points)):  # if there's not at least two points below the LOD
        lod_results = [float('Inf'), float('Inf')]

    lod_results = [LOD, std_noise]


    return lod_results


# find the intersection of the noise and linear regime
def calculate_fom(model_params, df, boot_results):

    m_noise, b_noise, m_linear, b_linear = model_params

    # calculate the standard deviation for the noise segment
    intersection = (b_linear - b_noise) / (m_noise - m_linear); #print intersection
    std_noise = np.std(df['area'].loc[(df['curvepoint'].astype(float) < intersection)]); #print "var_noise: ", var_noise

    if m_linear < 0:  # catch edge cases where there is only noise in the curve
        LOD = float('Inf')
    else:
        LOD = (b_linear - b_noise - std_noise) / (m_noise - m_linear)

    #sys.stderr.write("b_noise = %f; std_noise = %f; LOD = %f\n" % (b_noise, std_noise, LOD) )

    # LOD edge cases
    curve_points = set(list(df['curvepoint']))
    curve_points.remove(min(curve_points))
    curve_points.remove(max(curve_points))  # now max is 2nd highest point
    if LOD > max(x):  # if the intersection is higher than the top point of the curve or is a negative number,
        fom_results = [float('Inf'), float('Inf'), float('Inf'), float('Inf')]
        return fom_results
    elif LOD < float(min(curve_points)):  # if there's not at least two points below the LOD
        fom_results = [float('Inf'), float('Inf'), float('Inf'), float('Inf')]
        return fom_results

    # how to define the LOQ using the boot_results??
    LOQ = 0.0  # place holder for now
    PI_linear = 0.0  # placeholder for now

    # LOQ edge cases
    if LOQ >= float(max(set(list(df['curvepoint'])))):
        LOQ = float('Inf')
    elif LOQ <= 0:  # no idea why this should ever happen but just in case
        LOQ = float('Inf')

    fom_results = [LOD, LOQ, std_noise, PI_linear]; #print fom_results

    return fom_results


# determine prediction interval by bootstrapping
def bootstrap_pi(df, new_x, bootreps=100):

    def bootstrap_once(df, new_x, iter):

        resampled_df = df.sample(n=len(df), replace=True); #print resampled_df.head()
        boot_x = np.array(resampled_df['curvepoint'], dtype=float)
        boot_y = np.array(resampled_df['area'], dtype=float)
        fit_result, mini_result = fit_by_lmfit_yang(boot_x, boot_y); #print fit_result.params.pretty_print()
        new_intersection = float('Inf')

        if fit_result.params['a'].value > 0:
            new_intersection = (fit_result.params['b'].value - fit_result.params['c'].value) / (0. - fit_result.params['a'].value)
            # consider some special edge cases
            if new_intersection > max(boot_x):  # if the intersection is higher than the top point of the curve or is a negative number,
                new_intersection = float('Inf')  # then replace it with a value (Inf) indicating such
                #sys.stderr.write("Intersection greater than max boot_x.\n")
            elif new_intersection < 0.:  # if the intersection is less than zero
                new_intersection = float('Inf')  # then replace it with a value (Inf) indicating such
                #sys.stderr.write("Intersection less than zero.\n")

        #print new_intersection
        yresults = []
        for i in new_x:
            if i <= new_intersection:  # if the new_x is in the noise,
                pred_y = fit_result.params['c'].value
            elif i > new_intersection:
                pred_y = (fit_result.params['a'].value * i) + fit_result.params['b'].value
            yresults.append(pred_y)

        iter_results = pd.DataFrame(data={'boot_x': new_x, iter: yresults})

        return iter_results

    # Bootstrap the data (e.g. resample the data with replacement)
    # get the regression prediction (new_y) at each new_x
    boot_results = pd.DataFrame(data={'boot_x': new_x})
    for i in range(bootreps):
        iteration_results = bootstrap_once(df, new_x, i); #print iteration_results
        boot_results = pd.merge(boot_results, iteration_results, on='boot_x')

    # reshape the bootstrap results to be columns=boot_x and rows=boot_y results (each iteration is a row)
    boot_results = boot_results.T; #print boot_results.head()
    boot_results.columns = boot_results.iloc[0]; #print boot_results.head()
    boot_results = boot_results.drop(['boot_x'], axis='rows')
    boot_results.to_csv(path_or_buf=os.path.join(output_dir,
                                                 'bootstrapresults_' + str(list(set(df['peptide']))[0]) + '.csv'),
                        index=True)

    # calculate lower and upper 95% PI
    boot_summary = (boot_results.describe(percentiles=[.05, .95])).T
    boot_summary['boot_x'] = boot_summary.index; #print boot_results
    #boot_summary['numpy_std'] = np.std(boot_results, axis=0, ddof=1)

    # calculate the bootstrapped CV
    boot_summary['boot_cv'] = boot_summary['std']/boot_summary['mean']
    #boot_summary['boot_cv_numpystd'] = boot_summary['numpy_std']/boot_summary['mean']

    boot_summary.to_csv(path_or_buf=os.path.join(output_dir,
                                                 'bootstrapsummary_'+str(list(set(df['peptide']))[0])+'.csv'),
                     index=True)

    return boot_summary


# TEST PLOTS WITH BOOTSTRAPPED PREDINT'S PLOTTED
def build_plots(x, y, model_results, intersection, intersect_PI_linear, PI_noise, predint_results):

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
    #plt.plot(predint_results['boot_x'], predint_results['5%'], 'x')
    plt.plot(predint_results['boot_x'], (predint_results['mean']-predint_results['std']), 'x')
    #plt.plot(predint_results['boot_x'], predint_results['95%'], 'x')
    add_line_to_plot(slope_noise, intercept_noise, 'linear', '-', 'g')  # add noise segment line
    add_line_to_plot(slope_noise, (intercept_noise + PI_noise), 'linear', '--', setcolor='0.5')
    if slope_linear > 0:  # add linear segment line
        add_line_to_plot(slope_linear, intercept_linear, 'linear', '-', 'g')

    plt.axvline(x=intersection,
                color='m',
                label=('LOD = %.3e' % intersection))

    '''plt.axvline(x=intersect_PI_linear,
                color='c',
                label=('LOQ = %.3e' % intersect_PI_linear))'''

    #plt.title(peptide, y=1.08)
    plt.xlabel("curve point")
    plt.ylabel("area")

    # force axis ticks to be scientific notation so the plot is prettier
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))

    plt.xlim(xmin=min(x)-max(x)*0.01)  # anchor x and y to 0-ish
    plt.ylim(ymin=min(y)-max(y)*0.01, ymax=(max(y))*1.01)


    ###
    ### right hand plot: zoom in on LOD/LOQ scaled x axis
    plt.subplot(1, 2, 2)
    plt.semilogx(x, y, 'o')
    plt.semilogx(predint_results['boot_x'], (predint_results['mean']-predint_results['std']), 'x')
    #plt.semilogx(predint_results['boot_x'], predint_results['95%'], 'x')
    add_line_to_plot(slope_noise, intercept_noise, 'semilogx', '-', 'g')
    add_line_to_plot(slope_noise, (intercept_noise + PI_noise), 'semilogx', '--', setcolor='0.5')

    if slope_linear > 0:
        add_line_to_plot(slope_linear, intercept_linear, 'semilogx', '-', 'g')

    plt.axvline(x=intersection,
                color='m',
                label=('LOD = %.3e' % intersection))

    '''plt.axvline(x=intersect_PI_linear,
                color='c',
                label=('LOQ = %.3e' % intersect_PI_linear))'''

    #plt.title(peptide, y=1.08)
    plt.xlabel("curve point (log10)")
    plt.ylabel("area")

    # force y axis ticks to be scientific notation so the plot is prettier (x is already semilog)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    plt.xlim(xmin=min(x)-max(x)*0.01)  # anchor x and y to 0-ish
    plt.ylim(ymin=min(y)-max(y)*0.01, ymax=(max(y))*1.01)

    #plt.show()
    # save the figure
    # add legend with LOD and LOQ values
    #legend = plt.legend(loc=9, bbox_to_anchor=(0, -0.21, 1., .102), ncol=2)
    plt.savefig(os.path.join(output_dir, peptide + '.png'),
                #bbox_extra_artists=(legend,),
                bbox_inches='tight', pad_inches=0.25)
    plt.close()




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

    # set up the model and the parameters (yang's lmfit minimize function approach)
    result, mini = fit_by_lmfit_yang(x,y)
    slope_noise = 0.0
    slope_linear = result.params['a'].value
    intercept_linear = result.params['b'].value
    intercept_noise = result.params['c'].value

    model_parameters = np.asarray([slope_noise, intercept_noise, slope_linear, intercept_linear])

    lod_vals = calculate_lod(model_parameters, subset)
    LOD, PI_noise = lod_vals

    # calculate the prediction intervals for X bins over the linear range (default bins=10)
    # x_i: # of "new" concentration points to calculate y for (make this user-defined?)
    x_i_noise = np.linspace(min(x), LOD, num=len([val for val in x if val < LOD]), dtype=float)
    x_i_linear = np.linspace(LOD, max(x), num=len([val for val in x if val > LOD]), dtype=float)
    x_i = np.unique((np.concatenate((x_i_noise, x_i_linear), axis=None))); #print x_i
    # bootreps: number of times to do the bootstrapping for each new x-point
    bootstrap_df = bootstrap_pi(subset, new_x=x_i, bootreps=100)

    fom = calculate_fom(model_parameters, subset, bootstrap_df); #print fom
    LOD, LOQ, PI_noise, PI_linear = fom

    if plot_or_not == 'y':
        # make a plot of the curve points and the fit, in both linear and log space
        build_plots(x, y, model_parameters, LOD, LOQ, PI_noise, bootstrap_df)

    # make a dataframe row with the peptide and its figures of merit
    new_row = [peptide, LOD, LOQ, slope_linear, intercept_linear, intercept_noise, PI_noise, PI_linear]
    new_df_row = pd.DataFrame([new_row], columns=['peptide', 'LOD', 'LOQ',
                                                  'slope_linear', 'intercept_linear', 'intercept_noise',
                                                  'PI_noise', 'PI_linear'])
    peptide_fom = peptide_fom.append(new_df_row)

    peptide_fom.to_csv(path_or_buf=os.path.join(output_dir, 'figuresofmerit.csv'),
                       index=False)
