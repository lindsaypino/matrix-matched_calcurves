import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
plt.style.use('seaborn-v0_8-whitegrid')

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


# calculate %CV (= std/mean) for a given set of data
def calculate_oneCV(subset_df):
    std = np.std(subset_df['area'], ddof=1)  # setting "ddof" to 1 for sample std (default ddof=0 for population std)
    mean = np.mean(subset_df['area'])
    if mean != 0:
        cv = float((std / mean))
    else:
        cv = np.nan  # TODO what does it mean to have a mean = 0?
        # the only precursor with this problem is LTDFENLKNEYSK_2
        # std = 0
        # mean = 0
        # why is this precursor even in the dataset??

    return cv


# calculate LOQs from CVs for all peptides using the CV dataframe created above
def calculate_LOQ_byCV(df):
    sys.stderr.write("Calculating LOQ by %CV.\n")

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

        # calculate points' CV
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
        this_peptides_CVs = this_peptides_CVs.sort_values(by='curvepoint', ascending=True)
        curvepoints = this_peptides_CVs['curvepoint'].unique().tolist()
        curvepoints = [x for x in curvepoints if x > 0]

        # move down the curve points from lowest to highest, checking the
        # cv at each to find the lowest point with <= 20% CV
        loq = np.nan
        for i in curvepoints:
            if this_peptides_CVs.loc[this_peptides_CVs['curvepoint'] == i, '%CV'].iloc[0] <= 0.2:
                loq = i  # loq is lowest point with <= 20% cv
                break

        # write the 20% cv loq to dataframe
        new_loq_row = pd.DataFrame([[peptide, loq]], columns=loq_colnames)
        peptideLOQs = peptideLOQs.append(new_loq_row)


    peptideLOQs.to_csv(os.path.join(output_dir, "./loqsbycv.csv"), index=False)
    peptideCVs.to_csv(os.path.join(output_dir, "./peptidecvs.csv"), index=False)

    loq_byCV_df = pd.merge(df, pd.merge(peptideLOQs, peptideCVs, on=['peptide'], how='outer'),
                           on=['peptide', 'curvepoint'], how='outer')

    sys.stdout.write("Done calculating LOQs.\n")

    return loq_byCV_df


# create the plots showing area and CV values
def make_plotSAVE(df):
    x = df['curvepoint']
    y = df['area']
    cv = df['%CV']
    loq = df['loq'].unique().item()
    peptide = df['peptide'].unique().item()

    # make a plot of the curve points and the fit, in both linear and log space
    plt.figure(figsize=(10, 5))

    # create all axes we need
    ax0 = plt.subplot(121)
    ax1 = ax0.twinx()
    ax2 = plt.subplot(122)
    ax3 = ax2.twinx()

    # share the secondary axes
    ax1.get_shared_y_axes().join(ax1, ax3)

    # plot the lefthand scatter and label the two y-axes
    ax0.plot(x, y, 'o', color='b')
    ax1.plot(x, cv, '*', label='_nolegend_', color='r')

    # plot the righthand scatter and label the two y-axes
    ax2.semilogx(x, y, 'o', color='b')
    ax3.semilogx(x, cv, '*', label='_nolegend_', color='r')

    # label each figure's axes and give them titles
    ax0.set_title(peptide)
    ax0.set_xlabel("curve point")
    ax0.set_ylabel("area")
    ax1.set_ylabel("% CV")
    ax2.set_title(peptide)
    ax2.set_xlabel("curve point (log10)")
    ax2.set_ylabel("area")
    ax3.set_ylabel("% CV")

    # add horizontal lines marking the 20% CV level
    ax1.axhline(y=20.0, color='r', alpha=0.5, ls='dashed')
    ax3.axhline(y=20.0, color='r', alpha=0.5, ls='dashed')

    # add vertical lines marking the LOQ
    ax1.axvline(x=loq, color='c', label=('LOQ= %.3e' % loq))
    ax3.axvline(x=loq, color='c', label=('LOQ= %.3e' % loq))

    # force axis ticks to be scientific notation so the plot is prettier
    ax0.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    # add legend with LOD and LOQ values
    legend = plt.legend(loc=9, bbox_to_anchor=(0, -0.21, 1., .102), ncol=2)

    plt.tight_layout()

    # save the figure
    plt.savefig(os.path.join(output_dir, peptide + '.png'),
                bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.close()

def make_plot(df):

    x = df['curvepoint']
    y = df['area']
    cv = df['%CV']
    LOQ = df['loq'].unique().item()
    peptide = df['peptide'].unique().item()

    plt.figure(figsize=(5, 7))
    plt.suptitle(peptide, fontsize="large")


    ###
    ### top plot: linear scale x axis
    plt.subplot(2, 1, 1)
    plt.plot(x, y, 'o')  # scatterplot of the data

    plt.axvline(x=LOQ,
                color='c',
                label=('LOQ = %.3e' % LOQ))

    plt.ylabel("signal")

    # force axis ticks to be scientific notation so the plot is prettier
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))

    plt.xlim(xmin=min(x)-max(x)*0.01)  # anchor x and y to 0-ish
    plt.ylim(ymin=min(y)-max(y)*0.01, ymax=(max(y))*1.05)


    ###
    ### bottom plot: CVs for each point
    plt.subplot(2, 1, 2)
    plt.plot(x, cv, marker='x', color='k', label='_nolegend_', linestyle='None')

    plt.axvline(x=LOQ,
                color='c',
                label=('LOQ = %.3e' % LOQ))

    # add 20%CV reference line
    plt.axhline(y=0.20, color='r', linestyle='dashed')

    #plt.title(peptide, y=1.08)
    plt.xlabel("curve point")
    plt.ylabel("CV")

    # force y axis ticks to be scientific notation so the plot is prettier (x is already semilog)

    plt.xlim(xmin=min(x)-max(x)*0.01)  # anchor x and y to 0-ish.
    if len(cv) > 0:
        plt.ylim(ymin=-0.01,
                 ymax=(max(cv)*1.05))

    # save the figure
    # add legend with LOD and LOQ values
    legend = plt.legend(loc=8, bbox_to_anchor=(0, -0.5, 1., .102), ncol=2)
    plt.savefig(os.path.join(output_dir, peptide + '.png'),
                bbox_extra_artists=(legend,),
                bbox_inches='tight', pad_inches=0.5)
    #plt.show()
    plt.close()



if __name__ == "__main__":

    # usage statement and input descriptions
    parser = argparse.ArgumentParser(
        description="A coefficient-of-variation model for fitting calibration curve data. Takes calibration curve \
                    measurements as input, and returns the Limit of Quantitation (LOQ) for each peptide measured in the \
                    calibration curve.",
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
    plot_or_nah = args.plot

    # read in the data
    quant_df_melted = read_input(raw_file, col_conc_map_file)

    # associate multiplier with the curvepoint ratio (if there is a multiplier provided)
    if multiplier_file:
        quant_df_melted = associate_multiplier(quant_df_melted, multiplier_file)

    # run the code I guess
    loq_byCV_df = calculate_LOQ_byCV(quant_df_melted)

    # plot if requested
    if plot_or_nah == True:
        sys.stderr.write("Building plots.\n")
        for peptide in tqdm(loq_byCV_df.peptide.unique()):
            make_plot(loq_byCV_df.loc[(loq_byCV_df['peptide'] == peptide)])
        sys.stdout.write("Plots built.\n")
