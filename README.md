USAGE.
calculate-loq [options] <curve_data> <filename_concentration_map>


DESCRIPTION.
calculate-loq fits a piecewise prediction interval model to calibration curve data. A horizontal “noise segment” is fit to model the background signal from a blank or any concentration point below the LOD; a positive-sloped “linear segment” is fit to model the dynamic range from points with signal above the noise. Prediction intervals are drawn around each segment, then used to determine the point at which signal is confidently detected above noise.


INPUT.
curve_data – either the Encyclopedia \*.elib.peptides.txt file or a Skyline \*.csv custom export with peptides as rows, concentration points as columns, and areas as values
filename_concentration_map - a csv containing each filename as a row with its corresponding concentration point in a second column.


OUTPUT.
The program writes files to the folder curvefits-output by default. The following files will be created:

figures_of_merit.csv – a file containing the peptides in one column and their calculated LOQ in another. 
\*.png – plots of each peptide calibration curve with the fitted piecewise linear regression.

OPTIONS.
--multiplier_file - use a single-point multiplier associated with the curve data peptides
--output_path - specify an output path for figures of merit and plots
--plot - yes/no (y/n) to create individual calibration curve plots for each peptide


EXAMPLE.
python bin\calibration_curve_fitting_dataanalysis.py data\cptac_study9pt1_site56a_lightonly_totalarearatio_ssd.csv data\cptac_filename2concentration.csv
