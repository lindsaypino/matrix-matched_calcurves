USAGE.
calculate-loq [options] <quantitative matrix>


DESCRIPTION.
calculate-loq fits a piecewise prediction interval model to calibration curve data. A horizontal “noise segment” is fit to model the background signal from a blank or any concentration point below the LOD; a positive-sloped “linear segment” is fit to model the dynamic range from points with signal above the noise. Prediction intervals are drawn around each segment, then used to determine the point at which signal is confidently detected above noise.


INPUT.
quantitative matrix – either the Encyclopedia \*.elib.peptides.txt file or a Skyline \*.csv custom export with peptides as rows, concentration points as columns, and areas as values
filename to peptide map - a csv containing each filename as a row with its corresponding concentration point in a second column.


OUTPUT.
The program writes files to the folder curvefits-output by default. The following files will be created:

peptide-loqs.csv – a file containing the peptides in one column and their calculated LOQ in another. 
\*.png – plots of each peptide calibration curve with the fitted piecewise linear regression.

OPTIONS.
Coming soon?

