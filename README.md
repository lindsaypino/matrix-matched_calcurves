USAGE.
calculate-loq [options] <quantitative matrix>


DESCRIPTION.
calculate-loq fits a piecewise linear regression to label-free MS proteomics calibration curves. A horizontal “noise regime” is fit to model the background signal from a blank or any concentration point below the LOD; a positive-sloped “linear regime” is fit to model the dynamic range from points with signal above the noise.


INPUT.
quantitative matrix – a csv containing peptides as rows, concentration points as columns, and quantifications as values.


OUTPUT.
The program writes files to the folder loq-output by default. The following files will be created:

peptide-loqs.csv – a file containing the peptides in one column and their calculated LOQ in another. 
*.png – plots of each peptide calibration curve with the fitted piecewise linear regression.

OPTIONS.
Coming soon?

