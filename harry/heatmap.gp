# Set output file format to PNG image
set terminal pngcairo enhanced
set output 'heatmap_1d_31.png'

# Set the title of the plot
set title 'Solution Heatmap of Discretized Poisson Equation (1D Domain Decomposition)'

# Set labels for axes, if desired
set xlabel 'x'
set ylabel 'y'

# Define the plot to use a palette of colors representing the data values
set palette rgbformulae 22,13,-31

# Tell Gnuplot how to interpret the data
# assuming the data is in a matrix format
set pm3d map

# Optionally, adjust the plot range as needed
# set cbrange [0:10]
# set xrange [-0.5:2.5]
# set yrange [-0.5:2.5]
set yrange [0:31] reverse
set xrange[0:31]
# Plot the data
splot 'global_1d_31_0.txt' matrix with image
splot 'analytical_0.txt' matrix with image
