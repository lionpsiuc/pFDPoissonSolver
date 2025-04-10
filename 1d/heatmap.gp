# Set output to PNG with enhanced text features
set terminal pngcairo enhanced size 1600,600 font 'Arial,12'
set output 'heatmap.png'

# Enable multiplot to place two heatmaps side-by-side
set multiplot layout 1,2 title 'Analytical vs. Numerical Solution for the Poisson Equation' font 'Arial,16'

# Common settings for all plots
set palette rgbformulae 22,13,-31
set pm3d map
set yrange [0:31] reverse
set xrange[0:31]

# First plot - Numerical Solution
set title 'Numerical Solution'
set xlabel 'x'
set ylabel 'y'
set colorbox
splot 'global1dnx31.txt' matrix with image

# Second plot - Analytical Solution
set title 'Analytical Solution'
set xlabel 'x'
set ylabel 'y'
set colorbox
splot 'analytical.txt' matrix with image

# End multiplot
unset multiplot
