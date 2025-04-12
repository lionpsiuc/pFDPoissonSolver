# Set output to PNG with enhanced text features
set terminal pngcairo enhanced size 1600,600 font 'Arial,12'
set output 'heatmapnprocs4nx31.png'

# Enable multiplot to place two heatmaps
set multiplot layout 1,2 title 'Analytical vs. Numerical Solution for the Poisson Equation Using 2D Decomposition' font 'Arial,16'

# Common settings for all plots
set palette rgbformulae 22,13,-31
set pm3d map
set yrange [0:30] reverse
set xrange [0:30]

# First plot
set title 'Numerical Solution'
set xlabel 'x'
set ylabel 'y'
set colorbox
splot 'global2dnprocs4nx31.txt' matrix with image

# Second plot
set title 'Analytical Solution'
set xlabel 'x'
set ylabel 'y'
set colorbox
splot 'analyticalnprocs4nx31.txt' matrix with image

# End multiplot
unset multiplot
