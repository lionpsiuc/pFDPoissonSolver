# Poisson Equation Solver

This project provides a numerical solution to the discretized Poisson equation using 1D and 2D domain decomposition. It includes implementations in C that utilize the Jacobi method for iterative solution.

## Prerequisites

- OpenMPI


## Environment Setup

On chuck, load the necessary OpenMPI module with:
module load openmpi/3.1.5-gnu9.2.0

On seagull, the following equivalent module is available:
module load openmpi-3.1.6-gcc-9.3.0

## Compilation
The files are compiled by running 'make' in the terminal.

## File Structure
main_1d.c - Main program for 1D domain decomposition.
main_2d.c - Main program for 2D domain decomposition.
jacobi_1d.c, jacobi_1d.h - Jacobi solver for 1D decomposition.
jacobi_2d.c, jacobi_2d.h - Jacobi solver for 2D decomposition.
Makefile - Makefile for building the programs.
poisson.sh - Bash script for submitting batch jobs on seagull.
heatmap.gp - Gnuplot script for plotting the results.

And various PNG files of the plots created for the report.
Note files of the form 'output_Xd_Y.txt' are the local grids for each processor.
Files of the form 'global_Xd_0.txt' are the global grid used for generating the plots.
These txt files are generated when running the programs.

## Seagull Usage
To run the program on seagull, use the following command:
sbatch poisson.sh

## Plotting
To make use of gnuplot script, gnuplot must be installed, and then the following command can be used in the terminal:
gnuplot heatmap.gp

## Acknowledgement
The seagull and chuck clusters used for this program are managed and maintained by Research IT.
Information is available at the link below:

https://www.tchpc.tcd.ie/resources/acknowledgementpolicy
