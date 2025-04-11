#ifndef AUXFUNCS_H
#define AUXFUNCS_H

#include <mpi.h>
#include "poisson1d.h"

void init_full_grids(double a[][maxn], double b[][maxn], double f[][maxn]);
void print_full_grid(double x[][maxn]);
void print_in_order(double x[][maxn], MPI_Comm comm);
void print_grid_to_file(char *fname, double x[][maxn], int nx, int ny);

#endif /* AUXFUNCS_H */