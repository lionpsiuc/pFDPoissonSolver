#ifndef GATHER_H
#define GATHER_H

#include <mpi.h>
#include "poisson1d.h"

void GatherGrid(double global_grid[][maxn], double local_grid[][maxn], 
                int nx, int s, int e, MPI_Comm comm);

#endif /* GATHER_H */