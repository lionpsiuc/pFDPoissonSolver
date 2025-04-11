#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include "gather.h"
#include "poisson1d.h"
#include "decomp1d.h"

void GatherGrid(double global_grid[][maxn], double local_grid[][maxn], 
    int nx, int s, int e, MPI_Comm comm)
{
int rank, size;
MPI_Status status;

MPI_Comm_rank(comm, &rank);
MPI_Comm_size(comm, &size);

// If only one process, just copy the data directly
if (size == 1) {
for (int i = 0; i < maxn; i++) {
for (int j = 0; j < maxn; j++) {
    global_grid[i][j] = local_grid[i][j];
}
}
return;
}

// Process 0 collects data from all processes
if (rank == 0) {
// First, copy own data
for (int i = s; i <= e; i++) {
for (int j = 0; j <= nx+1; j++) {
    global_grid[i][j] = local_grid[i][j];
}
}

// Receive data from other processes
for (int p = 1; p < size; p++) {
int proc_s, proc_e;
MPE_Decomp1d(nx, size, p, &proc_s, &proc_e);

// Receive each row from the process
for (int i = proc_s; i <= proc_e; i++) {
    MPI_Recv(&global_grid[i][0], maxn, MPI_DOUBLE, p, i, comm, &status);
}
}
} else {
// Send data to process 0
for (int i = s; i <= e; i++) {
MPI_Send(&local_grid[i][0], maxn, MPI_DOUBLE, 0, i, comm);
}
}
}