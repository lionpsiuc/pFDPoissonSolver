#include "gather.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "decomp1d.h"
#include "poisson1d.h"

void GatherGrid(double global_grid[][maxn], double local_grid[][maxn], int nx,
                int s, int e, MPI_Comm comm) {
  int        rank, size;
  int*       recvcounts = NULL;
  int*       displs     = NULL;
  MPI_Status status;
  int        i, j, proc;

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  // Only rank 0 needs these arrays for gathering
  if (rank == 0) {
    recvcounts = (int*) malloc(size * sizeof(int));
    displs     = (int*) malloc(size * sizeof(int));
    if (!recvcounts || !displs) {
      fprintf(stderr, "Memory allocation error in GatherGrid\n");
      MPI_Abort(comm, 1);
    }
  }

  // Get the local grid sizes for each process
  int local_size = e - s + 1;

  // Gather local sizes to rank 0
  MPI_Gather(&local_size, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, comm);

  // Compute displacements for gathering
  if (rank == 0) {
    displs[0] = 0;
    for (i = 1; i < size; i++) {
      displs[i] = displs[i - 1] + recvcounts[i - 1];
    }
  }

  // Each process sends its local grid columns to rank 0 one by one
  for (i = s; i <= e; i++) {
    // Send column i of local_grid to rank 0
    if (rank != 0) {
      MPI_Send(&local_grid[i][0], nx + 2, MPI_DOUBLE, 0, i, comm);
    } else {
      // For rank 0, copy its own data directly
      for (j = 0; j <= nx + 1; j++) {
        global_grid[i][j] = local_grid[i][j];
      }
    }
  }

  // Rank 0 receives data from other processes
  if (rank == 0) {
    // For each process
    for (proc = 1; proc < size; proc++) {
      int start, end;
      // Get start and end for this process using MPE_Decomp1d
      MPE_Decomp1d(nx, size, proc, &start, &end);

      // Receive columns from this process
      for (i = start; i <= end; i++) {
        MPI_Recv(&global_grid[i][0], nx + 2, MPI_DOUBLE, proc, i, comm,
                 &status);
      }
    }
  }

  // Free allocated memory
  if (rank == 0) {
    free(recvcounts);
    free(displs);
  }
}
