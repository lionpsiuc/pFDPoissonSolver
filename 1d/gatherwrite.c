/**
 * @file gatherwrite.c
 *
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "poisson1d.h"

/**
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 *
 * @param[in/out/in,out] param Explain briefly.
 *
 * @return Explain briefly.
 */
void GatherGrid(double global_grid[][maxn], double a[][maxn], int s, int e,
                int nx, int ny, int myid, int nprocs, int* s_vals, int* e_vals,
                MPI_Comm comm) {
  if (myid == 0) {

    // Initialise the global grid first
    for (int i = 0; i < maxn; i++) {
      for (int j = 0; j < maxn; j++) {
        global_grid[i][j] = 0.0;
      }
    }

    // Copy local data from root process
    for (int i = s; i <= e; i++) {
      for (int j = 0; j <= ny + 1; j++) {
        global_grid[i][j] = a[i][j];
      }
    }

    double h;
    h = 1.0 / ((double) (nx + 1)); // Grid spacing

    // Set the top boundary where u(x,1)=1/((1+x)^2+1)
    for (int i = 0; i <= nx + 1; i++) {
      double x               = i * h;
      global_grid[i][ny + 1] = 1.0 / ((1.0 + x) * (1.0 + x) + 1.0);
    }

    // Set the left boundary where u(0,y)=y/(1+y^2)
    for (int j = 0; j <= ny + 1; j++) {
      double y = j * h;
      if (j == 0 || (1.0 + y * y) == 0.0) { // Protect against division by zero
        global_grid[0][j] = 0.0;
      } else {
        global_grid[0][j] = y / (1.0 + y * y);
      }
    }

    // Set the right boundary where u(1,y)=y/(4+y^2)
    for (int j = 0; j <= ny + 1; j++) {
      double y = j * h;
      if (j == 0 || (4.0 + y * y) == 0.0) { // Protect against division by zero
        global_grid[nx + 1][j] = 0.0;
      } else {
        global_grid[nx + 1][j] = y / (4.0 + y * y);
      }
    }
  }

  // Receive data into root process from other processes
  if (myid != 0) {
    for (int col = s; col <= e; col++) { // Send all local columns
      MPI_Send(&a[col][0], ny + 2, MPI_DOUBLE, 0, col, comm);
    }
  } else { // Root process receives from all other processes
    MPI_Status status;
    for (int p = 1; p < nprocs; p++) {
      for (int col = s_vals[p]; col <= e_vals[p]; col++) {
        MPI_Recv(&global_grid[col][0], ny + 2, MPI_DOUBLE, p, col, comm,
                 &status);
      }
    }
  }
}

/**
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 *
 * @param[in/out/in,out] param Explain briefly.
 *
 * @return Explain briefly.
 */
void write_grid(double a[][maxn], int nx, int ny, int rank, int s, int e,
                const char* filename, int write_to_stdout) {
  char full_filename[256];
  sprintf(full_filename, "%s.txt", filename);
  FILE* file = fopen(full_filename, "w");
  if (!file) {
    fprintf(stderr, "Error opening file with name %s\n", full_filename);
    return;
  }

  // Write to file in mesh/grid format; note that each row is a y-coordinate
  // whereas each column is an x-coordinate
  for (int j = ny + 1; j >= 0; j--) {
    for (int i = s; i <= e; i++) {
      fprintf(file, "%2.6lf ", a[i][j]);
    }
    fprintf(file, "\n");
  }
  fclose(file);

  // As per the assignment instructions, add an option to optionally write to
  // stdout
  if (write_to_stdout) {
    printf("Grid for process %d\n", rank);
    for (int j = ny + 1; j >= 0; j--) {
      for (int i = s; i <= e; i++) {
        printf("%2.6lf ", a[i][j]);
      }
      printf("\n");
    }
  }
}
