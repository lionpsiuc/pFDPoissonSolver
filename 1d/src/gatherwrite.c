/**
 * @file gatherwrite.c
 *
 * @brief Implementation of grid gathering and writing utilities.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/poisson1d.h"

/**
 * @brief Gathers distributed grid data from all processes to the root process.
 *
 * Collects grid sections from all MPI processes and combines them into a
 * complete global grid on the root process.
 *
 * @param[out] global_grid Array to store the complete gathered grid (only used
 * by root process)
 * @param[in] a Local grid array containing this process's portion of the
 * solution
 * @param[in] s Starting column index of local domain
 * @param[in] e Ending column index of local domain
 * @param[in] nx Number of interior grid points in x-axis.
 * @param[in] ny Number of interior grid points in y-axis.
 * @param[in] myid Rank of the current MPI process.
 * @param[in] nprocs Total number of MPI processes.
 * @param[in] s_vals Array containing start indices for all processes.
 * @param[in] e_vals Array containing end indices for all processes.
 * @param[in] comm MPI communicator.
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

    double h = 1.0 / ((double) (nx + 1)); // Grid spacing

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

  // Synchronise before data exchange
  MPI_Barrier(comm);

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

    // Message to state when the function has completed its task
    printf("Gathering complete\n");
  }
}

/**
 * @brief Writes grid data to a file or terminal for visualisation.
 *
 * @param[in] filename Base name of the file to write.
 * @param[in] a Grid array containing the data to write.
 * @param[in] nx Number of interior grid points in x-axis.
 * @param[in] ny Number of interior grid points in y-axis.
 * @param[in] rank Rank of the current MPI process.
 * @param[in] s Starting index of the grid portion to write.
 * @param[in] e Ending index of the grid portion to write.
 * @param[in] write_to_stdout Flag to control whether to also print grid to
 *                            standard output.
 */
void write_grid(char* filename, double a[][maxn],
                int nx __attribute__((unused)), int ny, int rank, int s, int e,
                int write_to_stdout) {

  // Create filename with extension
  char full_filename[256];
  sprintf(full_filename, "%s.txt", filename);

  // Open file for writing
  FILE* file = fopen(full_filename, "w");
  if (!file) {
    fprintf(stderr, "Error opening file %s for writing\n", full_filename);
    return;
  }

  // Write to file in mesh/grid format; note that each row is a y-coordinate
  // whereas each column is an x-coordinate
  for (int j = e; j >= s; j--) {
    for (int i = s; i <= e; i++) {
      fprintf(file, "%.6lf ", a[i][j]);
    }
    fprintf(file, "\n");
  }

  fclose(file);

  // Write to terminal if requested
  if (write_to_stdout) {
    printf("Grid for process %d\n", rank);
    for (int j = ny + 1; j >= 0; j--) {
      for (int i = s; i <= e; i++) {
        printf("%.6lf ", a[i][j]);
      }
      printf("\n");
    }
  }
}
