/**
 * @file gatherwrite.c
 *
 * @brief Implementation of 2D grid gathering and writing utilities.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/poisson2d.h"

/**
 * @brief Gathers distributed 2D grid data from all processes to the root
 *        process.
 *
 * Collects 2D grid sections from all MPI processes and combines them into a
 * complete global grid on the root process.
 *
 * @param[out] global_grid Array to store the complete gathered grid (only used
 * by root process)
 * @param[in] a Local grid array containing this process's portion of the
 * solution
 * @param[in] row_s Starting row index of local domain.
 * @param[in] row_e Ending row index of local domain.
 * @param[in] col_s Starting column index of local domain.
 * @param[in] col_e Ending column index of local domain.
 * @param[in] nx Number of interior grid points in x-axis.
 * @param[in] ny Number of interior grid points in y-axis.
 * @param[in] myid Rank of the current MPI process.
 * @param[in] nprocs Total number of MPI processes.
 * @param[in] row_s_vals Array containing start row indices for all processes.
 * @param[in] row_e_vals Array containing end row indices for all processes.
 * @param[in] col_s_vals Array containing start column indices for all
 *                       processes.
 * @param[in] col_e_vals Array containing end column indices for all processes.
 * @param[in] comm MPI communicator.
 */
void GatherGrid2D(double global_grid[][maxn], double a[][maxn], int row_s,
                  int row_e, int col_s, int col_e, int nx, int ny, int myid,
                  int nprocs, int* row_s_vals, int* row_e_vals, int* col_s_vals,
                  int* col_e_vals, MPI_Comm comm) {
  if (myid == 0) {

    // Initialise the global grid first
    for (int i = 0; i < maxn; i++) {
      for (int j = 0; j < maxn; j++) {
        global_grid[i][j] = 0.0;
      }
    }

    // Copy local data from root process
    for (int i = col_s; i <= col_e; i++) {
      for (int j = row_s; j <= row_e; j++) {
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
    int local_rows = row_e - row_s + 1;
    for (int col = col_s; col <= col_e; col++) {
      int tag = 10000 + myid * 100 + col; // Use a unique tag
      MPI_Send(&a[col][row_s], local_rows, MPI_DOUBLE, 0, tag, comm);
    }
  } else { // Root process receives from all other processes
    for (int p = 1; p < nprocs; p++) {
      int p_row_s      = row_s_vals[p];
      int p_row_e      = row_e_vals[p];
      int p_col_s      = col_s_vals[p];
      int p_col_e      = col_e_vals[p];
      int p_local_rows = p_row_e - p_row_s + 1;
      for (int col = p_col_s; col <= p_col_e; col++) {
        MPI_Status status;
        int        tag = 10000 + p * 100 + col; // Use a unique tag
        MPI_Recv(&global_grid[col][p_row_s], p_local_rows, MPI_DOUBLE, p, tag,
                 comm, &status);
      }
    }

    // Message to state when the function has completed its task
    printf("Gathering complete\n");
  }
}

/**
 * @brief Writes 2D grid data to a file or terminal for visualisation.
 *
 * @param[in] filename Base name of the file to write.
 * @param[in] a Grid array containing the data to write.
 * @param[in] nx Number of interior grid points in x-axis.
 * @param[in] ny Number of interior grid points in y-axis.
 * @param[in] rank Rank of the current MPI process.
 * @param[in] row_s Starting row index of the grid portion to write.
 * @param[in] row_e Ending row index of the grid portion to write.
 * @param[in] col_s Starting column index of the grid portion to write.
 * @param[in] col_e Ending column index of the grid portion to write.
 * @param[in] write_to_stdout Flag to control whether to also print grid to
 *                            standard output.
 */
void write_grid(char* filename, double a[][maxn],
                int nx __attribute__((unused)), int ny __attribute__((unused)),
                int rank, int row_s, int row_e, int col_s, int col_e,
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
  for (int j = row_e; j >= row_s; j--) {
    for (int i = col_s; i <= col_e; i++) {
      fprintf(file, "%.6lf ", a[i][j]);
    }
    fprintf(file, "\n");
  }

  fclose(file);

  // Write to terminal if requested
  if (write_to_stdout) {
    printf("Grid for process %d\n", rank);
    for (int j = row_e; j >= row_s; j--) {
      for (int i = col_s; i <= col_e; i++) {
        printf("%.6lf ", a[i][j]);
      }
      printf("\n");
    }
  }
}
