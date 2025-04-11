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

#include "poisson2d.h"

/**
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 *
 * @param[in/out/in,out] param Explain briefly.
 *
 * @return Explain briefly.
 */
void GatherGrid2D(double global_grid[maxn][maxn], double a[maxn][maxn],
                  int row_s, int row_e, int col_s, int col_e, int nx, int ny,
                  int myid, int nprocs, int* row_s_vals, int* row_e_vals,
                  int* col_s_vals, int* col_e_vals) {
  int lny;
  lny = row_e - row_s + 1;
  if (myid == 0) {
    double h = 1.0 / ((double) (nx + 1)); // Grid spacing
    for (int i = col_s - 1; i <= col_e; i++) {
      for (int j = row_s; j < row_e + 1; j++) {
        global_grid[i][j] = a[i][j];
      }
    }
    for (int i = col_s; i < maxn - 1; i++) {
      double x               = i * h;
      global_grid[i][ny + 1] = 1.0 / ((1 + x) * (1 + x) + 1);
    }
    for (int i = col_s; i < maxn - 1; i++) {
      global_grid[i][0] = 0.0;
    }
    for (int j = maxn - 1; j >= 1; j--) {
      double y          = j * h;
      global_grid[0][j] = y / (1 + y * y);
    }
    for (int j = maxn - 1; j >= 1; j--) {
      double y               = j * h;
      global_grid[nx + 1][j] = y / (4 + y * y);
    }
  }
  if (myid != 0) {
    for (int p = col_s; p <= col_e; p++)
      MPI_Send(&a[p][row_s], lny, MPI_DOUBLE, 0, p + myid, MPI_COMM_WORLD);
  }
  if (myid == 0) {
    MPI_Status status;
    for (int proc = 1; proc < nprocs; proc++) {
      for (int col = col_s_vals[proc]; col <= col_e_vals[proc]; col++) {
        MPI_Recv(&global_grid[col][row_s_vals[proc]], lny, MPI_DOUBLE, proc,
                 proc + col, MPI_COMM_WORLD, &status);
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
void write_grid(char* filename, double a[][maxn], int nx, int row_s, int row_e,
                int col_s, int col_e) {
  char full_filename[256];
  sprintf(full_filename, "%s.txt", filename);
  FILE* file = fopen(full_filename, "w");
  if (!file) {
    fprintf(stderr, "Error opening file with name %s\n", full_filename);
    return;
  }
  for (int j = row_e + 1; j >= row_s - 1; j--) {
    for (int i = col_s - 1; i <= col_e + 1; i++) {
      fprintf(file, "%2.6lf ", a[i][j]);
    }
    fprintf(file, "\n");
  }
}
