/**
 * @file aux.c
 *
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 */

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "aux.h"
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
void init_full_grid(double g[][maxn]) {
  const double junkval = -5;
  for (int i = 0; i < maxn; i++) {
    for (int j = 0; j < maxn; j++) {
      g[i][j] = junkval;
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
void init_full_grids(double a[][maxn], double b[][maxn], double f[][maxn]) {
  const double junkval = -5;
  for (int i = 0; i < maxn; i++) {
    for (int j = 0; j < maxn; j++) {
      a[i][j] = junkval;
      b[i][j] = junkval;
      f[i][j] = junkval;
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
void init_twod(double a[][maxn], double b[][maxn], double f[][maxn], int nx,
               int ny, int row_s, int row_e, int col_s, int col_e) {
  double h = 1.0 / ((double) (nx + 1)); // Grid spacing

  // Set everything to zero first
  for (int i = col_s - 1; i <= col_e + 1; i++) {
    for (int j = row_s - 1; j <= row_e + 1; j++) {
      a[i][j] = 0.0;
      b[i][j] = 0.0;
      f[i][j] = 0.0;
    }
  }

  if (row_e == nx) {
    for (int i = col_s; i <= col_e; i++) {
      double x     = i * h;
      a[i][nx + 1] = 1.0 / ((1 + x) * (1 + x) + 1);
      b[i][nx + 1] = 1.0 / ((1 + x) * (1 + x) + 1);
    }
  }
  if (row_s == 1) {
    for (int i = col_s; i <= col_e; i++) {
      a[i][0] = 0.0;
      b[i][0] = 0.0;
    }
  }
  if (col_s == 1) {
    for (int j = row_s; j <= row_e; j++) {
      double y = j * h;
      a[0][j]  = y / (1 + y * y);
      b[0][j]  = y / (1 + y * y);
    }
  }
  if (col_e == nx) {
    for (int j = row_s; j <= row_e; j++) {
      double y     = j * h;
      a[nx + 1][j] = y / (4 + y * y);
      b[nx + 1][j] = y / (4 + y * y);
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
void print_full_grid(double x[][maxn]) {
  for (int j = maxn - 1; j >= 0; j--) {
    for (int i = 0; i < maxn; i++) {
      if (x[i][j] < 10000.0) {
        printf("|%2.6lf| ", x[i][j]);
      } else {
        printf("%9.2lf ", x[i][j]);
      }
    }
    printf("\n");
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
void print_grid_to_file(char* fname, double x[][maxn], int nx, int ny) {
  FILE* fp;
  fp = fopen(fname, "w");
  if (!fp) {
    fprintf(stderr, "Can't open file %s\n", fname);
    exit(4);
  }
  for (int j = ny + 1; j >= 0; j--) {
    for (int i = 0; i < nx + 2; i++) {
      fprintf(fp, "%lf ", x[i][j]);
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
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
void print_in_order(double x[][maxn], MPI_Comm comm) {
  int myid, size;
  MPI_Comm_rank(comm, &myid);
  MPI_Comm_size(comm, &size);
  MPI_Barrier(comm);
  printf("Attempting to print in order");
  sleep(1);
  MPI_Barrier(comm);
  for (int i = 0; i < size; i++) {
    if (i == myid) {
      printf("Grid for process %d\n", myid);
      print_full_grid(x);
    }
    fflush(stdout);
    usleep(500);
    MPI_Barrier(
        comm); // I believe this should be comm instead of MPI_COMM_WORLD
  }
}
