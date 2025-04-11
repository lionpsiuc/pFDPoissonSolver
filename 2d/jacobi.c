/**
 * @file jacobi.c
 *
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "jacobi.h"
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
void exchang2d_1(double x[][maxn], int nx, int row_s, int row_e, int col_s,
                 int col_e, MPI_Comm comm, int nbrleft, int nbrright, int nbrup,
                 int nbrdown, MPI_Datatype row_type) {
  int lny = row_e - row_s + 1; // Explain how this works

  // Exchange in horizontal direction (i.e., left to right); these are
  // contiguous in memory
  MPI_Sendrecv(&x[col_e][row_s], lny, MPI_DOUBLE, nbrright, 0,
               &x[col_s - 1][row_s], lny, MPI_DOUBLE, nbrleft, 0, comm,
               MPI_STATUS_IGNORE); // Explain how this works
  MPI_Sendrecv(&x[col_s][row_s], lny, MPI_DOUBLE, nbrleft, 1,
               &x[col_e + 1][row_s], lny, MPI_DOUBLE, nbrright, 1, comm,
               MPI_STATUS_IGNORE); // Explain how this works

  // Exchange in vertical direction (i.e., up to down); these are
  // non-contiguous in memory
  MPI_Sendrecv(&x[col_s][row_e], 1, row_type, nbrup, 2, &x[col_s][row_s - 1], 1,
               row_type, nbrdown, 2, comm,
               MPI_STATUS_IGNORE); // Explain how this works
  MPI_Sendrecv(&x[col_s][row_s], 1, row_type, nbrdown, 3, &x[col_s][row_e + 1],
               1, row_type, nbrup, 3, comm,
               MPI_STATUS_IGNORE); // Explain how this works
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
void exchang2d_nb(double x[][maxn], int nx, int row_s, int row_e, int col_s,
                  int col_e, MPI_Comm comm, int nbrleft, int nbrright,
                  int nbrup, int nbrdown, MPI_Datatype row_type) {
  int         lny = row_e - row_s + 1; // Explain how this works
  MPI_Request reqs[8];                 // Explain how this works

  // Left boundary column, which is contiguous
  MPI_Irecv(&x[col_s - 1][row_s], lny, MPI_DOUBLE, nbrleft, 0, comm,
            &reqs[0]); // Explain how this works
  // Right boundary column, which is contiguous
  MPI_Irecv(&x[col_e + 1][row_s], lny, MPI_DOUBLE, nbrright, 1, comm,
            &reqs[1]); // Explain how this works
  // Bottom boundary row, which is non-contiguous and thus, is using row_type
  MPI_Irecv(&x[col_s][row_s - 1], 1, row_type, nbrdown, 2, comm,
            &reqs[2]); // Explain how this works
  // Top boundary row, which is non-contiguous and thus, is using row_type
  MPI_Irecv(&x[col_s][row_e + 1], 1, row_type, nbrup, 3, comm,
            &reqs[3]); // Explain how this works

  // Send rightmost column to right neighbour
  MPI_Isend(&x[col_e][row_s], lny, MPI_DOUBLE, nbrright, 0, comm,
            &reqs[4]); // Explain how this works
  // Send leftmost column to left neighbour
  MPI_Isend(&x[col_s][row_s], lny, MPI_DOUBLE, nbrleft, 1, comm,
            &reqs[5]); // Explain how this works
  // Send topmost row to top neighbour - non-contiguous
  MPI_Isend(&x[col_s][row_e], 1, row_type, nbrup, 2, comm,
            &reqs[6]); // Explain how this works
  // Send bottommost row to bottom neighbour - non-contiguous
  MPI_Isend(&x[col_s][row_s], 1, row_type, nbrdown, 3, comm,
            &reqs[7]); // Explain how this works

  // Wait for all communications to complete
  MPI_Waitall(8, reqs, MPI_STATUSES_IGNORE);
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
double griddiff2d(double a[][maxn], double b[][maxn], int nx, int row_s,
                  int row_e, int col_s, int col_e) {
  double sum = 0.0;
  double tmp;
  for (int i = col_s; i <= col_e; i++) {
    for (int j = row_s; j <= row_e; j++) {
      tmp = (a[i][j] - b[i][j]);
      sum = sum + tmp * tmp;
    }
  }
  return sum;
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
void sweep2d(double a[][maxn], double f[][maxn], int nx, int row_s, int row_e,
             int col_s, int col_e, double b[][maxn]) {
  double h = 1.0 / ((double) (nx + 1)); // Grid spacing
  for (int i = col_s; i <= col_e; i++) {
    for (int j = row_s; j <= row_e; j++) {
      b[i][j] = 0.25 * (a[i - 1][j] + a[i + 1][j] + a[i][j + 1] + a[i][j - 1] -
                        h * h * f[i][j]);
    }
  }
}
