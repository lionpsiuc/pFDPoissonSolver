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
                 int nbrdown) {
  int rank;
  int ny;
  int lnx, lny;
  lnx = col_e - col_s + 1;
  lny = row_e - row_s + 1;
  MPI_Comm_rank(comm, &rank);
  ny = nx;
  MPI_Sendrecv(&x[col_e][row_s], lny, MPI_DOUBLE, nbrright, 0,
               &x[col_s - 1][row_s], lny, MPI_DOUBLE, nbrleft, 0, comm,
               MPI_STATUS_IGNORE);
  MPI_Sendrecv(&x[col_s][row_s], lny, MPI_DOUBLE, nbrleft, 1,
               &x[col_e + 1][row_s], lny, MPI_DOUBLE, nbrright, 1, comm,
               MPI_STATUS_IGNORE);
  MPI_Datatype column_type;
  MPI_Type_vector(lnx, 1, maxn, MPI_DOUBLE, &column_type);
  MPI_Type_commit(&column_type);
  MPI_Sendrecv(&x[col_s][row_e], 1, column_type, nbrup, 2, &x[col_s][row_s - 1],
               1, column_type, nbrdown, 2, comm, MPI_STATUS_IGNORE);
  MPI_Sendrecv(&x[col_s][row_s], 1, column_type, nbrdown, 3,
               &x[col_s][row_e + 1], 1, column_type, nbrup, 3, comm,
               MPI_STATUS_IGNORE);
  MPI_Type_free(&column_type);
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
  double sum;
  double tmp;
  int    i, j;
  sum = 0.0;
  for (i = col_s; i <= col_e; i++) {
    for (j = row_s; j <= row_e; j++) {
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
  double h;
  int    i, j;
  h = 1.0 / ((double) (nx + 1));
  for (i = col_s; i <= col_e; i++) {
    for (j = row_s; j <= row_e; j++) {
      b[i][j] = 0.25 * (a[i - 1][j] + a[i + 1][j] + a[i][j + 1] + a[i][j - 1] -
                        h * h * f[i][j]);
    }
  }
}
