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

#include "../include/jacobi.h"
#include "../include/poisson1d.h"

/**
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 *
 * @param[in/out/in,out] param Explain briefly.
 *
 * @return Explain briefly.
 */
void exchang1(double x[][maxn], int nx, int s, int e, MPI_Comm comm,
              int nbrleft, int nbrright) {
  int rank;
  int ny;
  MPI_Comm_rank(comm, &rank);
  ny = nx;
  MPI_Ssend(&x[e][1], ny, MPI_DOUBLE, nbrright, 0, comm);
  MPI_Recv(&x[s - 1][1], ny, MPI_DOUBLE, nbrleft, 0, comm, MPI_STATUS_IGNORE);
  MPI_Ssend(&x[s][1], ny, MPI_DOUBLE, nbrleft, 1, comm);
  MPI_Recv(&x[e + 1][1], ny, MPI_DOUBLE, nbrright, 1, comm, MPI_STATUS_IGNORE);
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
void exchangi1(double x[][maxn], int nx, int s, int e, MPI_Comm comm,
               int nbrleft, int nbrright) {
  MPI_Request reqs[4];
  MPI_Irecv(&x[s - 1][1], nx, MPI_DOUBLE, nbrleft, 0, comm, &reqs[0]);
  MPI_Irecv(&x[e + 1][1], nx, MPI_DOUBLE, nbrright, 0, comm, &reqs[1]);
  MPI_Isend(&x[e][1], nx, MPI_DOUBLE, nbrright, 0, comm, &reqs[2]);
  MPI_Isend(&x[s][1], nx, MPI_DOUBLE, nbrleft, 0, comm, &reqs[3]);
  MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
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
double griddiff(double a[][maxn], double b[][maxn], int nx, int s, int e) {
  double sum;
  double tmp;
  sum = 0.0;
  for (int i = s; i <= e; i++) {
    for (int j = 1; j < nx + 1; j++) {
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
void sweep1d(double a[][maxn], double f[][maxn], int nx, int s, int e,
             double b[][maxn]) {
  double h = 1.0 / ((double) (nx + 1)); // Grid spacing
  for (int i = s; i <= e; i++) {
    for (int j = 1; j < nx + 1; j++) {
      b[i][j] = 0.25 * (a[i - 1][j] + a[i + 1][j] + a[i][j + 1] + a[i][j - 1] -
                        h * h * f[i][j]);
    }
  }
}
