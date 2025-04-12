/**
 * @file jacobi.c
 *
 * @brief Implementation of Jacobi iteration functions for the 1D parallel
 *        Poisson solver.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "../include/jacobi.h"
#include "../include/poisson1d.h"

/**
 * @brief Exchanges ghost cells with neighbouring processes using blocking
 *        communication.
 *
 * Performs ghost cell exchange between neighbouring processes using blocking
 * MPI_Ssend and MPI_Recv calls. This function ensures that each process has
 * up-to-date boundary values from its neighbours before performing
 * computations.
 *
 * @param[in,out] x Grid array to exchange ghost cells for.
 * @param[in] nx Number of interior grid points in x-axis.
 * @param[in] s Starting column index of local domain.
 * @param[in] e Ending column index of local domain.
 * @param[in] comm MPI communicator.
 * @param[in] nbrleft Rank of the left neighbouring process.
 * @param[in] nbrright Rank of the right neighbouring process.
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
 * @brief Exchanges ghost cells with neighbouring processes using non-blocking
 *        communication.
 *
 * Performs ghost cell exchange between neighbouring processes using
 * non-blocking MPI_Isend and MPI_Irecv calls. This allows for potential overlap
 * of communication and computation, improving performance.
 *
 * @param[in,out] x Grid array to exchange ghost cells for.
 * @param[in] nx Number of interior grid points in x-axis.
 * @param[in] s Starting column index of local domain.
 * @param[in] e Ending column index of local domain.
 * @param[in] comm MPI communicator.
 * @param[in] nbrleft Rank of the left neighbouring process.
 * @param[in] nbrright Rank of the right neighbouring process.
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
 * @brief Calculates the squared difference between two grid arrays.
 *
 * Computes the sum of squared differences between two grid arrays, which is
 * used to check for convergence between iterations of the Jacobi method.
 *
 * @param[in] a First grid array.
 * @param[in] b Second grid array.
 * @param[in] nx Number of interior grid points in x-axis.
 * @param[in] s Starting column index of local domain.
 * @param[in] e Ending column index of local domain.
 *
 * @return Sum of squared differences between the two grid arrays.
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
 * @brief Performs one Jacobi iteration step.
 *
 * Updates the grid values for one iteration of the Jacobi method. For each
 * point, computes the average of its four neighbours, adjusted by the
 * right-hand side function values, to solve the Poisson equation.
 *
 * @param[in] a Current iteration grid array.
 * @param[in] f Right-hand side function values.
 * @param[in] nx Number of interior grid points in x-axis.
 * @param[in] s Starting column index of local domain.
 * @param[in] e Ending column index of local domain.
 * @param[out] b Next iteration grid array to store the updated values.
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
