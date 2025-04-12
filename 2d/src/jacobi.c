/**
 * @file jacobi.c
 *
 * @brief Implementation of Jacobi iteration functions for the 2D parallel
 *        Poisson solver.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "../include/jacobi.h"
#include "../include/poisson2d.h"

/**
 * @brief Exchanges ghost cells with neighbouring processes using blocking
 *        communication.
 *
 * Performs ghost cell exchange between neighbouring processes using blocking
 * MPI_Sendrecv calls in both horizontal and vertical directions. Uses a custom
 * MPI datatype for exchanging non-contiguous vertical data.
 *
 * @param[in,out] x Grid array to exchange ghost cells for.
 * @param[in] nx Number of interior grid points in x-axis.
 * @param[in] row_s Starting row index of local domain.
 * @param[in] row_e Ending row index of local domain.
 * @param[in] col_s Starting column index of local domain.
 * @param[in] col_e Ending column index of local domain.
 * @param[in] comm MPI communicator.
 * @param[in] nbrleft Rank of the left neighbouring process.
 * @param[in] nbrright Rank of the right neighbouring process.
 * @param[in] nbrup Rank of the upper neighbouring process.
 * @param[in] nbrdown Rank of the lower neighbouring process.
 * @param[in] row_type MPI datatype for exchanging non-contiguous row data.
 */
void exchang2d_1(double x[][maxn], int nx __attribute__((unused)), int row_s,
                 int row_e, int col_s, int col_e, MPI_Comm comm, int nbrleft,
                 int nbrright, int nbrup, int nbrdown, MPI_Datatype row_type) {
  int lny =
      row_e - row_s +
      1; // Calculates the number of rows in the local domain for this process

  // Exchange in horizontal direction (i.e., left to right); these are
  // contiguous in memory
  MPI_Sendrecv(&x[col_e][row_s], lny, MPI_DOUBLE, nbrright, 0,
               &x[col_s - 1][row_s], lny, MPI_DOUBLE, nbrleft, 0, comm,
               MPI_STATUS_IGNORE); // Sends the rightmost column to the right
                                   // neighbour and simultaneously receives the
                                   // left ghost column from the left neighbour
  MPI_Sendrecv(
      &x[col_s][row_s], lny, MPI_DOUBLE, nbrleft, 1, &x[col_e + 1][row_s], lny,
      MPI_DOUBLE, nbrright, 1, comm,
      MPI_STATUS_IGNORE); // Sends the leftmost column to the left
                          // neighbour and simultaneously receives the right
                          // ghost column from the right neighbour

  // Exchange in vertical direction (i.e., up to down); these are
  // non-contiguous in memory
  MPI_Sendrecv(&x[col_s][row_e], 1, row_type, nbrup, 2, &x[col_s][row_s - 1], 1,
               row_type, nbrdown, 2, comm,
               MPI_STATUS_IGNORE); // Sends the topmost row to the top neighbour
                                   // and simultaneously receives the bottom
                                   // ghost row from the bottom neighbour
  MPI_Sendrecv(&x[col_s][row_s], 1, row_type, nbrdown, 3, &x[col_s][row_e + 1],
               1, row_type, nbrup, 3, comm,
               MPI_STATUS_IGNORE); // Sends the bottommost row to the bottom
                                   // neighbour and simultaneously receives the
                                   // top ghost row from the top neighbour
}

/**
 * @brief Exchanges ghost cells with neighbouring processes using non-blocking
 *        communication.
 *
 * Performs ghost cell exchange between neighbouring processes using
 * non-blocking MPI_Isend and MPI_Irecv calls in both horizontal and vertical
 * directions. Uses a custom MPI datatype for exchanging non-contiguous vertical
 * data. This allows for potential overlap of communication and computation,
 * improving performance.
 *
 * @param[in,out] x Grid array to exchange ghost cells for.
 * @param[in] nx er of interior grid points in x-axis.
 * @param[in] row_s Starting row index of local domain.
 * @param[in] row_e Ending row index of local domain.
 * @param[in] col_s Starting column index of local domain.
 * @param[in] col_e Ending column index of local domain.
 * @param[in] comm MPI communicator.
 * @param[in] nbrleft Rank of the left neighbouring process.
 * @param[in] nbrright Rank of the right neighbouring process.
 * @param[in] nbrup Rank of the upper neighbouring process.
 * @param[in] nbrdown Rank of the lower neighbouring process.
 * @param[in] row_type MPI datatype for exchanging non-contiguous row data.
 */
void exchang2d_nb(double x[][maxn], int nx __attribute__((unused)), int row_s,
                  int row_e, int col_s, int col_e, MPI_Comm comm, int nbrleft,
                  int nbrright, int nbrup, int nbrdown, MPI_Datatype row_type) {
  int lny =
      row_e - row_s +
      1; // Calculates the number of rows in the local domain for this process
  MPI_Request reqs[8]; // Array to hold eight MPI request handles

  // Left boundary column, which is contiguous
  MPI_Irecv(&x[col_s - 1][row_s], lny, MPI_DOUBLE, nbrleft, 0, comm,
            &reqs[0]); // Receives the ghost column from the left neighbour into
                       // the column at index col_s - 1

  // Right boundary column, which is contiguous
  MPI_Irecv(&x[col_e + 1][row_s], lny, MPI_DOUBLE, nbrright, 1, comm,
            &reqs[1]); // Receives the ghost column from the right neighbour
                       // into the column at index col_e + 1

  // Bottom boundary row, which is non-contiguous and thus, is using row_type
  MPI_Irecv(&x[col_s][row_s - 1], 1, row_type, nbrdown, 2, comm,
            &reqs[2]); // Receives the ghost row from the bottom neighbour into
                       // the row at index row_s - 1

  // Top boundary row, which is non-contiguous and thus, is using row_type
  MPI_Irecv(&x[col_s][row_e + 1], 1, row_type, nbrup, 3, comm,
            &reqs[3]); // Receives the ghost row from the top neighbour into the
                       // row at index row_e + 1

  // Send rightmost column to right neighbour
  MPI_Isend(&x[col_e][row_s], lny, MPI_DOUBLE, nbrright, 0, comm, &reqs[4]);

  // Send leftmost column to left neighbour
  MPI_Isend(&x[col_s][row_s], lny, MPI_DOUBLE, nbrleft, 1, comm, &reqs[5]);

  // Send topmost row to top neighbour, which is non-contiguous
  MPI_Isend(&x[col_s][row_e], 1, row_type, nbrup, 2, comm, &reqs[6]);

  // Send bottommost row to bottom neighbour, which is non-contiguous
  MPI_Isend(&x[col_s][row_s], 1, row_type, nbrdown, 3, comm, &reqs[7]);

  // Wait for all communications to complete
  MPI_Waitall(8, reqs, MPI_STATUSES_IGNORE);
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
 * @param[in] row_s Starting row index of local domain.
 * @param[in] row_e Ending row index of local domain.
 * @param[in] col_s Starting column index of local domain.
 * @param[in] col_e Ending column index of local domain.
 *
 * @return Sum of squared differences between the two grid arrays.
 */
double griddiff2d(double a[][maxn], double b[][maxn],
                  int nx __attribute__((unused)), int row_s, int row_e,
                  int col_s, int col_e) {
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
 * @brief Performs one Jacobi iteration step.
 *
 * Updates the grid values for one iteration of the Jacobi method. For each
 * point, computes the average of its four neighbours, adjusted by the
 * right-hand side function values, to solve the Poisson equation.
 *
 * @param[in] a Current iteration grid array.
 * @param[in] f Right-hand side function values.
 * @param[in] nx Number of interior grid points in x-axis.
 * @param[in] row_s Starting row index of local domain.
 * @param[in] row_e Ending row index of local domain.
 * @param[in] col_s Starting column index of local domain.
 * @param[in] col_e Ending column index of local domain.
 * @param[out] b Next iteration grid array to store the updated values.
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
