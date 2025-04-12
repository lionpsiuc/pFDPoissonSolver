/**
 * @file jacobi.h
 *
 * @brief Jacobi iteration functions for the 1D parallel Poisson solver.
 */

#include "poisson1d.h"

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
              int nbrleft, int nbrright);

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
               int nbrleft, int nbrright);

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
double griddiff(double a[][maxn], double b[][maxn], int nx, int s, int e);

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
             double b[][maxn]);
