/**
 * @file jacobi.h
 *
 * @brief Jacobi iteration functions for the 2D parallel Poisson solver.
 */

#include "poisson2d.h"

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
                 int nbrright, int nbrup, int nbrdown, MPI_Datatype row_type);

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
                  int nbrright, int nbrup, int nbrdown, MPI_Datatype row_type);

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
                  int col_s, int col_e);

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
             int col_s, int col_e, double b[][maxn]);
