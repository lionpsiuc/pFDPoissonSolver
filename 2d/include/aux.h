/**
 * @file aux.h
 *
 * @brief Utility functions for grid initialisation in the 2D parallel Poisson
 *        solver.
 */

#include "poisson2d.h"

/**
 * @brief Initialises grid arrays with a default value.
 *
 * Sets all elements in the grid arrays to an initial junk value.
 *
 * @param[out] a Grid array for current solution iteration.
 * @param[out] b Grid array for next solution iteration.
 * @param[out] f Grid array for right-hand side function values.
 */
void init_full_grids(double a[][maxn], double b[][maxn], double f[][maxn]);

/**
 * @brief Initialises the local grid portion with boundary conditions.
 *
 * Sets up the local portion of the grid assigned to a process, including ghost
 * cells and boundary conditions. Interior points are set to zero. Sets the
 * appropriate Dirichlet boundary conditions for the Poisson problem.
 *
 * @param[out] a Grid array for current solution iteration.
 * @param[out] b Grid array for next solution iteration.
 * @param[out] f Grid array for right-hand side function values.
 * @param[in] nx Number of interior grid points in x-axis.
 * @param[in] ny Number of interior grid points in y-axis.
 * @param[in] row_s Starting row index of local domain.
 * @param[in] row_e Ending row index of local domain.
 * @param[in] col_s Starting column index of local domain.
 * @param[in] col_e Ending column index of local domain.
 */
void init_twod(double a[][maxn], double b[][maxn], double f[][maxn], int nx,
               int ny __attribute__((unused)), int row_s, int row_e, int col_s,
               int col_e);
