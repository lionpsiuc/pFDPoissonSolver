/**
 * @file aux.h
 *
 * @brief Utility functions for grid initialisation in the 1D parallel Poisson
 *        solver.
 */

#include "poisson1d.h"

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
 * @param[in] s Starting column index of local domain.
 * @param[in] e Ending column index of local domain.
 */
void init_oned(double a[][maxn], double b[][maxn], double f[][maxn], int nx,
               int ny, int s, int e);
