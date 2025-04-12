/**
 * @file aux.c
 *
 * @brief Implementation of utility functions for grid initialisation.
 */

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "../include/aux.h"
#include "../include/poisson1d.h"

/**
 * @brief Initialises grid arrays with a default value.
 *
 * Sets all elements in the grid arrays to an initial junk value.
 *
 * @param[out] a Grid array for current solution iteration.
 * @param[out] b Grid array for next solution iteration.
 * @param[out] f Grid array for right-hand side function values.
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
               int ny, int s, int e) {
  double h = 1.0 / ((double) (nx + 1)); // Grid spacing

  // Set everything to zero first
  for (int i = s - 1; i <= e + 1; i++) {
    for (int j = 0; j <= nx + 1; j++) {
      a[i][j] = 0.0;
      b[i][j] = 0.0;
      f[i][j] = 0.0;
    }
  }

  // Set the top boundary where u(x,1)=1/((1+x)^2+1)
  for (int i = s; i <= e; i++) {
    double x     = i * h; // Transform to coordinate system
    a[i][nx + 1] = 1.0 / ((1.0 + x) * (1.0 + x) + 1.0);
    b[i][nx + 1] = 1.0 / ((1.0 + x) * (1.0 + x) + 1.0);
  }

  // Set the left boundary where u(0,y)=y/(1+y^2)
  if (s == 1) {
    for (int j = 0; j <= ny + 1; j++) {
      double y = j * h;                     // Transform to coordinate system
      if (j == 0 || (1.0 + y * y) == 0.0) { // Protect against division by zero
        a[0][j] = 0.0;
        b[0][j] = 0.0;
      } else {
        a[0][j] = y / (1.0 + y * y);
        b[0][j] = y / (1.0 + y * y);
      }
    }
  }

  // Set the right boundary where u(1,y)=y/(4+y^2)
  if (e == nx) {
    for (int j = 0; j <= ny + 1; j++) {
      double y = j * h;                     // Transform to coordinate system
      if (j == 0 || (4.0 + y * y) == 0.0) { // Protect against division by zero
        a[nx + 1][j] = 0.0;
        b[nx + 1][j] = 0.0;
      } else {
        a[nx + 1][j] = y / (4.0 + y * y);
        b[nx + 1][j] = y / (4.0 + y * y);
      }
    }
  }
}
