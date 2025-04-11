/**
 * @file aux.c
 *
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 */

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "aux.h"
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
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 *
 * @param[in/out/in,out] param Explain briefly.
 *
 * @return Explain briefly.
 */
void init_twod(double a[][maxn], double b[][maxn], double f[][maxn], int nx,
               int ny, int row_s, int row_e, int col_s, int col_e) {
  double h = 1.0 / ((double) (nx + 1)); // Grid spacing

  // Set everything to zero first
  for (int i = col_s - 1; i <= col_e + 1; i++) {
    for (int j = row_s - 1; j <= row_e + 1; j++) {
      a[i][j] = 0.0;
      b[i][j] = 0.0;
      f[i][j] = 0.0;
    }
  }

  if (row_e == nx) {
    for (int i = col_s; i <= col_e; i++) {
      double x     = i * h; // Transform to coordinate system
      a[i][nx + 1] = 1.0 / ((1.0 + x) * (1.0 + x) + 1.0);
      b[i][nx + 1] = 1.0 / ((1.0 + x) * (1.0 + x) + 1.0);
    }
  }
  if (row_s == 1) {
    for (int i = col_s; i <= col_e; i++) {
      a[i][0] = 0.0;
      b[i][0] = 0.0;
    }
  }
  if (col_s == 1) {
    for (int j = row_s; j <= row_e; j++) {
      double y = j * h; // Transform to coordinate system
      a[0][j]  = y / (1.0 + y * y);
      b[0][j]  = y / (1.0 + y * y);
    }
  }
  if (col_e == nx) {
    for (int j = row_s; j <= row_e; j++) {
      double y     = j * h; // Transform to coordinate system
      a[nx + 1][j] = y / (4.0 + y * y);
      b[nx + 1][j] = y / (4.0 + y * y);
    }
  }
}
