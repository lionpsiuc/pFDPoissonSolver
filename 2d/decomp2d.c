/**
 * @file decomp2d.c
 *
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 *
 * @param[in/out/in,out] param Explain briefly.
 *
 * @return Explain briefly.
 */
int MPE_Decomp2d(int nrows, int ncols, int rank, int* coords, int* row_s,
                 int* row_e, int* col_s, int* col_e, int* dims) {
  int rows_per_proc, cols_per_proc, row_deficit, col_deficit;
  rows_per_proc = nrows / dims[0];
  row_deficit   = nrows % dims[0];
  *row_s        = coords[0] * rows_per_proc +
           ((coords[0] < row_deficit) ? coords[0] : row_deficit) + 1;
  if (coords[0] < row_deficit)
    rows_per_proc++;
  *row_e = *row_s + rows_per_proc - 1;
  if (*row_e > nrows || coords[0] == dims[0] - 1)
    *row_e = nrows;
  int total_rows_assigned = *row_e - *row_s + 1;
  *row_e                  = nrows - (*row_s - 1);
  *row_s                  = *row_e - total_rows_assigned + 1;
  cols_per_proc           = ncols / dims[1];
  col_deficit             = ncols % dims[1];
  *col_s                  = coords[1] * cols_per_proc +
           ((coords[1] < col_deficit) ? coords[1] : col_deficit) + 1;
  if (coords[1] < col_deficit)
    cols_per_proc++;
  *col_e = *col_s + cols_per_proc - 1;
  if (*col_e > ncols || coords[1] == dims[1] - 1)
    *col_e = ncols;
  return MPI_SUCCESS;
}
