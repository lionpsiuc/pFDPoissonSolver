/**
 * @file decomp2d.h
 *
 * @brief 2D domain decomposition utility for MPI parallelisation.
 *
 * This header provides a function for dividing a 2D domain among multiple MPI
 * processes arranged in a 2D process grid, ensuring a balanced distribution of
 * work across both dimensions.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * @brief Calculates 2D domain decomposition for an MPI process.
 *
 * Divides a 2D grid of  elements as evenly as possible among processes arranged
 * in a 2D process grid. Determines the start and end indices for both row and
 * column dimensions for the process located at the given coordinates in the
 * process grid. The function ensures balanced distribution in both dimensions
 * by giving one extra element to processes that need it.
 *
 * @param[in] nrows Total number of rows in the global grid.
 * @param[in] ncols Total number of columns in the global grid.
 * @param[in] rank Process rank.
 * @param[in] coords Array containing the process's coordinates in the 2D
 *                   process grid.
 * @param[out] row_s Pointer to store the starting row index for this process.
 * @param[out] row_e Pointer to store the ending row index for this process.
 * @param[out] col_s Pointer to store the starting column index for this
 *                   process.
 * @param[out] col_e Pointer to store the ending column index for this process.
 * @param[in] dims Array containing the dimensions of the 2D process grid.
 *
 * @return MPI_SUCCESS on successful completion.
 */
int MPE_Decomp2d(int nrows, int ncols, int rank __attribute__((unused)),
                 int* coords, int* row_s, int* row_e, int* col_s, int* col_e,
                 int* dims) {
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
