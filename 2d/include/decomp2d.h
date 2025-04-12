/**
 * @file decomp2d.h
 *
 * @brief 2D domain decomposition utility for MPI parallelisation.
 *
 * This header provides a function for dividing a 2D domain among multiple MPI
 * processes arranged in a 2D process grid, ensuring a balanced distribution of
 * work across both dimensions.
 */

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
                 int* dims);
