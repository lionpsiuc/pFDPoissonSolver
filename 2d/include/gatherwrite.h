/**
 * @file gatherwrite.h
 *
 * @brief Utility functions for collecting and writing distributed 2D grid data.
 */

#include "poisson2d.h"

/**
 * @brief Gathers distributed 2D grid data from all processes to the root
 *        process.
 *
 * Collects 2D grid sections from all MPI processes and combines them into a
 * complete global grid on the root process.
 *
 * @param[out] global_grid Array to store the complete gathered grid (only used
 * by root process)
 * @param[in] a Local grid array containing this process's portion of the
 * solution
 * @param[in] row_s Starting row index of local domain.
 * @param[in] row_e Ending row index of local domain.
 * @param[in] col_s Starting column index of local domain.
 * @param[in] col_e Ending column index of local domain.
 * @param[in] nx Number of interior grid points in x-axis.
 * @param[in] ny Number of interior grid points in y-axis.
 * @param[in] myid Rank of the current MPI process.
 * @param[in] nprocs Total number of MPI processes.
 * @param[in] row_s_vals Array containing start row indices for all processes.
 * @param[in] row_e_vals Array containing end row indices for all processes.
 * @param[in] col_s_vals Array containing start column indices for all
 *                       processes.
 * @param[in] col_e_vals Array containing end column indices for all processes.
 * @param[in] comm MPI communicator.
 */
void GatherGrid2D(double global_grid[][maxn], double a[][maxn], int row_s,
                  int row_e, int col_s, int col_e, int nx, int ny, int myid,
                  int nprocs, int* row_s_vals, int* row_e_vals, int* col_s_vals,
                  int* col_e_vals, MPI_Comm comm);

/**
 * @brief Writes 2D grid data to a file or terminal for visualisation.
 *
 * @param[in] filename Base name of the file to write.
 * @param[in] a Grid array containing the data to write.
 * @param[in] nx Number of interior grid points in x-axis.
 * @param[in] ny Number of interior grid points in y-axis.
 * @param[in] rank Rank of the current MPI process.
 * @param[in] row_s Starting row index of the grid portion to write.
 * @param[in] row_e Ending row index of the grid portion to write.
 * @param[in] col_s Starting column index of the grid portion to write.
 * @param[in] col_e Ending column index of the grid portion to write.
 * @param[in] write_to_stdout Flag to control whether to also print grid to
 *                            standard output.
 */
void write_grid(char* filename, double a[][maxn],
                int nx __attribute__((unused)), int ny __attribute__((unused)),
                int rank, int row_s, int row_e, int col_s, int col_e,
                int write_to_stdout);
