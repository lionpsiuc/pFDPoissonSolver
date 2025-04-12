/**
 * @file gatherwrite.h
 *
 * @brief Utility functions for collecting and writing distributed grid data.
 */

/**
 * @brief Gathers distributed grid data from all processes to the root process.
 *
 * Collects grid sections from all MPI processes and combines them into a
 * complete global grid on the root process.
 *
 * @param[out] global_grid Array to store the complete gathered grid (only used
 * by root process)
 * @param[in] a Local grid array containing this process's portion of the
 * solution
 * @param[in] s Starting column index of local domain
 * @param[in] e Ending column index of local domain
 * @param[in] nx Number of interior grid points in x-axis.
 * @param[in] ny Number of interior grid points in y-axis.
 * @param[in] myid Rank of the current MPI process.
 * @param[in] nprocs Total number of MPI processes.
 * @param[in] s_vals Array containing start indices for all processes.
 * @param[in] e_vals Array containing end indices for all processes.
 * @param[in] comm MPI communicator.
 */
void GatherGrid(double global_grid[][maxn], double a[][maxn], int s, int e,
                int nx, int ny, int myid, int nprocs, int* s_vals, int* e_vals,
                MPI_Comm comm);

/**
 * @brief Writes grid data to a file or terminal for visualisation.
 *
 * @param[in] filename Base name of the file to write.
 * @param[in] a Grid array containing the data to write.
 * @param[in] nx Number of interior grid points in x-axis.
 * @param[in] ny Number of interior grid points in y-axis.
 * @param[in] rank Rank of the current MPI process.
 * @param[in] s Starting index of the grid portion to write.
 * @param[in] e Ending index of the grid portion to write.
 * @param[in] write_to_stdout Flag to control whether to also print grid to
 *                            standard output.
 */
void write_grid(char* filename, double a[][maxn],
                int nx __attribute__((unused)), int ny, int rank, int s, int e,
                int write_to_stdout);
