/**
 * @file gatherwrite.h
 *
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 */

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
void GatherGrid2D(double global_grid[][maxn], double a[][maxn], int row_s,
                  int row_e, int col_s, int col_e, int nx, int ny, int myid,
                  int nprocs, int* row_s_vals, int* row_e_vals, int* col_s_vals,
                  int* col_e_vals, MPI_Comm comm);

/**
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 *
 * @param[in/out/in,out] param Explain briefly.
 *
 * @return Explain briefly.
 */
void write_grid(char* filename, double a[][maxn],
                int nx __attribute__((unused)), int ny __attribute__((unused)),
                int rank, int row_s, int row_e, int col_s, int col_e,
                int write_to_stdout);
