/**
 * @file gatherwrite.h
 *
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 */

/**
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 *
 * @param[in/out/in,out] param Explain briefly.
 *
 * @return Explain briefly.
 */
void GatherGrid(double global_grid[][maxn], double a[][maxn], int s, int e,
                int nx, int ny, int myid, int nprocs, int* s_vals, int* e_vals,
                MPI_Comm comm);

/**
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 *
 * @param[in/out/in,out] param Explain briefly.
 *
 * @return Explain briefly.
 */
void write_grid(double a[][maxn], int nx, int ny, int rank, int s, int e,
                const char* filename, int write_to_stdout);
