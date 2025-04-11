/**
 * @file aux.h
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
void init_full_grid(double g[][maxn]);

/**
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 *
 * @param[in/out/in,out] param Explain briefly.
 *
 * @return Explain briefly.
 */
void init_full_grids(double a[][maxn], double b[][maxn], double f[][maxn]);

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
               int ny, int row_s, int row_e, int col_s, int col_e);

/**
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 *
 * @param[in/out/in,out] param Explain briefly.
 *
 * @return Explain briefly.
 */
void print_full_grid(double x[][maxn]);

/**
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 *
 * @param[in/out/in,out] param Explain briefly.
 *
 * @return Explain briefly.
 */
void print_grid_to_file(char* fname, double x[][maxn], int nx, int ny);

/**
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 *
 * @param[in/out/in,out] param Explain briefly.
 *
 * @return Explain briefly.
 */
void print_in_order(double x[][maxn], MPI_Comm comm);
