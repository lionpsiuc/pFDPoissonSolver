/**
 * @file jacobi.h
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
void exchang2d_1(double x[][maxn], int nx __attribute__((unused)), int row_s,
                 int row_e, int col_s, int col_e, MPI_Comm comm, int nbrleft,
                 int nbrright, int nbrup, int nbrdown, MPI_Datatype row_type);

/**
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 *
 * @param[in/out/in,out] param Explain briefly.
 *
 * @return Explain briefly.
 */
void exchang2d_nb(double x[][maxn], int nx __attribute__((unused)), int row_s,
                  int row_e, int col_s, int col_e, MPI_Comm comm, int nbrleft,
                  int nbrright, int nbrup, int nbrdown, MPI_Datatype row_type);

/**
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 *
 * @param[in/out/in,out] param Explain briefly.
 *
 * @return Explain briefly.
 */
double griddiff2d(double a[][maxn], double b[][maxn],
                  int nx __attribute__((unused)), int row_s, int row_e,
                  int col_s, int col_e);

/**
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 *
 * @param[in/out/in,out] param Explain briefly.
 *
 * @return Explain briefly.
 */
void sweep2d(double a[][maxn], double f[][maxn], int nx, int row_s, int row_e,
             int col_s, int col_e, double b[][maxn]);
