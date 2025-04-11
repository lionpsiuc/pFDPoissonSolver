/**
 * @file jacobi.h
 *
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 */

#include "poisson1d.h"

/**
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 *
 * @param[in/out/in,out] param Explain briefly.
 *
 * @return Explain briefly.
 */
void exchang1(double x[][maxn], int nx, int s, int e, MPI_Comm comm,
              int nbrbottom, int nbrtop);

/**
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 *
 * @param[in/out/in,out] param Explain briefly.
 *
 * @return Explain briefly.
 */
void exchangi1(double x[][maxn], int nx, int s, int e, MPI_Comm comm,
               int nbrleft, int nbrright);

/**
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 *
 * @param[in/out/in,out] param Explain briefly.
 *
 * @return Explain briefly.
 */
double griddiff(double a[][maxn], double b[][maxn], int nx, int s, int e);

/**
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 *
 * @param[in/out/in,out] param Explain briefly.
 *
 * @return Explain briefly.
 */
void sweep1d(double a[][maxn], double f[][maxn], int nx, int s, int e,
             double b[][maxn]);
