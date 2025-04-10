/**
 * @file jacobih
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
void exchang2(double x[][maxn], int nx, int s, int e, MPI_Comm comm,
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
void exchang3(double x[][maxn], int nx, int s, int e, MPI_Comm comm,
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
void nbxchange_and_sweep(double u[][maxn], double f[][maxn], int nx, int ny,
                         int s, int e, double unew[][maxn], MPI_Comm comm,
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
void sweep1d(double a[][maxn], double f[][maxn], int nx, int s, int e,
             double b[][maxn]);
