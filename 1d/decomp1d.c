/**
 * @file decomp1d.c
 *
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 *
 * @param[in/out/in,out] param Explain briefly.
 *
 * @return Explain briefly.
 */
int MPE_Decomp1d(int n, int size, int rank, int* s, int* e) {
  int nlocal, deficit;
  nlocal  = n / size;
  *s      = rank * nlocal + 1;
  deficit = n % size;
  *s      = *s + ((rank < deficit) ? rank : deficit);
  if (rank < deficit)
    nlocal++;
  *e = *s + nlocal - 1;
  if (*e > n || rank == size - 1)
    *e = n;
  return MPI_SUCCESS;
}
