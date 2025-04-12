/**
 * @file decomp1d.h
 *
 * @brief 1D domain decomposition utility for MPI parallelisation.
 *
 * This header provides a function for dividing a 1D domain among multiple MPI
 * processes, ensuring a balanced distribution of work.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * @brief Calculates 1D domain decomposition for an MPI process.
 *
 * Divides elements as evenly as possible among size processes, and determines
 * the start and end indices for the given process rank. The function ensures
 * a balanced distribution by giving one extra element to the first deficit
 * processes when the number of elements is not evenly divisible by the total
 * number of MPI processes.
 *
 * @param[in] n The total number of elements to be distributed.
 * @param[in] size The total number of MPI processes.
 * @param[in] rank The rank of the current MPI process.
 * @param[out] s Pointer to store the starting index for this process.
 * @param[out] e Pointer to store the ending index for this process.
 *
 * @return MPI_SUCCESS on successful completion.
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
