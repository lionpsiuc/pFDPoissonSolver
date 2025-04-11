/**
 * @file jacobi.c
 *
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "jacobi.h"
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
void exchang2d_1(double x[][maxn], int nx, int row_s, int row_e, int col_s,
                 int col_e, MPI_Comm comm, int nbrleft, int nbrright, int nbrup,
                 int nbrdown) {
  int rank;
  int ny;
  int lnx, lny;
  lnx = col_e - col_s + 1;
  lny = row_e - row_s + 1;
  MPI_Comm_rank(comm, &rank);
  ny = nx;
  MPI_Sendrecv(&x[col_e][row_s], lny, MPI_DOUBLE, nbrright, 0,
               &x[col_s - 1][row_s], lny, MPI_DOUBLE, nbrleft, 0, comm,
               MPI_STATUS_IGNORE);
  MPI_Sendrecv(&x[col_s][row_s], lny, MPI_DOUBLE, nbrleft, 1,
               &x[col_e + 1][row_s], lny, MPI_DOUBLE, nbrright, 1, comm,
               MPI_STATUS_IGNORE);
  MPI_Datatype column_type;
  MPI_Type_vector(lnx, 1, maxn, MPI_DOUBLE, &column_type);
  MPI_Type_commit(&column_type);
  MPI_Sendrecv(&x[col_s][row_e], 1, column_type, nbrup, 2, &x[col_s][row_s - 1],
               1, column_type, nbrdown, 2, comm, MPI_STATUS_IGNORE);
  MPI_Sendrecv(&x[col_s][row_s], 1, column_type, nbrdown, 3,
               &x[col_s][row_e + 1], 1, column_type, nbrup, 3, comm,
               MPI_STATUS_IGNORE);
  MPI_Type_free(&column_type);
}

/**
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 *
 * @param[in/out/in,out] param Explain briefly.
 *
 * @return Explain briefly.
 */
double griddiff2d(double a[][maxn], double b[][maxn], int nx, int row_s,
                  int row_e, int col_s, int col_e) {
  double sum;
  double tmp;
  int    i, j;
  sum = 0.0;
  for (i = col_s; i <= col_e; i++) {
    for (j = row_s; j <= row_e; j++) {
      tmp = (a[i][j] - b[i][j]);
      sum = sum + tmp * tmp;
    }
  }
  return sum;
}

// Need to make this 2D
void nbxchange_and_sweep(double u[][maxn], double f[][maxn], int nx, int ny,
                         int s, int e, double unew[][maxn], MPI_Comm comm,
                         int nbrleft, int nbrright) {
  MPI_Request req[4];
  MPI_Status  status;
  int         idx;
  double      h;
  int         i, j, k;

  int myid;
  MPI_Comm_rank(comm, &myid);

  h = 1.0 / ((double) (nx + 1));

  /* int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, */
  /*               int source, int tag, MPI_Comm comm, MPI_Request *request); */
  /* int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest,
   */
  /* 		  int tag, MPI_Comm comm, MPI_Request *request); */

  MPI_Irecv(&u[s - 1][1], ny, MPI_DOUBLE, nbrleft, 1, comm, &req[0]);
  MPI_Irecv(&u[e + 1][1], ny, MPI_DOUBLE, nbrright, 2, comm, &req[1]);

  MPI_Isend(&u[e][1], ny, MPI_DOUBLE, nbrright, 1, comm, &req[2]);
  MPI_Isend(&u[s][1], ny, MPI_DOUBLE, nbrleft, 2, comm, &req[3]);

  /* perform purely local updates (that don't need ghosts) */
  /* 2 cols or less means all are on processor boundary */
  if (e - s + 1 > 2) {
    for (i = s + 1; i < e; i++) {
      for (j = 1; j < ny + 1; j++) {
        unew[i][j] = 0.25 * (u[i - 1][j] + u[i + 1][j] + u[i][j + 1] +
                             u[i][j - 1] - h * h * f[i][j]);
      }
    }
  }

  /* perform updates in j dir only for boundary cols */
  for (j = 1; j < ny + 1; j++) {
    unew[s][j] = 0.25 * (u[s][j + 1] + u[s][j - 1] - h * h * f[s][j]);
    unew[e][j] = 0.25 * (u[e][j + 1] + u[e][j - 1] - h * h * f[e][j]);
  }

  /* int MPI_Waitany(int count, MPI_Request array_of_requests[], */
  /*      int *index, MPI_Status *status) */
  for (k = 0; k < 4; k++) {

    MPI_Waitany(4, req, &idx, &status);
    /* printf("(--------------------------------- rank: %d): idx = %d\n",myid,
     * idx); */

    /* idx 0, 1 are recvs */
    switch (idx) {
      case 0:
        /* printf("myid: %d case idx 0: status.MPI_TAG: %d; status.MPI_SOURCE:
         * %d (idx: %d)\n",myid,status.MPI_TAG, status.MPI_SOURCE,idx); */

        /* left ghost update completed; update local leftmost column */
        for (j = 1; j < ny + 1; j++) {
          unew[s][j] += 0.25 * (u[s - 1][j]);
        }
        break;
      case 1:
        /* printf("myid: %d case idx 1: status.MPI_TAG: %d; status.MPI_SOURCE:
         * %d (idx: %d)\n",myid, status.MPI_TAG, status.MPI_SOURCE,idx); */

        /* right ghost update completed; update local rightmost
       column */
        for (j = 1; j < ny + 1; j++) {
          unew[e][j] += 0.25 * (u[e + 1][j]);
        }
        break;
      default: break;
    }
  }
  /* splitting this off to take account of case of one column assigned
  to proc -- so left and right node neighbours are ghosts so both
  the recvs must be complete*/
  for (j = 1; j < ny + 1; j++) {
    unew[s][j] += 0.25 * (u[s + 1][j]);
    unew[e][j] += 0.25 * (u[e - 1][j]);
  }
}

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
             int col_s, int col_e, double b[][maxn]) {
  double h;
  int    i, j;
  h = 1.0 / ((double) (nx + 1));
  for (i = col_s; i <= col_e; i++) {
    for (j = row_s; j <= row_e; j++) {
      b[i][j] = 0.25 * (a[i - 1][j] + a[i + 1][j] + a[i][j + 1] + a[i][j - 1] -
                        h * h * f[i][j]);
    }
  }
}
