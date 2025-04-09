#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>
#define maxn 31 + 2
#include "jacobi2d.h"

/**
 * @brief Carries out the ghost column and row exchanges using MPI sendrecv.
 *
 * @param x The local grid on each process.
 * @param nx Problem size.
 * @param row_s Starting index of rows each process is responsible for.
 * @param row_e Ending index of rows each process is responsible for.
 * @param col_s Starting index of columns each process is responsible for.
 * @param col_e Ending index of columns each process is responsible for.
 * @param comm MPI communicator
 * @param nbrleft Left neighbor of processor.
 * @param nbrright Right neighbor of processor.
 * @param nbrup Upper neighbor of processor.
 * @param nbrdown Lower neighbor of processor.
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

  // cols

  MPI_Sendrecv(&x[col_e][row_s], lny, MPI_DOUBLE, nbrright, 0,
               &x[col_s - 1][row_s], lny, MPI_DOUBLE, nbrleft, 0, comm,
               MPI_STATUS_IGNORE);
  // printf("(myid: %d) sent \"col\" %d with %d entries to nbr: %d\n",nbrleft,
  // col_e, ny, nbrright); printf("(myid: %d) recvd into \"col\" %d from %d
  // entries from nbr: %d\n",nbrright, col_s-1, ny, nbrleft);

  MPI_Sendrecv(&x[col_s][row_s], lny, MPI_DOUBLE, nbrleft, 1,
               &x[col_e + 1][row_s], lny, MPI_DOUBLE, nbrright, 1, comm,
               MPI_STATUS_IGNORE);
  // printf("(myid: %d) sent \"col\" %d with %d entries to nbr: %d\n",nbrright,
  // col_e, ny, nbrleft); printf("(myid: %d) recvd into \"col\" %d from %d
  // entries from nbr: %d\n",nbrleft, col_e+1, ny, nbrright);

  // rows, need strided datatype
  MPI_Datatype column_type;
  MPI_Type_vector(lnx, 1, maxn, MPI_DOUBLE, &column_type);
  MPI_Type_commit(&column_type);

  // send last col (top row to nbrup from nbrdown
  MPI_Sendrecv(&x[col_s][row_e], 1, column_type, nbrup, 2, &x[col_s][row_s - 1],
               1, column_type, nbrdown, 2, comm, MPI_STATUS_IGNORE);
  MPI_Sendrecv(&x[col_s][row_s], 1, column_type, nbrdown, 3,
               &x[col_s][row_e + 1], 1, column_type, nbrup, 3, comm,
               MPI_STATUS_IGNORE);

  MPI_Type_free(&column_type);
}

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

/* sendrecv */
void exchang3(double x[][maxn], int nx, int s, int e, MPI_Comm comm,
              int nbrleft, int nbrright) {

  MPI_Sendrecv(&x[e][1], nx, MPI_DOUBLE, nbrright, 0, &x[s - 1][1], nx,
               MPI_DOUBLE, nbrleft, 0, comm, MPI_STATUS_IGNORE);

  MPI_Sendrecv(&x[s][1], nx, MPI_DOUBLE, nbrleft, 1, &x[e + 1][1], nx,
               MPI_DOUBLE, nbrright, 1, comm, MPI_STATUS_IGNORE);
}

void exchangi1(double x[][maxn], int nx, int s, int e, MPI_Comm comm,
               int nbrleft, int nbrright) {
  MPI_Request reqs[4];

  MPI_Irecv(&x[s - 1][1], nx, MPI_DOUBLE, nbrleft, 0, comm, &reqs[0]);
  MPI_Irecv(&x[e + 1][1], nx, MPI_DOUBLE, nbrright, 0, comm, &reqs[1]);
  MPI_Isend(&x[e][1], nx, MPI_DOUBLE, nbrright, 0, comm, &reqs[2]);
  MPI_Isend(&x[s][1], nx, MPI_DOUBLE, nbrleft, 0, comm, &reqs[3]);
  /* not doing anything useful here */

  MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
}

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
    printf("(--------------------------------- rank: %d): idx = %d\n", myid,
           idx);

    /* idx 0, 1 are recvs */
    switch (idx) {
      case 0:
        /* printf("myid: %d case idx 0: status.MPI_TAG: %d; status.MPI_SOURCE:
         * %d (idx: %d)\n",myid,status.MPI_TAG, status.MPI_SOURCE,idx); */
        if (nbrleft != MPI_PROC_NULL &&
            (status.MPI_TAG != 1 || status.MPI_SOURCE != nbrleft)) {
          fprintf(stderr,
                  "Error: I don't understand the world: (tag %d; source %d)\n",
                  status.MPI_TAG, status.MPI_SOURCE);
          MPI_Abort(comm, 1);
        }

        /* left ghost update completed; update local leftmost column */
        for (j = 1; j < ny + 1; j++) {
          unew[s][j] += 0.25 * (u[s - 1][j]);
        }
        break;
      case 1:
        /* printf("myid: %d case idx 1: status.MPI_TAG: %d; status.MPI_SOURCE:
         * %d (idx: %d)\n",myid, status.MPI_TAG, status.MPI_SOURCE,idx); */
        if (nbrright != MPI_PROC_NULL &&
            (status.MPI_TAG != 2 || status.MPI_SOURCE != nbrright)) {
          fprintf(stderr,
                  "Error: I don't understand the world: (tag %d; source %d)\n",
                  status.MPI_TAG, status.MPI_SOURCE);
          MPI_Abort(comm, 1);
        }
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
