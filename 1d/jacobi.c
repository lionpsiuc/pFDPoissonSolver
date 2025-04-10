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
              int nbrleft, int nbrright) {
  int rank;
  int ny;
  MPI_Comm_rank(comm, &rank);
  ny = nx;
  MPI_Ssend(&x[e][1], ny, MPI_DOUBLE, nbrright, 0, comm);
  // printf("(myid: %d) sent \"col\" %d with %d entries to nbr: %d\n", rank, e,
  // ny,
  //        nbrright);
  MPI_Recv(&x[s - 1][1], ny, MPI_DOUBLE, nbrleft, 0, comm, MPI_STATUS_IGNORE);
  // printf("(myid: %d) recvd into \"col\" %d from %d entries from nbr: %d\n",
  //        rank, s - 1, ny, nbrleft);
  MPI_Ssend(&x[s][1], ny, MPI_DOUBLE, nbrleft, 1, comm);
  // printf("(myid: %d) sent \"col\" %d with %d entries to nbr: %d\n", rank, s,
  // ny,
  //        nbrleft);
  MPI_Recv(&x[e + 1][1], ny, MPI_DOUBLE, nbrright, 1, comm, MPI_STATUS_IGNORE);
  // printf("(myid: %d) recvd into \"col\" %d from %d entries from nbr: %d\n",
  //        rank, e + 1, ny, nbrright);
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
void exchangi1(double x[][maxn], int nx, int s, int e, MPI_Comm comm,
               int nbrleft, int nbrright) {
  MPI_Request reqs[4];
  MPI_Irecv(&x[s - 1][1], nx, MPI_DOUBLE, nbrleft, 0, comm, &reqs[0]);
  MPI_Irecv(&x[e + 1][1], nx, MPI_DOUBLE, nbrright, 0, comm, &reqs[1]);
  MPI_Isend(&x[e][1], nx, MPI_DOUBLE, nbrright, 0, comm, &reqs[2]);
  MPI_Isend(&x[s][1], nx, MPI_DOUBLE, nbrleft, 0, comm, &reqs[3]);
  MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
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
void exchang2(double x[][maxn], int nx, int s, int e, MPI_Comm comm,
              int nbrleft, int nbrright) {
  int coord;
  int rank;
  MPI_Comm_rank(comm, &rank);
  coord = rank;
  if (coord % 2 == 0) {
    MPI_Ssend(&x[e][1], nx, MPI_DOUBLE, nbrright, 0, comm);
    MPI_Recv(&x[s - 1][1], nx, MPI_DOUBLE, nbrleft, 0, comm, MPI_STATUS_IGNORE);
    MPI_Ssend(&x[s][1], nx, MPI_DOUBLE, nbrleft, 1, comm);
    MPI_Recv(&x[e + 1][1], nx, MPI_DOUBLE, nbrright, 1, comm,
             MPI_STATUS_IGNORE);
  } else {
    MPI_Recv(&x[s - 1][1], nx, MPI_DOUBLE, nbrleft, 0, comm, MPI_STATUS_IGNORE);
    MPI_Ssend(&x[e][1], nx, MPI_DOUBLE, nbrright, 0, comm);
    MPI_Recv(&x[e + 1][1], nx, MPI_DOUBLE, nbrright, 1, comm,
             MPI_STATUS_IGNORE);
    MPI_Ssend(&x[s][1], nx, MPI_DOUBLE, nbrleft, 1, comm);
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
void exchang3(double x[][maxn], int nx, int s, int e, MPI_Comm comm,
              int nbrleft, int nbrright) {
  MPI_Sendrecv(&x[e][1], nx, MPI_DOUBLE, nbrright, 0, &x[s - 1][1], nx,
               MPI_DOUBLE, nbrleft, 0, comm, MPI_STATUS_IGNORE);
  MPI_Sendrecv(&x[s][1], nx, MPI_DOUBLE, nbrleft, 1, &x[e + 1][1], nx,
               MPI_DOUBLE, nbrright, 1, comm, MPI_STATUS_IGNORE);
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
double griddiff(double a[][maxn], double b[][maxn], int nx, int s, int e) {
  double sum;
  double tmp;
  sum = 0.0;
  for (int i = s; i <= e; i++) {
    for (int j = 1; j < nx + 1; j++) {
      tmp = (a[i][j] - b[i][j]);
      sum = sum + tmp * tmp;
    }
  }
  return sum;
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
void nbxchange_and_sweep(double u[][maxn], double f[][maxn], int nx, int ny,
                         int s, int e, double unew[][maxn], MPI_Comm comm,
                         int nbrleft, int nbrright) {
  MPI_Request req[4];
  MPI_Status  status;
  int         idx;
  double      h;
  h = 1.0 / ((double) (nx + 1)); // Grid spacing
  int myid;
  MPI_Comm_rank(comm, &myid);
  MPI_Irecv(&u[s - 1][1], ny, MPI_DOUBLE, nbrleft, 1, comm, &req[0]);
  MPI_Irecv(&u[e + 1][1], ny, MPI_DOUBLE, nbrright, 2, comm, &req[1]);
  MPI_Isend(&u[e][1], ny, MPI_DOUBLE, nbrright, 1, comm, &req[2]);
  MPI_Isend(&u[s][1], ny, MPI_DOUBLE, nbrleft, 2, comm, &req[3]);
  if (e - s + 1 > 2) {
    for (int i = s + 1; i < e; i++) {
      for (int j = 1; j < ny + 1; j++) {
        unew[i][j] = 0.25 * (u[i - 1][j] + u[i + 1][j] + u[i][j + 1] +
                             u[i][j - 1] - h * h * f[i][j]);
      }
    }
  }
  for (int j = 1; j < ny + 1; j++) {
    unew[s][j] = 0.25 * (u[s][j + 1] + u[s][j - 1] - h * h * f[s][j]);
    unew[e][j] = 0.25 * (u[e][j + 1] + u[e][j - 1] - h * h * f[e][j]);
  }
  for (int k = 0; k < 4; k++) {
    MPI_Waitany(4, req, &idx, &status);
    printf("rank = %d has idx = %d\n", myid, idx);
    switch (idx) {
      case 0:
        printf("myid = %d (case 0, and where idx = %d) has the following "
               "statuses: status.MPI_TAG: %d; status.MPI_SOURCE: %d\n",
               myid, idx, status.MPI_TAG, status.MPI_SOURCE);
        for (int j = 1; j < ny + 1; j++) {
          unew[s][j] += 0.25 * (u[s - 1][j]);
        }
        break;
      case 1:
        printf("myid = %d (case 1, and where idx = %d) has the following "
               "statuses: status.MPI_TAG: %d; status.MPI_SOURCE: %d\n",
               myid, idx, status.MPI_TAG, status.MPI_SOURCE);
        for (int j = 1; j < ny + 1; j++) {
          unew[e][j] += 0.25 * (u[e + 1][j]);
        }
        break;
      default: break;
    }
  }
  for (int j = 1; j < ny + 1; j++) {
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
void sweep1d(double a[][maxn], double f[][maxn], int nx, int s, int e,
             double b[][maxn]) {
  double h;
  h = 1.0 / ((double) (nx + 1));
  for (int i = s; i <= e; i++) {
    for (int j = 1; j < nx + 1; j++) {
      b[i][j] = 0.25 * (a[i - 1][j] + a[i + 1][j] + a[i][j + 1] + a[i][j - 1] -
                        h * h * f[i][j]);
    }
  }
}
