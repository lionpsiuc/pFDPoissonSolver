/**
 * @file main_1d.c
 * @author H.Hodgins (hodginsh@tcd.ie)
 * @brief A program which computes the solution to the discretised Poisson
 equation using the Jacobi method for finite difference matrix. Uses a 1D domain
 decomposition.
 * @version 1.0
 * @date 2024-04-15
 *
 *
 */
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "jacobi_1d.h"

#define maxit 2000
#define maxn 31 + 2

void write_grid(char* filename, double a[][maxn], int nx, int ny, int rank,
                int s, int e);
void GatherGrid(double global_grid[maxn][maxn], double a[maxn][maxn], int s,
                int e, int nx, int ny, int myid, int nprocs, int* s_vals,
                int* e_vals);
int  MPE_Decomp1d(int n, int size, int rank, int* s, int* e);
void init_full_grid(double g[][maxn]);
void init_full_grids(double a[][maxn], double b[][maxn], double f[][maxn]);

void onedinit_basic(double a[][maxn], double b[][maxn], double f[][maxn],
                    int nx, int ny, int s, int e);

void print_full_grid(double x[][maxn]);
void print_in_order(double x[][maxn], MPI_Comm comm);
void print_grid_to_file(char* fname, double x[][maxn], int nx, int ny);

int main(int argc, char** argv) {
  double a[maxn][maxn], b[maxn][maxn], f[maxn][maxn];
  int    nx, ny;
  int    myid, nprocs;
  /* MPI_Status status; */
  int    nbrleft, nbrright, s, e, it;
  double glob_diff;
  double ldiff;
  double t1, t2;
  double tol = 1.0E-11;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  if (myid == 0) {
    /* set the size of the problem */
    if (argc > 2) {
      fprintf(stderr, "---->Usage: mpirun -np <nproc> %s <nx>\n", argv[0]);
      fprintf(stderr, "---->(for this code nx=ny)\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (argc == 2) {
      nx = atoi(argv[1]);
    }
    if (argc == 1) {
      nx = 15;
    }

    if (nx > maxn - 2) {
      fprintf(stderr, "grid size too large\n");
      exit(1);
    }
  }

  MPI_Bcast(&nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
  printf("(myid: %d) nx = %d\n", myid, nx);
  ny = nx;

  init_full_grids(a, b, f);

  /*
  MPI_Cart_create
  */
  int      ndims;
  int      dims[1];
  int      periods[1];
  int      reorder;
  MPI_Comm cartcomm;
  ndims      = 1;
  dims[0]    = nprocs;
  periods[0] = 0;
  reorder    = 0;

  MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &cartcomm);
  MPI_Cart_shift(cartcomm, 0, 1, &nbrleft, &nbrright);

  if (myid == 0) {
    nbrleft = MPI_PROC_NULL;
  }

  if (myid == nprocs - 1) {
    nbrright = MPI_PROC_NULL;
  }

  MPE_Decomp1d(nx, nprocs, myid, &s, &e);

  printf("(myid: %d) nx: %d s: %d; e: %d; nbrleft: %d; nbrright: %d\n", myid,
         nx, s, e, nbrleft, nbrright);

  onedinit_basic(a, b, f, nx, ny, s, e);

  t1 = MPI_Wtime();

  glob_diff = 1000;
  for (it = 0; it < maxit; it++) {

    exchang1(a, ny, s, e, MPI_COMM_WORLD, nbrleft, nbrright);
    sweep1d(a, f, nx, s, e, b);

    exchang1(b, nx, s, e, MPI_COMM_WORLD, nbrleft, nbrright);
    sweep1d(b, f, nx, s, e, a);

    ldiff = griddiff(a, b, nx, s, e);
    MPI_Allreduce(&ldiff, &glob_diff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (myid == 0 && it % 10 == 0) {
      printf("(myid %d) locdiff: %lf; glob_diff: %lf\n", myid, ldiff,
             glob_diff);
    }
    if (glob_diff < tol) {
      if (myid == 0) {
        printf("iterative solve converged\n");
      }
      break;
    }
  }

  t2 = MPI_Wtime();

  printf("DONE! (it: %d)\n", it);

  double h = 1.0 / (nx + 1);
  double g[maxn][maxn];
  double x, y;
  int    l, m;
  for (l = 0; l <= nx + 1; l++) {
    for (m = 0; m <= ny + 1; m++) {
      x       = (l) *h;
      y       = (m) *h;
      g[l][m] = y / ((1 + x) * (1 + x) + y * y);
    }
  }

  if (myid == 0) {
    if (it == maxit) {
      fprintf(stderr, "Failed to converge\n");
    }
    printf("Run took %lf s\n", t2 - t1);
  }

  print_in_order(a, MPI_COMM_WORLD);
  if (myid == 0) {
    printf("Analytical Solution:\n");
    print_full_grid(g);
  }

  if (nprocs == 1) {
    print_grid_to_file("grid1.txt", a, nx, ny);
    print_full_grid(a);
  }

  write_grid("output_1d", a, nx, ny, myid, s, e); // local grids

  // creating arrays to hold the start and end col indexes for each proc
  // we need these for the gather grid function later.
  int* s_vals = NULL;
  int* e_vals = NULL;
  if (myid == 0) {
    s_vals = (int*) malloc(nprocs * sizeof(int));
    e_vals = (int*) malloc(nprocs * sizeof(int));
  }

  // the arrays are populated using gather.
  MPI_Gather(&s, 1, MPI_INT, s_vals, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Gather(&e, 1, MPI_INT, e_vals, 1, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Barrier(cartcomm);

  double global_grid[maxn]
                    [maxn]; // the array which will contain the gathered grids.
  GatherGrid(global_grid, a, s, e, nx, ny, myid, nprocs, s_vals,
             e_vals); // gather the grids onto proc 0.

  if (myid == 0) {
    write_grid("global_1d_31", global_grid, nx, ny, myid, 0, maxn - 1);
    write_grid("analytical", g, nx, ny, myid, 0, maxn - 1); // analytical soln
  }
  MPI_Finalize();

  return 0;
}

void onedinit_basic(double a[][maxn], double b[][maxn], double f[][maxn],
                    int nx, int ny, int s, int e) {
  int    i, j;
  double h = 1.0 / (nx + 1);

  /* set everything to 0 first */
  for (i = s - 1; i <= e + 1; i++) {
    for (j = 0; j <= nx + 1; j++) {
      a[i][j] = 0.0;
      b[i][j] = 0.0;
      f[i][j] = 0.0;
    }
  }

  /* deal with boundaries */
  for (i = s; i <= e; i++) {
    double x     = (i) *h;
    a[i][0]      = 0.0;
    b[i][0]      = 0.0;
    a[i][nx + 1] = 1 / ((1 + x) * (1 + x) + 1); // top boundary
    b[i][nx + 1] = 1 / ((1 + x) * (1 + x) + 1);
  }

  /* this is true for proc 0 */
  if (s == 1) {
    for (j = 0; j < ny + 1; j++) {
      double y = (j) *h;
      a[0][j]  = y / (1 + y * y); // left boundary
      b[0][j]  = y / (1 + y * y);
    }
  }

  /* this is true for proc size-1 */
  if (e == nx) {
    for (j = 0; j < nx + 1; j++) {
      double y     = (j) *h;
      a[nx + 1][j] = y / (4 + y * y); // right boundary
      b[nx + 1][j] = y / (4 + y * y);
    }
  }
}

void init_full_grid(double g[][maxn]) {
  int          i, j;
  const double junkval = -5;

  for (i = 0; i < maxn; i++) {
    for (j = 0; j < maxn; j++) {
      g[i][j] = junkval;
    }
  }
}

/* set global a,b,f to initial arbitrarily chosen junk value */
void init_full_grids(double a[][maxn], double b[][maxn], double f[][maxn]) {
  int          i, j;
  const double junkval = -5;

  for (i = 0; i < maxn; i++) {
    for (j = 0; j < maxn; j++) {
      a[i][j] = junkval;
      b[i][j] = junkval;
      f[i][j] = junkval;
    }
  }
}

/* prints to stdout in GRID view */
void print_full_grid(double x[][maxn]) {
  int i, j;
  for (j = maxn - 1; j >= 0; j--) {
    for (i = 0; i < maxn; i++) {
      if (x[i][j] < 10000.0) {
        printf("|%2.6lf| ", x[i][j]);
      } else {
        printf("%9.2lf ", x[i][j]);
      }
    }
    printf("\n");
  }
}

void print_in_order(double x[][maxn], MPI_Comm comm) {
  int myid, size;
  int i;

  MPI_Comm_rank(comm, &myid);
  MPI_Comm_size(comm, &size);
  MPI_Barrier(comm);
  printf("Attempting to print in order\n");
  sleep(1);
  MPI_Barrier(comm);

  for (i = 0; i < size; i++) {
    if (i == myid) {
      printf("proc %d\n", myid);
      print_full_grid(x);
    }
    fflush(stdout);
    usleep(500);
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

void print_grid_to_file(char* fname, double x[][maxn], int nx, int ny) {
  FILE* fp;
  int   i, j;

  fp = fopen(fname, "w");
  if (!fp) {
    fprintf(stderr, "Error: can't open file %s\n", fname);
    exit(4);
  }

  for (j = ny + 1; j >= 0; j--) {
    for (i = 0; i < nx + 2; i++) {
      fprintf(fp, "%lf ", x[i][j]);
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
}

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

/**
 * @brief A function which writes the grid belonging to each processor to file
 * and also to stdout.
 *
 * @param filename Name of first part of output file.
 * @param a The grid we are writing to file.
 * @param nx Number of rows in grid.
 * @param ny Number of columns in grid.
 * @param rank Process id.
 * @param s Starting index of column belonging to each process.
 */
void write_grid(char* filename, double a[][maxn], int nx, int ny, int rank,
                int s, int e) {
  char full_filename[256];
  sprintf(full_filename, "%s_%d.txt", filename, rank);
  FILE* file = fopen(full_filename, "w");
  if (!file) {
    fprintf(stderr, "Error opening file\n");
    exit(-1);
  }

  int i, j;

  // write to file
  for (j = ny + 1; j >= 0; j--) // if a is colum major conceptually
  {
    for (i = s; i <= e; i++) {
      fprintf(file, "%2.6lf ", a[i][j]);
    }
    fprintf(file, "\n");
  }

  // print to stdout
  printf("write_grid output proc %d\n", rank);
  for (j = ny + 1; j >= 0; j--) // a is colum major conceptually
  {
    for (i = s; i <= e; i++) {
      printf("%2.6lf ", a[i][j]);
    }
    printf("\n");
  }
}
/**
 * @brief Gathers the grids from each process onto process 0.
 *
 * @param global_grid Used to store the combined grids from each process.
 * @param a The local grid on each process. Must have conceptually column major
 * ordering (Mesh style).
 * @param s Starting index of columns belonging to process.
 * @param e Ending index of columns belonging to process.
 * @param nx Dimension of interior points.
 * @param ny Dimension of interior points.
 * @param myid Rank of process.
 * @param nprocs Number of processes.
 * @param s_vals Array holding values of start indexes for each process in
 * order.
 * @param e_vals Array holding values of end indexes for each process in order.
 */
void GatherGrid(double global_grid[maxn][maxn], double a[maxn][maxn], int s,
                int e, int nx, int ny, int myid, int nprocs, int* s_vals,
                int* e_vals) {

  if (myid == 0) {
    int    i, j;
    double h = 1.0 / (nx + 1);
    for (i = s - 1; i <= e; i++) // include left boundary
    {
      for (j = 1; j < nx + 1; j++) {
        global_grid[i][j] = a[i][j]; // copy in the col
      }
    }

    /* deal with boundaries */

    for (i = s; i < maxn - 1; i++) {
      double x               = (i) *h;
      global_grid[i][nx + 1] = 1.0 / ((1 + x) * (1 + x) + 1); // top boundary
    }

    for (i = s; i < maxn - 1; i++) {
      global_grid[i][0] = 0.0; // bottom boundary
    }

    for (j = maxn - 1; j >= 1; j--) {
      double y          = (j) *h;
      global_grid[0][j] = y / (1 + y * y); // left boundary
    }

    for (j = maxn - 1; j >= 1; j--) {
      double y               = (j) *h;
      global_grid[nx + 1][j] = y / (4 + y * y); // right boundary
    }
    // printf("copy complete:\n");
  }
  if (myid != 0) // if id is not root, send the columns to root
  {
    int p;
    for (p = s; p <= e; p++) {
      // printf("sending column %d to process 0 from process %d\n",p,myid);
      MPI_Send(&a[p][1], nx, MPI_DOUBLE, 0, p + myid,
               MPI_COMM_WORLD); // send col p to root
      // printf("sent\n");
    }
  }
  if (myid == 0) // receive the cols
  {

    // receive cols that were sent
    int        proc, col, i, j;
    MPI_Status status;
    for (proc = 1; proc < nprocs; proc++) {
      for (col = s_vals[proc]; col <= e_vals[proc]; col++) {
        MPI_Recv(&global_grid[col][1], nx, MPI_DOUBLE, proc, proc + col,
                 MPI_COMM_WORLD, &status);
        // printf("received column %d from process %d\n",col,proc);
      }
    }
  }
}
