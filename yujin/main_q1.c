#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <mpi.h>

#include "jacobi.h"
#include "poisson1d.h"

#define maxit 1000

#include "decomp1d.h"

void init_full_grid(double g[][maxn]);
void init_full_grids(double a[][maxn], double b[][maxn], double f[][maxn]);
void init_basic_bv(double a[][maxn], double b[][maxn], double f[][maxn], int nx,
                   int ny, int s, int e);
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
  char   name[1024];
  int    namelen;

  /* Variables for MPI Cartesian topology */
  MPI_Comm cart_comm;
  int      ndims = 1; /* 1D decomposition */
  int dims[1]; /* Array specifying the number of processes in each dimension */
  int periods[1] = {0}; /* Non-periodic grid */
  int reorder    = 1;   /* Allow reordering of ranks */
  int cart_rank;        /* Rank in the new Cartesian communicator */

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  MPI_Get_processor_name(name, &namelen);
  printf("(myid %d): running on node: %s\n", myid, name);

  /* Create a 1D Cartesian topology */
  dims[0] = nprocs;
  MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &cart_comm);

  /* Get rank in the new communicator */
  MPI_Comm_rank(cart_comm, &cart_rank);

  /* Get neighbors using MPI_Cart_shift */
  MPI_Cart_shift(cart_comm, 0, 1, &nbrleft, &nbrright);

  if (myid == 0) {
    /* set the size of the problem */
    if (argc > 2) {
      fprintf(stderr, "---->Usage: mpirun -np <nproc> %s <nx>\n", argv[0]);
      fprintf(stderr, "---->(for this code nx=ny)\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (argc == 2) {
      nx = atoi(argv[1]);
    } else {
      nx = maxn - 2; // Default size if not specified
    }

    if (nx > maxn - 2) {
      fprintf(stderr, "grid size too large\n");
      exit(1);
    }
  }

  MPI_Bcast(&nx, 1, MPI_INT, 0, cart_comm);
  printf("(myid: %d) nx = %d\n", cart_rank, nx);
  ny = nx;

  init_full_grids(a, b, f);

  /* Note: nbrleft and nbrright are now set by MPI_Cart_shift */

  MPE_Decomp1d(nx, nprocs, cart_rank, &s, &e);

  printf("(myid: %d) nx: %d s: %d; e: %d; nbrleft: %d; nbrright: %d\n",
         cart_rank, nx, s, e, nbrleft, nbrright);

  init_basic_bv(a, b, f, nx, ny, s, e);

  print_in_order(a, cart_comm);

  /* MPI_Barrier(cart_comm); */
  /* MPI_Abort(cart_comm, 1); */

  t1 = MPI_Wtime();

  glob_diff = 1000;
  for (it = 0; it < maxit; it++) {

    exchangi1(a, ny, s, e, cart_comm, nbrleft, nbrright);
    sweep1d(a, f, nx, s, e, b);

    exchangi1(b, nx, s, e, cart_comm, nbrleft, nbrright);
    sweep1d(b, f, nx, s, e, a);

    ldiff = griddiff(a, b, nx, s, e);
    MPI_Allreduce(&ldiff, &glob_diff, 1, MPI_DOUBLE, MPI_SUM, cart_comm);
    if (cart_rank == 0 && it % 10 == 0) {
      printf("(myid %d) locdiff: %lf; glob_diff: %lf\n", cart_rank, ldiff,
             glob_diff);
    }
    /* if(it%5==0){ */
    /*   print_in_order(a, cart_comm); */
    /* } */
    if (glob_diff < tol) {
      if (cart_rank == 0) {
        printf("iterative solve converged\n");
      }
      break;
    }
  }

  t2 = MPI_Wtime();

  printf("DONE! (it: %d)\n", it);

  if (cart_rank == 0) {
    if (it == maxit) {
      fprintf(stderr, "Failed to converge\n");
    }
    printf("Run took %lf s\n", t2 - t1);
  }

  print_in_order(a, cart_comm);
  if (nprocs == 1) {
    print_grid_to_file("grid", a, nx, ny);
    print_full_grid(a);
  }

  /* Free the Cartesian communicator */
  MPI_Comm_free(&cart_comm);

  MPI_Finalize();
  return 0;
}

void init_basic_bv(double a[][maxn], double b[][maxn], double f[][maxn], int nx,
                   int ny, int s, int e) {
  int i, j;

  double left, bottom, right, top;

  left   = -1.0;
  bottom = 1.0;
  right  = 2.0;
  top    = 3.0;

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
    a[i][0]      = bottom;
    b[i][0]      = bottom;
    a[i][nx + 1] = top;
    b[i][nx + 1] = top;
  }

  /* this is true for proc 0 */
  if (s == 1) {
    for (j = 1; j < nx + 1; j++) {
      a[0][j] = left;
      b[0][j] = left;
    }
  }

  /* this is true for proc size-1 */
  if (e == nx) {
    for (j = 1; j < nx + 1; j++) {
      a[nx + 1][j] = right;
      b[nx + 1][j] = right;
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
