#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <mpi.h>

#include "auxfuncs.h"
#include "jacobi.h"
#include "poisson1d.h"

#define maxit 2000 // Maximum iterations as per requirements

#include "decomp1d.h"

// Function declarations
void   init_full_grid(double g[][maxn]);
void   init_full_grids(double a[][maxn], double b[][maxn], double f[][maxn]);
void   init_dirichlet_boundary(double a[][maxn], double b[][maxn],
                               double f[][maxn], int nx, int ny, int s, int e);
void   print_full_grid(double x[][maxn]);
void   print_in_order(double x[][maxn], MPI_Comm comm);
void   print_grid_to_file(char* fname, double x[][maxn], int nx, int ny);
double analytic_solution(double x, double y);
double calculate_error(double a[][maxn], int nx, int s, int e);

void init_dirichlet_boundary(double a[][maxn], double b[][maxn],
                             double f[][maxn], int nx, int ny, int s, int e) {
  int    i, j;
  double h = 1.0 / (nx + 1); // Grid spacing
  double x, y;

  /* set everything to 0 first */
  for (i = s - 1; i <= e + 1; i++) {
    for (j = 0; j <= ny + 1; j++) {
      a[i][j] = 0.0;
      b[i][j] = 0.0;
      f[i][j] = 0.0; // f = 0 for Laplace equation
    }
  }

  /* Bottom boundary: u(x,0) = 0 */
  for (i = s; i <= e; i++) {
    a[i][0] = 0.0;
    b[i][0] = 0.0;
  }

  /* Top boundary: u(x,1) = 1/((1+x)^2 + 1) */
  for (i = s; i <= e; i++) {
    x            = i * h;
    a[i][ny + 1] = 1.0 / ((1.0 + x) * (1.0 + x) + 1.0);
    b[i][ny + 1] = 1.0 / ((1.0 + x) * (1.0 + x) + 1.0);
  }

  /* Left boundary: u(0,y) = y/(1+y^2) */
  if (s == 1) {
    for (j = 1; j <= ny; j++) {
      y       = j * h;
      a[0][j] = y / (1.0 + y * y);
      b[0][j] = y / (1.0 + y * y);
    }
  }

  /* Right boundary: u(1,y) = y/(4+y^2) */
  if (e == nx) {
    for (j = 1; j <= ny; j++) {
      y            = j * h;
      a[nx + 1][j] = y / (4.0 + y * y);
      b[nx + 1][j] = y / (4.0 + y * y);
    }
  }
}

/* Calculate the analytic solution for a given grid point */
double analytic_solution(double x, double y) {
  return y / ((1.0 + x) * (1.0 + x) + y * y);
}

/* Calculate the maximum error between numerical and analytic solutions */
double calculate_error(double a[][maxn], int nx, int s, int e) {
  int    i, j;
  double h = 1.0 / (nx + 1); // Grid spacing
  double x, y, exact_sol, error, max_error = 0.0;

  for (i = s; i <= e; i++) {
    for (j = 1; j <= nx; j++) {
      x         = i * h;
      y         = j * h;
      exact_sol = analytic_solution(x, y);
      error     = fabs(a[i][j] - exact_sol);
      if (error > max_error) {
        max_error = error;
      }
    }
  }

  return max_error;
}
int main(int argc, char** argv) {
  double a[maxn][maxn], b[maxn][maxn], f[maxn][maxn];
  int    nx, ny;
  int    myid, nprocs;
  int    nbrleft, nbrright, s, e, it;
  double glob_diff, glob_max_error;
  double ldiff, local_max_error;
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
    if (argc == 2) {
      nx = atoi(argv[1]);
    } else {
      /* Default grid size if not specified (use 15 for testing) */
      nx = 15;
    }

    if (nx > maxn - 2) {
      fprintf(stderr, "Grid size too large\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

  MPI_Bcast(&nx, 1, MPI_INT, 0, cart_comm);
  printf("(myid: %d) nx = %d\n", cart_rank, nx);
  ny = nx;

  init_full_grids(a, b, f);

  MPE_Decomp1d(nx, nprocs, cart_rank, &s, &e);

  printf("(myid: %d) nx: %d s: %d; e: %d; nbrleft: %d; nbrright: %d\n",
         cart_rank, nx, s, e, nbrleft, nbrright);

  /* Initialize with the specified boundary conditions */
  init_dirichlet_boundary(a, b, f, nx, ny, s, e);

  /* Print initial grid */
  if (cart_rank == 0) {
    printf("\nInitial grid:\n");
  }
  print_in_order(a, cart_comm);

  /* Start timer */
  t1 = MPI_Wtime();

  /* Jacobi iteration */
  glob_diff = 1000;
  for (it = 0; it < maxit && glob_diff > tol; it++) {
    exchangi1(a, ny, s, e, cart_comm, nbrleft, nbrright);
    sweep1d(a, f, nx, s, e, b);

    exchangi1(b, nx, s, e, cart_comm, nbrleft, nbrright);
    sweep1d(b, f, nx, s, e, a);

    ldiff = griddiff(a, b, nx, s, e);
    MPI_Allreduce(&ldiff, &glob_diff, 1, MPI_DOUBLE, MPI_SUM, cart_comm);

    if (cart_rank == 0 && (it % 100 == 0 || it == maxit - 1)) {
      printf("(Iteration %d) Global difference: %e\n", it, glob_diff);
    }
  }

  t2 = MPI_Wtime();

  if (cart_rank == 0) {
    if (it >= maxit) {
      printf("Maximum iterations (%d) reached.\n", maxit);
    } else {
      printf("Converged after %d iterations.\n", it);
    }
    printf("Computation time: %lf seconds\n", t2 - t1);
  }

  /* Calculate error between numerical and analytic solutions */
  local_max_error = calculate_error(a, nx, s, e);
  MPI_Reduce(&local_max_error, &glob_max_error, 1, MPI_DOUBLE, MPI_MAX, 0,
             cart_comm);

  if (cart_rank == 0) {
    printf("\nMaximum error between numerical and analytic solutions: %e\n",
           glob_max_error);
  }

  /* Print final grid */
  if (cart_rank == 0) {
    printf("\nFinal grid after %d iterations:\n", it);
  }
  print_in_order(a, cart_comm);

  /* Write grid to file for the case where grid size is nx */
  char filename[100];
  sprintf(filename, "poisson_sol_nx%d", nx);
  if (cart_rank == 0) {
    print_grid_to_file(filename, a, nx, ny);
    printf("Solution written to file: %s\n", filename);
  }

  /* Free the Cartesian communicator */
  MPI_Comm_free(&cart_comm);

  MPI_Finalize();
  return 0;
}
