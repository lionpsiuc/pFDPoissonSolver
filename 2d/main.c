/**
 * @file main.c
 *
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 */

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "aux.h"
#include "decomp2d.h"
#include "gatherwrite.h"
#include "jacobi.h"
#include "poisson2d.h"

#define maxit 2000

/**
 * @brief Main function.
 *
 * Further explanation, if required.
 *
 * @param[in] argc Explain briefly.
 * @param[in] argv Explain briefly.
 *
 * @return 0 upon successful execution.
 */
int main(int argc, char** argv) {
  double a[maxn][maxn], b[maxn][maxn], f[maxn][maxn];
  int    nx, ny;
  int    myid, nprocs;
  /* MPI_Status status; */
  int    it;
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
      fprintf(stderr, "Usage is as follows: mpirun -np nprocs %s nx\n",
              argv[0]);
      fprintf(stderr, "Note that nx = ny, so specifying nx is enough\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (argc == 2) {
      nx = atoi(argv[1]);
    }
    if (argc == 1) { // We default to a 31x31 grid as per the fourth question
      nx = 31;
    }

    if (nx > maxn - 2) {
      fprintf(stderr, "Grid size is too large\n");
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
  //   nbrleft  = myid - 1;
  //   nbrright = myid + 1;

  int      ndims;
  int      dims[2]    = {0, 0};
  int      periods[2] = {0, 0};
  int      reorder;
  MPI_Comm cartcomm;
  ndims      = 2;
  periods[0] = 0;
  periods[1] = 0;
  reorder    = 0;
  int coords[2];

  MPI_Dims_create(nprocs, ndims, dims);
  MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &cartcomm);

  MPI_Cart_coords(cartcomm, myid, ndims, coords);
  int nbrup, nbrdown, nbrleft, nbrright;
  MPI_Cart_shift(cartcomm, 0, 1, &nbrup, &nbrdown);
  MPI_Cart_shift(cartcomm, 1, 1, &nbrleft, &nbrright);
  if (coords[0] == 0) { // first row
    nbrup = MPI_PROC_NULL;
  }
  if (coords[0] == dims[0] - 1) { // last row
    nbrdown = MPI_PROC_NULL;
  }
  if (coords[1] == 0) { // first column
    nbrleft = MPI_PROC_NULL;
  }
  if (coords[1] == dims[1] - 1) { // last column
    nbrright = MPI_PROC_NULL;
  }
  int row_s, row_e, col_s, col_e;
  printf("(myid: %d) nbrleft: %d; nbrright: %d,nbrup: %d; nbrdown: %d\n", myid,
         nbrleft, nbrright, nbrup, nbrdown);
  MPE_Decomp2d(nx, ny, myid, coords, &row_s, &row_e, &col_s, &col_e, dims);
  printf("Rank %d: Rows %d to %d, Cols %d to %d\n", myid, row_s, row_e, col_s,
         col_e);

  init_twod(a, b, f, nx, ny, row_s, row_e, col_s, col_e);
  print_in_order(a, cartcomm); // Print the grid for debugging purposes
  t1 = MPI_Wtime();

  glob_diff = 1000;
  for (it = 0; it < maxit; it++) {

    exchang2d_1(a, nx, row_s, row_e, col_s, col_e, MPI_COMM_WORLD, nbrleft,
                nbrright, nbrup, nbrdown);
    sweep2d(a, f, nx, row_s, row_e, col_s, col_e, b);

    exchang2d_1(b, nx, row_s, row_e, col_s, col_e, MPI_COMM_WORLD, nbrleft,
                nbrright, nbrup, nbrdown);
    sweep2d(b, f, nx, row_s, row_e, col_s, col_e, a);

    ldiff = griddiff2d(a, b, nx, row_s, row_e, col_s, col_e);
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

  // print_in_order(a, MPI_COMM_WORLD);
  // if(myid == 0)
  // {
  //   printf("Analytical Solution:\n");
  //   print_full_grid(g);
  // }

  // if( nprocs == 1  ){
  //   print_grid_to_file("grid1.txt", a,  nx, ny);
  //   print_full_grid(a);
  // }

  int* row_s_vals = NULL;
  int* row_e_vals = NULL;
  int* col_s_vals = NULL;
  int* col_e_vals = NULL;
  if (myid == 0) {
    row_s_vals = (int*) malloc(nprocs * sizeof(int));
    row_e_vals = (int*) malloc(nprocs * sizeof(int));
    col_s_vals = (int*) malloc(nprocs * sizeof(int));
    col_e_vals = (int*) malloc(nprocs * sizeof(int));
  }

  MPI_Gather(&row_s, 1, MPI_INT, row_s_vals, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Gather(&row_e, 1, MPI_INT, row_e_vals, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Gather(&col_s, 1, MPI_INT, col_s_vals, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Gather(&col_e, 1, MPI_INT, col_e_vals, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // MPI_Barrier(MPI_COMM_WORLD);
  // MPI_Barrier(cartcomm);

  double global_grid[maxn][maxn];
  GatherGrid2D(global_grid, a, row_s, row_e, col_s, col_e, nx, ny, myid, nprocs,
               row_s_vals, row_e_vals, col_s_vals, col_e_vals);
  if (myid == 0) {
    write_grid("global2dnx31", global_grid, nx, ny, myid, 1, maxn - 2, 1,
               maxn - 2);
  }
  MPI_Finalize();

  return 0;
}
