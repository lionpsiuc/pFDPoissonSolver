/**
 * @file main_2d.c
 * @author H.Hodgins (hodginsh@tcd.ie)
 * @brief A program which computes the solution to the discretised Poisson
 equation using the Jacobi method for finite difference matrix. Uses a 2D domain
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

#include "jacobi2d.h"

#define maxit 2000
#define maxn 31 + 2

int  MPE_Decomp2d(int nrows, int ncols, int rank, int* coords, int* row_s,
                  int* row_e, int* col_s, int* col_e, int* dims);
void GatherGrid(double global_grid[maxn][maxn], double a[maxn][maxn], int row_s,
                int row_e, int col_s, int col_e, int nx, int ny, int myid,
                int nprocs, int* row_s_vals, int* row_e_vals, int* col_s_vals,
                int* col_e_vals);
void write_grid(char* filename, double a[][maxn], int nx, int ny, int rank,
                int row_s, int row_e, int col_s, int col_e);
void init_full_grid(double g[][maxn]);
void init_full_grids(double a[][maxn], double b[][maxn], double f[][maxn]);

void twodinit_basic(double a[][maxn], double b[][maxn], double f[][maxn],
                    int nx, int ny, int row_s, int row_e, int col_s, int col_e);

void print_full_grid(double x[][maxn]);
void print_in_order(double x[][maxn], MPI_Comm comm);
void print_grid_to_file(char* fname, double x[][maxn], int nx, int ny);

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

  twodinit_basic(a, b, f, nx, ny, row_s, row_e, col_s, col_e);
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

  write_grid("output2d", a, nx, ny, myid, row_s, row_e, col_s,
             col_e); // local grids

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
  GatherGrid(global_grid, a, row_s, row_e, col_s, col_e, nx, ny, myid, nprocs,
             row_s_vals, row_e_vals, col_s_vals, col_e_vals);
  if (myid == 0) {
    write_grid("global_2d_31", global_grid, nx, ny, myid, 1, maxn - 2, 1,
               maxn - 2);
  }
  MPI_Finalize();

  return 0;
}

void twodinit_basic(double a[][maxn], double b[][maxn], double f[][maxn],
                    int nx, int ny, int row_s, int row_e, int col_s,
                    int col_e) {
  int    i, j;
  double h = 1.0 / (nx + 1);

  /* set everything to 0 first */
  for (i = col_s - 1; i <= col_e + 1; i++) {
    for (j = row_s - 1; j <= row_e + 1; j++) {
      a[i][j] = 0.0;
      b[i][j] = 0.0;
      f[i][j] = 0.0;
    }
  }

  /* deal with boundaries */

  if (row_e == nx) {
    for (i = col_s; i <= col_e; i++) {
      double x     = (i) *h;
      a[i][nx + 1] = 1.0 / ((1 + x) * (1 + x) + 1); // top boundary
      b[i][nx + 1] = 1.0 / ((1 + x) * (1 + x) + 1);
    }
  }

  if (row_s == 1) {
    for (i = col_s; i <= col_e; i++) {
      a[i][0] = 0.0; // bottom boundary
      b[i][0] = 0.0;
    }
  }

  /* this is true for proc 0 */
  if (col_s == 1) {
    for (j = row_s; j <= row_e; j++) {
      double y = (j) *h;
      a[0][j]  = y / (1 + y * y); // left boundary
      b[0][j]  = y / (1 + y * y);
    }
  }

  /* this is true for proc size-1 */
  if (col_e == nx) {
    for (j = row_s; j <= row_e; j++) {
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

/**
 * @brief Caluclates which rows and columns each processor is responsible for.
 * Makes use of the Cart Create method.
 * @return int
 */

int MPE_Decomp2d(int nrows, int ncols, int rank, int* coords, int* row_s,
                 int* row_e, int* col_s, int* col_e, int* dims) {
  int rows_per_proc, cols_per_proc, row_deficit, col_deficit;

  // Row decomposition
  rows_per_proc = nrows / dims[0];
  row_deficit   = nrows % dims[0];
  *row_s        = coords[0] * rows_per_proc +
           ((coords[0] < row_deficit) ? coords[0] : row_deficit) + 1;
  if (coords[0] < row_deficit)
    rows_per_proc++;
  *row_e = *row_s + rows_per_proc - 1;
  if (*row_e > nrows || coords[0] == dims[0] - 1)
    *row_e = nrows;
  int total_rows_assigned =
      *row_e - *row_s + 1; // The total number of rows assigned to this rank
  *row_e =
      nrows - (*row_s - 1); // Invert the start index to get the new end index
  *row_s = *row_e - total_rows_assigned + 1;

  // Column decomposition
  cols_per_proc = ncols / dims[1];
  col_deficit   = ncols % dims[1];
  *col_s        = coords[1] * cols_per_proc +
           ((coords[1] < col_deficit) ? coords[1] : col_deficit) + 1;
  if (coords[1] < col_deficit)
    cols_per_proc++;
  *col_e = *col_s + cols_per_proc - 1;
  if (*col_e > ncols || coords[1] == dims[1] - 1)
    *col_e = ncols;

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
                int row_s, int row_e, int col_s, int col_e) {
  char full_filename[256];
  sprintf(full_filename, "%s_%d.txt", filename, rank);
  FILE* file = fopen(full_filename, "w");
  if (!file) {
    fprintf(stderr, "Error opening file\n");
    exit(-1);
  }

  int i, j;

  // write to file
  for (j = row_e + 1; j >= row_s - 1; j--) {
    for (i = col_s - 1; i <= col_e + 1; i++) {
      fprintf(file, "%2.6lf ", a[i][j]);
    }
    fprintf(file, "\n");
  }

  // print to stdout
  //  printf("write_grid output proc %d\n",rank);
  //  for (j = row_e; j >=row_s; j--)
  //  {
  //    for(i = col_s; i <=col_e;i++)
  //    {
  //      printf("%2.6lf ", a[i][j]);
  //    }
  //    printf("\n");
  //  }
}

/**
 * @brief Gathers the grids from each process onto process 0.
 *
 * @param global_grid Used to store the combined grids from each process.
 * @param a The local grid on each process. Must have conceptually column major
 * ordering (Mesh style).
 * @param row_s Starting index of rows belonging to process.
 * @param row_e Ending index of rows belonging to process.
 * @param col_s Starting index of columns belonging to process.
 * @param col_e Ending index of columns belonging to process.
 * @param nx Dimension of interior points.
 * @param ny Dimension of interior points.
 * @param myid Rank of process.
 * @param nprocs Total number of processes.
 * @param row_s_vals Array holding values of start row indexes for each process
 * in order.
 * @param row_e_vals Array holding values of end row indexes for each process in
 * order.
 * @param col_s_vals Array holding values of start column indexes for each
 * process in order.
 * @param col_e_vals Array holding values of end column indexes for each process
 * in order.
 */
void GatherGrid(double global_grid[maxn][maxn], double a[maxn][maxn], int row_s,
                int row_e, int col_s, int col_e, int nx, int ny, int myid,
                int nprocs, int* row_s_vals, int* row_e_vals, int* col_s_vals,
                int* col_e_vals) {
  int lny;
  lny = row_e - row_s + 1;

  if (myid == 0) {
    int    i, j;
    double h = 1.0 / (nx + 1);
    for (i = col_s - 1; i <= col_e; i++) // include left boundary
    {
      for (j = row_s; j < row_e + 1; j++) {
        global_grid[i][j] = a[i][j]; // copy in the col
      }
    }

    /* deal with boundaries */

    for (i = col_s; i < maxn - 1; i++) {
      double x               = (i) *h;
      global_grid[i][nx + 1] = 1.0 / ((1 + x) * (1 + x) + 1); // top boundary
    }

    for (i = col_s; i < maxn - 1; i++) {
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
  }
  if (myid != 0) // if id is not root, send the columns to root
  {
    int p;
    for (p = col_s; p <= col_e; p++) {
      // printf("sending column %d to process 0 from process %d\n",p,myid);
      MPI_Send(&a[p][row_s], lny, MPI_DOUBLE, 0, p + myid,
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
      for (col = col_s_vals[proc]; col <= col_e_vals[proc]; col++) {
        MPI_Recv(&global_grid[col][row_s_vals[proc]], lny, MPI_DOUBLE, proc,
                 proc + col, MPI_COMM_WORLD, &status);
        // printf("received column %d from process %d\n",col,proc);
      }
    }
  }
}
