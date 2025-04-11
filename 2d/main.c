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

  // Solution storage arrays
  double a[maxn][maxn];           // Current solution grid
  double b[maxn][maxn];           // Next iteration solution grid
  double f[maxn][maxn];           // Right-hand side function values
  double global_grid[maxn][maxn]; // Global solution after gathering from all
                                  // processes using GatherGrid2D

  // Problem size; note that for this assignment, they are equal
  int nx; // Size of the x-axis; interior points only
  int ny; // Size of the y-axis; interior points only

  // MPI process information
  int  myid, nprocs; // Process rank and number of processes
  char name[1024];   // Processor name which is used for debugging
  int  namelen;      // Length of processor name

  // Domain decomposition
  int snbrup, nbrright, nbrdown, nbrleft; // Explain how this works
  int row_s, row_e, col_s, col_e;         // Explain how this works

  // Iteration and convergence
  int    it;            // Iteration counter
  double glob_diff;     // Global differences between iterations
  double ldiff;         // Local difference on the respective process
  double tol = 1.0E-11; // Convergence tolerance

  double t1, t2; // Timing

  // Initialise the MPI environment
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Get_processor_name(name, &namelen);
  // printf("myid = %d is running on node %s\n", myid, name);

  // Programme header
  if (myid == 0) {
    printf("\n=======================================================\n");
    printf("                   2D Implementation                   \n");
    printf("=======================================================\n\n");
    printf("Number of processes: %d\n", nprocs);

    // Parse command-line arguments
    if (argc > 2) {
      fprintf(stderr, "Usage: %s [grid_size]\n", argv[0]);
      fprintf(stderr,
              "Note: grid_size will be used for both x and y dimensions\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (argc == 2) {
      nx = atoi(argv[1]);
    } else {
      nx = 31; // Default to grid size 31 as per the assignment
    }

    printf("Grid size: %d x %d\n\n", nx, nx);

    if (nx > maxn - 2) {
      fprintf(stderr, "Error: Grid size exceeds maximum allowed (%d)\n",
              maxn - 2);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

  // Broadcast grid size to all processes
  MPI_Bcast(&nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
  ny = nx; // Square grid

  // Initialize arrays
  init_full_grids(a, b, f);

  // Create 2D Cartesian topology
  if (myid == 0) {
    printf("Setting up 2D Cartesian process topology...\n");
  }

  // MPI_Cart_create as per the assignment instructions
  int      ndims      = 2;      // Explain how this works
  int      dims[2]    = {0, 0}; // Explain how this works
  int      periods[2] = {0, 0}; // Explain how this works
  int      reorder    = 1;      // Explain how this works
  MPI_Comm cart_comm;           // Explain how this works
  int      coords[2];           // Explain how this works

  // Create optimal process grid dimensions
  MPI_Dims_create(nprocs, ndims, dims);

  // Create the Cartesian communicator
  MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &cart_comm);

  // Get this process's coordinates in the grid
  MPI_Cart_coords(cart_comm, myid, ndims, coords);

  // Find neighbouring processes
  MPI_Cart_shift(cart_comm, 0, 1, &nbrup, &nbrdown); // Up and down neighbours
  MPI_Cart_shift(cart_comm, 1, 1, &nbrleft,
                 &nbrright); // Left and right neighbours

  // Explain how this works
  if (coords[0] == 0) {
    nbrup = MPI_PROC_NULL; // Explain how this works
  }
  if (coords[0] == dims[0] - 1) {
    nbrdown = MPI_PROC_NULL; // Explain how this works
  }
  if (coords[1] == 0) {
    nbrleft = MPI_PROC_NULL; // Explain how this works
  }
  if (coords[1] == dims[1] - 1) {
    nbrright = MPI_PROC_NULL; // Explain how this works
  }

  // Calculate local domain boundaries
  MPE_Decomp2d(nx, ny, myid, coords, &row_s, &row_e, &col_s, &col_e, dims);

  // Print process layout
  MPI_Barrier(MPI_COMM_WORLD);
  if (myid == 0) {
    printf("\n--- PROCESS GRID LAYOUT ---\n");
  }
  for (int p = 0; p < nprocs; p++) {
    if (p == myid) {
      printf("Process %2d: Coords=(%d,%d) | Domain=(rows %2d-%2d, cols "
             "%2d-%2d) | Neighbors=(U:%2d,D:%2d,L:%2d,R:%2d)\n",
             myid, coords[0], coords[1], row_s, row_e, col_s, col_e, nbrup,
             nbrdown, nbrleft, nbrright);
      fflush(stdout);
    }
    usleep(1000); // Small delay
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // Initialise local domain with boundary conditions
  init_twod(a, b, f, nx, ny, row_s, row_e, col_s, col_e);

  // Create an MPI_Datatype for row exchanges (i.e., non-contiguous data);
  // explain how this works
  int          lnx = col_e - col_s + 1;
  MPI_Datatype row_type;
  MPI_Type_vector(lnx, 1, maxn, MPI_DOUBLE, &row_type);
  MPI_Type_commit(&row_type);

  // Start timing
  if (myid == 0) {
    printf("\nStarting iterative solver...\n");
  }
  t1 = MPI_Wtime();

  // Main iteration loop
  glob_diff = 1000; // Initial difference
  for (it = 0; it < maxit; it++) {
    // First half-iteration: Use blocking Sendrecv for ghost cell exchange
    exchang2d_1(a, nx, row_s, row_e, col_s, col_e, cart_comm, nbrleft, nbrright,
                nbrup, nbrdown, row_type);
    sweep2d(a, f, nx, row_s, row_e, col_s, col_e, b);

    // Second half-iteration: Use non-blocking communication for ghost exchange
    exchang2d_nb(b, nx, row_s, row_e, col_s, col_e, cart_comm, nbrleft,
                 nbrright, nbrup, nbrdown, row_type);
    sweep2d(b, f, nx, row_s, row_e, col_s, col_e, a);

    // Check convergence
    ldiff = griddiff2d(a, b, nx, row_s, row_e, col_s, col_e);
    MPI_Allreduce(&ldiff, &glob_diff, 1, MPI_DOUBLE, MPI_SUM, cart_comm);

    // Report progress (less frequently to reduce output)
    if (myid == 0 && (it % 100 == 0 || glob_diff < tol)) {
      printf("Iteration %4d: Global difference = %.6e\n", it, glob_diff);
    }

    // Check for convergence
    if (glob_diff < tol) {
      if (myid == 0) {
        printf("\nCONVERGED after %d iterations! (tolerance: %.6e)\n", it + 1,
               tol);
      }
      break;
    }
  }

  // End timing
  t2 = MPI_Wtime();

  if (myid == 0) {
    printf("\nSolver completed in %.6f seconds\n", t2 - t1);
    if (it == maxit) {
      printf("WARNING: Maximum iterations (%d) reached without convergence.\n",
             maxit);
    }
  }

  // Calculate analytical solution for comparison
  double g[maxn][maxn];
  double h = 1.0 / (nx + 1);
  double x, y;

  for (int i = 0; i <= nx + 1; i++) {
    for (int j = 0; j <= ny + 1; j++) {
      x       = i * h;
      y       = j * h;
      g[i][j] = y / ((1 + x) * (1 + x) + y * y); // Analytical solution
    }
  }

  // Allocate memory for domain decomposition information (only on rank 0)
  int* row_s_vals = NULL;
  int* row_e_vals = NULL;
  int* col_s_vals = NULL;
  int* col_e_vals = NULL;

  if (myid == 0) {
    row_s_vals = (int*) malloc(nprocs * sizeof(int));
    row_e_vals = (int*) malloc(nprocs * sizeof(int));
    col_s_vals = (int*) malloc(nprocs * sizeof(int));
    col_e_vals = (int*) malloc(nprocs * sizeof(int));

    if (!row_s_vals || !row_e_vals || !col_s_vals || !col_e_vals) {
      fprintf(stderr, "Error: Memory allocation failed\n");
      MPI_Abort(cart_comm, 1);
    }
  }

  // Gather domain decomposition information to rank 0
  MPI_Gather(&row_s, 1, MPI_INT, row_s_vals, 1, MPI_INT, 0, cart_comm);
  MPI_Gather(&row_e, 1, MPI_INT, row_e_vals, 1, MPI_INT, 0, cart_comm);
  MPI_Gather(&col_s, 1, MPI_INT, col_s_vals, 1, MPI_INT, 0, cart_comm);
  MPI_Gather(&col_e, 1, MPI_INT, col_e_vals, 1, MPI_INT, 0, cart_comm);

  GatherGrid2D(global_grid, a, row_s, row_e, col_s, col_e, nx, ny, myid, nprocs,
               row_s_vals, row_e_vals, col_s_vals, col_e_vals);

  // Write results to files
  if (myid == 0) {
    char filename[64];
    sprintf(filename, "global2dnx%d", nx);

    printf("\nWriting final solution to files...\n");

    // Write numerical solution
    write_grid(filename, global_grid, nx, ny, myid, 1, nx, 1, nx);

    // Write analytical solution
    write_grid("analytical", g, nx, ny, myid, 1, nx, 1, nx);

    // Calculate error statistics
    double max_error = 0.0;
    double avg_error = 0.0;
    int    count     = 0;

    for (int i = 1; i <= nx; i++) {
      for (int j = 1; j <= ny; j++) {
        double error = fabs(global_grid[i][j] - g[i][j]);
        avg_error += error;
        count++;
        if (error > max_error) {
          max_error = error;
        }
      }
    }
    avg_error /= count;

    // Report error statistics
    printf("\nERROR ANALYSIS:\n");
    printf("  Maximum error: %.8e\n", max_error);
    printf("  Average error: %.8e\n", avg_error);
    printf("\nFinal results written to:\n");
    printf("  - %s.txt (numerical solution)\n", filename);
    printf("  - analytical.txt (analytical solution)\n");
  }

  // Clean up
  MPI_Type_free(&row_type);

  if (myid == 0) {
    free(row_s_vals);
    free(row_e_vals);
    free(col_s_vals);
    free(col_e_vals);

    printf("=======================================================");
    printf("                        SUCCESS                        ");
    printf("=======================================================");
  }

  MPI_Finalize();
  return 0;
}
