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
#include "decomp1d.h"
#include "gatherwrite.h"
#include "jacobi.h"
#include "poisson1d.h"

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
                                  // processes using GatherGrid

  // Problem size; note that for this assignment, they are equal
  int nx; // Size of the x-axis; interior points only
  int ny; // Size of the y-axis; interior points only

  // MPI process information
  int  myid, nprocs; // Process rank and number of processes
  char name[1024];   // Processor name which is used for debugging
  int  namelen;      // Length of processor name

  // Domain decomposition
  int nbrleft, nbrright; // Ranks of the left and right neighbouring processes;
                         // note that these are set by MPI_Cart_shift
  int s, e; // The start and end indices of the local domain; these are the
            // columns assigned to the respective process

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

  // Initialise all arrays to zero
  for (int i = 0; i < maxn; i++) {
    for (int j = 0; j < maxn; j++) {
      a[i][j]           = 0.0;
      b[i][j]           = 0.0;
      f[i][j]           = 0.0;
      global_grid[i][j] = 0.0;
    }
  }

  // Process the command-line arguments which in turn, sets the size of our
  // problem (i.e., the grid size to use)
  if (myid == 0) {
    if (argc > 2) {
      fprintf(stderr, "Usage is as follows: mpirun -np nprocs %s nx\n",
              argv[0]);
      fprintf(stderr, "Note that nx = ny, so specifying nx is enough\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (argc == 2) {
      nx = atoi(argv[1]);
    }
    if (argc == 1) { // We default to a 31x31 grid as per the third question
      nx = 31;
    }
    if (nx > maxn - 2) {
      fprintf(stderr, "Grid size is too large\n");
      exit(1);
    }
    printf(
        "\nSolving the Poisson equation on a %dx%d grid with %d processors\n\n",
        nx, nx, nprocs);
  }

  // MPI_Cart_create as per the first question
  int ndims =
      1; // Number of dimensions in the Cartesian topology; it is 1 for 1D
  int dims[1] = {
      nprocs}; // Number of processes in each dimension; since we only have one
               // dimension, we choose all processes here
  int periods[1] = {
      0}; // This is set to 0 to ensure non-periodic boundaries meaning that
          // processes at the edge have no neighbour in that direction
  int reorder = 1; // Optimisation flag allowing MPI to reorder process ranks
  int cart_rank;   // Process rank in the Cartesian communicator; note that this
                   // may differ from the original rank if reordering is
                   // enabled... I think
  MPI_Comm cart_comm; // Our communicator

  // Cartesian topology as per the first question
  MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &cart_comm);
  MPI_Comm_rank(cart_comm, &cart_rank);

  // Get the neighbouring processes replacing the manual calculations
  MPI_Cart_shift(cart_comm, 0, 1, &nbrleft, &nbrright);

  // Use MPI_Bcast to broadcast the grid size to all processes
  MPI_Bcast(&nx, 1, MPI_INT, 0, cart_comm);
  printf("cart_rank = %d has nx = %d\n", cart_rank, nx); // Debugging
  ny = nx;

  // Compute local domain bounds using a 1D decomposition
  MPE_Decomp1d(nx, nprocs, cart_rank, &s, &e);
  printf("cart_rank = %d is responsible for columns %d through %d, inclusive, "
         "with nbrleft (i.e., which process is to its left) = %d and nbrright "
         "(i.e., which process is to its right) = %d\n\n",
         cart_rank, s, e, nbrleft, nbrright);

  // Initialise grid with boundary conditions given in the second question
  init_oned(a, b, f, nx, ny, s, e);

  print_in_order(a, cart_comm); // Print the grid for debugging purposes

  // Start timing
  if (myid == 0) {
    printf("\nStarting iterative solver...\n");
  }
  t1 = MPI_Wtime(); // Start timing

  // Solution to the second question where we must solve the Poisson equation
  // using 1D decomposition
  glob_diff = 1000;
  for (it = 0; it < maxit; it++) {
    exchangi1(a, ny, s, e, cart_comm, nbrleft,
              nbrright); // Exchange ghost cells with neighbouring processes
    sweep1d(a, f, nx, s, e, b); // Perform a Jacobi sweep
    exchangi1(b, nx, s, e, cart_comm, nbrleft,
              nbrright); // Exchange ghost cells again for the next iteration
    sweep1d(b, f, nx, s, e, a); // Perform yet another Jacobi sweep

    // Check for convergence
    ldiff = griddiff(a, b, nx, s, e);
    MPI_Allreduce(&ldiff, &glob_diff, 1, MPI_DOUBLE, MPI_SUM, cart_comm);

    // Print progress every 100 iterations
    if (cart_rank == 0 && it % 100 == 0) {
      printf("For iteration %d, the global difference is %e\n", it, glob_diff);
    }

    // Break if convergence criteria is satisfied
    if (glob_diff < tol) {
      if (cart_rank == 0) {
        printf("Converged after %d iterations\n", it);
      }
      break;
    }
  }

  // Stop timing and report performance
  t2 = MPI_Wtime();
  if (cart_rank == 0) {
    if (it == maxit) {
      printf("Failed to converge after %d iterations\n", maxit);
    }
    printf("Run took %lf seconds\n", t2 - t1);
  }

  // Calculate the analytical solution for comparison
  double g[maxn][maxn]; // Array to store analytical solution values
  double h = 1.0 / ((double) (nx + 1)); // Grid spacing
  double x, y; // These are our x- and y-coordinates, respectively
  for (int i = 0; i <= nx + 1; i++) {
    for (int j = 0; j <= ny + 1; j++) {
      x = i * h; // Convert grid index to a physical x-coordinate
      y = j * h; // Convert grid index to a physical y-coordinate
      if (y == 0.0 || ((1.0 + x) * (1.0 + x) + y * y) ==
                          0.0) { // Protect against division by zero
        g[i][j] = 0.0;
      } else {

        // Analytical solution where u(x,y)=y/((1+x)^2+y^2)
        g[i][j] = y / ((1.0 + x) * (1.0 + x) + y * y);
      }
    }
  }

  // Write local grid to a file as per the requirements of the third question
  char local_filename[256];
  sprintf(local_filename, "local1dproc%d", cart_rank);
  write_grid(a, nx, ny, cart_rank, s, e, local_filename, 0);

  // Gather the distributed solution to the root process; this is required for
  // the third question
  int* s_vals = NULL;
  int* e_vals = NULL;
  if (cart_rank == 0) {
    s_vals = (int*) malloc(nprocs * sizeof(int));
    e_vals = (int*) malloc(nprocs * sizeof(int));
    if (!s_vals || !e_vals) {
      fprintf(stderr, "Memory allocation error\n");
      MPI_Abort(cart_comm, 1);
    }
  }

  // Gather domain decomposition information to the root process
  MPI_Gather(&s, 1, MPI_INT, s_vals, 1, MPI_INT, 0, cart_comm);
  MPI_Gather(&e, 1, MPI_INT, e_vals, 1, MPI_INT, 0, cart_comm);

  // Use GatherGrid to collect the solution from all the processes
  GatherGrid(global_grid, a, s, e, nx, ny, cart_rank, nprocs, s_vals, e_vals,
             cart_comm);

  // Write the global grid and analytical solution to files
  if (cart_rank == 0) {
    char global_filename[256];
    sprintf(global_filename, "global1dnx%d", nx);
    write_grid(global_grid, nx, ny, cart_rank, 1, nx, global_filename, 0);
    write_grid(g, nx, ny, cart_rank, 1, nx, "analytical", 0);
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

    // Print error statistics to demonstrate the accuracy of our solution as per
    // the requirements of the second question
    printf("\nResults for grid size %d x %d\n", nx, ny);
    printf("Maximum error: %e\n", max_error);
    printf("Average error: %e\n", avg_error);
  }

  // Clean up and finalise
  if (cart_rank == 0) {
    free(s_vals);
    free(e_vals);
  }
  MPI_Comm_free(&cart_comm);
  MPI_Finalize();

  return 0;
}
