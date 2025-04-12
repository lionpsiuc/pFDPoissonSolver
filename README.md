# High-Performance Computing Software II <!-- omit from toc -->

## Assignment 1 - 2D Processor Decomposition Poisson Solver <!-- omit from toc -->

- [Mathematical Background](#mathematical-background)
- [Folder Structure and Usage Details](#folder-structure-and-usage-details)
  - [1D Implementation](#1d-implementation)
  - [1D Implementation](#1d-implementation-1)
- [How to Run](#how-to-run)
  - [1D Implementation](#1d-implementation-2)
  - [2D Implementation](#2d-implementation)

### Mathematical Background

This project implements a parallel solver for the 2D Poisson equation using the finite difference method. The Poisson equation is a partial differential equation of the form

```math
\nabla^2u(x,y)=\frac{\partial^2u(x,y)}{\partial x^2}+\frac{\partial^2u(x,y)}{\partial y^2}=f(x,y),\quad0\leq x,y \leq 1.
```

We solve this equation on the region $\Omega=[0,1]\times[0,1]$ on $\partial\Omega$ (i.e., Dirichlet boundary conditions).

The implementation uses the Jacobi iterative method to solve the discrete form of this equation on a uniform grid, distributing the computation across multiple processors using MPI. Two domain decomposition approaches are implemented:

- **1D Decomposition**: The domain is divided into column-wise strips, with each processor handling a set of adjacent columns.
- **2D Decomposition**: The domain is divided into rectangular blocks, with each processor handling a two-dimensional portion of the grid.

### Folder Structure and Usage Details

The repository is organised to address the requirements of the assignment with separate directories for 1D and 2D implementations.

#### 1D Implementation

Located in `1d`, where this folder contains the C code required for the assignment, addressing the first, second, and third question. It is organised as follows:

- `src/main.c`: The main driver programme implementing the 1D parallel Poisson solver.
- `src/aux.c`: Utility functions for grid initialisation.
- `src/jacobi.c`: Implementation of the Jacobi iteration method, including ghost cell exchange.
- `src/gatherwrite.c`: Functions for gathering distributed grid data and writing output.
- `src/decomp1d.c`: Domain decomposition utilities for dividing the grid among processors.
- `include/`: Header files defining function prototypes and constants.
- `scripts/heatmap.gp`: Script for generating heatmap visualisations.
- `Makefile`: Automates the compilation process. See [here](#how-to-run) for details on how to compile and run.

#### 1D Implementation

Located in `2d`, where this folder contains the C code required for the assignment, addressing the fourth question. It is organised as follows:

- `src/main.c`: The main driver programme implementing the 2D parallel Poisson solver.
- `src/aux.c`: Utility functions for grid initialisation.
- `src/jacobi.c`: Implementation of the Jacobi iteration method, including functions for ghost cell exchange in two directions.
- `src/gatherwrite.c`: Functions for gathering distributed 2D grid data and writing output.
- `src/decomp2d.c`: Two-dimensional domain decomposition utilities for dividing the grid among processors.
- `include/`: Header files defining function prototypes and constants.
- `scripts/heatmap.gp`: Script for generating heatmap visualisations.
- `Makefile`: Automates the compilation process. See [here](#how-to-run) for details on how to compile and run.

### How to Run

#### 1D Implementation

1. **Compilation**: Navigate to the `1d` folder and run:

    ```bash
    make
    ```

    This will compile the source files and generate the executable `bin/poiss1d`.

2. **Execution**: To run the solver with four processors (and a grid size of `nx = 31`):

    ```bash
    make run
    ```

    Also, it can be ran manually, where the user can specify a grid size themselves. Please note, however, that `include/poisson1d.h` contains `#define maxn 31 + 2` which means that grids, regardless of what grid size is specified when ran, will be initialised to have `maxn` rows and columns. To change this, change `31` to the required grid size (e.g., `15`), run `make clean`, compile once more as before, and ensure to include the specified grid size in the run command (i.e., `mpirun -np 4 bin/poiss1d 15`) since it defaults to `31` if a grid size is not specified. Thus, to run manually, using a grid size of `15` and four processors, run the following (after having made the necessary changes to `header/poisson1d.h`):

    ```bash
    mpirun -np 4 bin/poiss1d 15
    ```

3. **Visualisation**: After running the solver (e.g., using `nprocs = 4` and `nx = 31`), generate a heatmap of the solution:

    ```bash
    make heatmap
    ```

    This will generate a file called `heatmapnprocs4nx31.png`.

    > It is important to note that `global1dnprocs4nx31.txt` and `analyticalnprocs4nx31.txt` (the gathered global grid for a grid size of `nx = 31` using `nprocs = 4` processors and analytical solution for the same parameters) are hardcoded into the plotting function (i.e., `scripts/heatmap.gp`), as are the lengths of the $x$- and $y$-axes. To generate a heatmap for a different grid size, please change `global1dnprocs4nx31.txt` and `analyticalnprocs4nx31.txt` to the respective files (e.g., `global1dnprocs4nx15.txt` and `analyticalnprocs4nx15.txt`), and don't forget the $x$ and $y$ values too. Also, for consistency, change the title too (i.e., `heatmapnprocs4nx15.png`, for the previous example).

4. **Cleaning**: To remove all compiled files and output:

    ```bash
    make clean
    ```

#### 2D Implementation

1. **Compilation**: Navigate to the `2d` folder and run:

    ```bash
    make
    ```

    This will compile the source files and generate the executable `bin/poiss2d`.

2. **Execution**: To run the solver with four processors (and a grid size of `nx = 31`):

    ```bash
    make run4
    ```

    Also, it can be ran manually, where the user can specify a grid size themselves. Please note, however, that `include/poisson2d.h` contains `#define maxn 31 + 2` which means that grids, regardless of what grid size is specified when ran, will be initialised to have `maxn` rows and columns. To change this, change `31` to the required grid size (e.g., `15`), run `make clean`, compile once more as before, and ensure to include the specified grid size in the run command (i.e., `mpirun -np 4 bin/poiss2d 15`) since it defaults to `31` if a grid size is not specified. Thus, to run manually, using a grid size of `15` and four processors, run the following (after having made the necessary changes to `header/poisson2d.h`):

    ```bash
    mpirun -np 4 bin/poiss2d 15
    ```

    Moreover, there is an option to run with 16 processors (and a grid size of `nx = 31`):
    
    ```bash
    make run16
    ```

3. **Visualisation**: After running the solver, generate a heatmap of the solution:

    ```bash
    make heatmap
    ```

    This will generate a file called `heatmapnprocs4nx31.png`.

    > It is important to note that `global2dnprocs4nx31.txt` and `analyticalnprocs4nx31.txt` (the gathered global grid for a grid size of `nx = 31` using `nprocs = 4` processors and analytical solution for the same parameters) are hardcoded into the plotting function (i.e., `scripts/heatmap.gp`), as are the lengths of the $x$- and $y$-axes. To generate a heatmap for a different grid size, please change `global2dnprocs4nx31.txt` and `analyticalnprocs4nx31.txt` to the respective files (e.g., `global2dnprocs4nx15.txt` and `analyticalnprocs4nx15.txt`), and don't forget the $x$ and $y$ values too. Also, for consistency, change the title too (i.e., `heatmapnprocs4nx15.png`, for the previous example).

4. **Cleaning**: To remove all compiled files and output:

    ```bash
    make clean
    ```
