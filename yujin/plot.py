import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
import os


def plot_solution(nx=31):
    """
    Plot the Poisson equation solution from the generated data file.

    Parameters:
    -----------
    nx : int
        Grid size (default: 31)
    """
    filename = f"poisson_global_sol_nx{nx}"

    # Check if the file exists
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found.")
        print("Make sure to run the MPI program first to generate the solution file.")
        return

    # Load data from file
    data = []
    x_vals = []
    y_vals = []
    z_vals = []

    with open(filename, "r") as f:
        lines = f.readlines()
        # Skip header lines
        data_lines = [
            line.strip() for line in lines if not line.startswith("#") and line.strip()
        ]

        for line in data_lines:
            if line.strip():
                try:
                    x, y, z = map(float, line.split())
                    x_vals.append(x)
                    y_vals.append(y)
                    z_vals.append(z)
                except ValueError:
                    continue

    if not x_vals:
        print("Error: No valid data found in the file.")
        return

    # Convert to numpy arrays
    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    z_vals = np.array(z_vals)

    # Determine the grid size from the data
    unique_x = len(np.unique(x_vals))
    unique_y = len(np.unique(y_vals))
    grid_size = unique_x

    # Check if we can reshape correctly
    if len(x_vals) != grid_size * grid_size:
        # Try to handle files with blank lines between rows
        points_per_row = 0
        current_row = y_vals[0]
        for y in y_vals:
            if y == current_row:
                points_per_row += 1
            else:
                break

        grid_size = points_per_row

        # Create a proper grid
        x_grid = np.zeros((grid_size, grid_size))
        y_grid = np.zeros((grid_size, grid_size))
        z_grid = np.zeros((grid_size, grid_size))

        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                if idx < len(x_vals):
                    x_grid[grid_size - 1 - i, j] = x_vals[idx]
                    y_grid[grid_size - 1 - i, j] = y_vals[idx]
                    z_grid[grid_size - 1 - i, j] = z_vals[idx]
    else:
        # Reshape the data into a grid
        x_grid = x_vals.reshape(grid_size, grid_size)
        y_grid = y_vals.reshape(grid_size, grid_size)
        z_grid = z_vals.reshape(grid_size, grid_size)

    # Create heatmap with correct value range (0 to 0.5)
    plt.figure(figsize=(10, 8))
    plt.contourf(x_grid, y_grid, z_grid, 50, cmap=cm.viridis, vmin=0, vmax=0.5)
    plt.colorbar(label="u(x,y)")
    plt.title(f"Poisson Equation Solution Heatmap (nx={nx})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(f"poisson_heatmap_nx{nx}.png", dpi=300)
    print(f"Saved heatmap to poisson_heatmap_nx{nx}.png")
    plt.show()

    # Create 3D surface plot
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        x_grid, y_grid, z_grid, cmap=cm.viridis, vmin=0, vmax=0.5, edgecolor="none"
    )
    ax.set_title(f"Poisson Equation Solution 3D Surface (nx={nx})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u(x,y)")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label="u(x,y)")
    plt.savefig(f"poisson_surface_nx{nx}.png", dpi=300)
    print(f"Saved 3D surface plot to poisson_surface_nx{nx}.png")
    plt.show()

    # Calculate and display analytical solution error
    h = 1.0 / (grid_size - 1)
    analytical_errors = []

    for i in range(grid_size):
        for j in range(grid_size):
            x = x_grid[i, j]
            y = y_grid[i, j]
            z_numerical = z_grid[i, j]

            if 0 <= x <= 1 and 0 <= y <= 1:
                z_analytical = y / ((1.0 + x) ** 2 + y**2)
                error = abs(z_numerical - z_analytical)
                analytical_errors.append(error)

    if analytical_errors:
        max_error = max(analytical_errors)
        print(
            f"\nMaximum error between numerical and analytical solution: {max_error:.6e}"
        )

    # Create error visualization (optional)
    if analytical_errors:
        plt.figure(figsize=(10, 8))
        plt.hist(analytical_errors, bins=50)
        plt.title(f"Error Distribution (nx={nx})")
        plt.xlabel("Absolute Error")
        plt.ylabel("Frequency")
        plt.savefig(f"poisson_error_hist_nx{nx}.png", dpi=300)
        print(f"Saved error histogram to poisson_error_hist_nx{nx}.png")
        plt.show()


if __name__ == "__main__":
    # Allow specifying grid size from command line
    nx = 31  # Default grid size
    if len(sys.argv) > 1:
        try:
            nx = int(sys.argv[1])
        except ValueError:
            print("Error: Grid size must be an integer.")
            sys.exit(1)

    plot_solution(nx)
