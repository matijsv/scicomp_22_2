import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import src.noa.metrics as metrics
import numpy as np
import os

@njit
def simulate_DLA_numba(height, width, num_particles, ps, max_steps=1000000):
    """
    Monte Carlo DLA simulation with a max step limit, ensuring walkers move 
    until they find an unoccupied cell if they encounter a boundary.

    Parameters:
      height, width: Grid size.
      num_particles: Total number of particles in the cluster.
      ps: Sticking probability.
      max_steps: Maximum steps a walker can take before being abandoned.

    Returns:
      grid: 2D numpy array with the cluster.
    """

    grid = np.zeros((height, width), dtype=np.int32)
    
    # Seed the initial cluster at the bottom center
    seed_y, seed_x = height - 1, width // 2
    grid[seed_y, seed_x] = 1
    particles_added = 1

    moves = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=np.int32)

    while particles_added < num_particles:
        x = np.random.randint(0, width)
        y = 0
        step_count = 0  # Track walker steps

        while step_count < max_steps:
            # Choose a random move until it lands on a valid, unoccupied cell
            for _ in range(10):  # Try a max of 10 random moves to find a valid one
                move_index = np.random.randint(0, 4)
                dx, dy = moves[move_index]
                new_x = (x + dx) % width  # Periodic boundary horizontally
                new_y = y + dy

                # If walker moves out of the vertical bounds, reset it at the top
                if new_y < 0 or new_y >= height:
                    x = np.random.randint(0, width)
                    y = 0
                    step_count = 0
                    break  # Restart walker

                # Ensure the walker does not move into an occupied cell
                if grid[new_y, new_x] == 0:
                    x, y = new_x, new_y
                    step_count += 1
                    break  # Found a valid move, break out of retry loop

            # Check if walker is adjacent to the cluster
            for i in range(4):
                nx = (x + moves[i, 0]) % width
                ny = y + moves[i, 1]
                if 0 <= ny < height and grid[ny, nx] == 1:
                    if np.random.random() < ps:  # Sticking probability
                        grid[y, x] = 1
                        particles_added += 1
                        step_count = max_steps  # Force exit from walker loop
                    break  # Stop checking neighbors

    return grid

def generate_cluster_data(height,width,num_particles,ps, N):

    for ps_i in ps:
        for i in range(N):
            cluster = simulate_DLA_numba(height, width, num_particles, ps_i)
            np.save(f"cluster_data/cluster_{ps_i}_{i}.npy", cluster)

def data_analysis(ps, N):
    fractal_dims = []
    fractal_dims_std = []
    rgs = []
    rgs_std = []

    for ps_i in ps:
        fractal_dims_per_ps = []
        rgs_per_ps = []

        for i in range(N):
            grid = np.load(f"cluster_data/cluster_{ps_i}_{i}.npy") 
            fractal_dims_per_ps.append(metrics.fractal_dimension(grid))
            # r_values, density_values = radial_distribution(grid1)
            rgs_per_ps.append(metrics.radius_of_gyration(grid))

        fractal_dims.append(np.mean(fractal_dims_per_ps))
        fractal_dims_std.append(np.std(fractal_dims_per_ps))
        rgs.append(np.mean(rgs_per_ps))
        rgs_std.append(np.std(rgs_per_ps))

        fractal_dims = list(map(float, fractal_dims))
        fractal_dims_std = list(map(float, fractal_dims_std))
        rgs = list(map(float, rgs))
        rgs_std = list(map(float, rgs_std))

    return rgs, rgs_std, fractal_dims, fractal_dims_std

def plot_data_analysis(ps, rgs, rgs_std, fractal_dims, fractal_dims_std):

    # Create figure and axes
    fig, ax1 = plt.subplots(figsize=(4, 3))

    # Plot first set (x_values with error bars)
    ax1.errorbar(ps, fractal_dims, yerr=fractal_dims_std, fmt='-o', label="Fractal Dimension", capsize=2, markersize=5)
    ax1.set_xlabel(r"$p_s$")
    ax1.set_ylabel("Fractal Dimension")
    ax1.tick_params(axis='y')
    ax1.set_xlim(0,1.02)
    # Title and layout adjustments
    # plt.title("Fractal Dimension and Radius of Gyration with Error Bars")
    fig.tight_layout()

    # Show plot
    plt.show()
    

def plot_dla(height, width, num_particles):
    """
    Simulate and plot DLA clusters for different sticking probabilities.

    Parameters:
    height, width: Grid dimensions.
    num_particles: Number of particles in the simulation.
    """

    # Define sticking probabilities to test
    sticking_probs = [1.0, 0.5, 0.2, 0.05]
    cluster_data = {}

    # Create directory for saving cluster data
    os.makedirs("cluster_data", exist_ok=True)

    # Run the simulation for each sticking probability and save results
    for ps in sticking_probs:
        cluster = simulate_DLA_numba(height, width, num_particles, ps)
        file_path = f"cluster_data/cluster_ps{ps:.2f}.npy"
        np.save(file_path, cluster)
        cluster_data[ps] = cluster  # Store for plotting

    # Plot the resulting clusters
    fig, axes = plt.subplots(1, len(sticking_probs), figsize=(20, 5))

    for ax, (ps, cluster) in zip(axes, cluster_data.items()):
        ax.imshow(cluster, cmap='hot', interpolation='nearest')
        ax.set_title(f'DLA, p_s = {ps:.2f}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    height, width = 100, 100
    num_particles = 800
    # plot_dla(height,width,num_particles)
    ps = [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    N = 25
    # generate_cluster_data(height,width,num_particles,ps, N)
    rgs, rgs_std, fractal_dims, fractal_dims_std = data_analysis(ps,N)
    plot_data_analysis(ps, rgs, rgs_std, fractal_dims, fractal_dims_std)