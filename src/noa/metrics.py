import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import os

def fractal_dimension(grid, sizes=np.logspace(1, 2, num=10, base=2, dtype=int)):
    """
    Compute the fractal dimension using the box-counting method.
    
    Parameters:
      grid: 2D numpy array representing the cluster.
      sizes: Box sizes to use for counting occupied boxes.
    
    Returns:
      float: estimated fractal dimension.
    """

    nonzero_points = np.column_stack(np.nonzero(grid))  # Extract occupied sites
    counts = []
    
    for size in sizes:
        if size > min(grid.shape):
            continue
        # Count number of boxes that contain at least one occupied pixel
        grid_coarse = grid[::size, ::size]  # Downsample
        count = np.sum(grid_coarse > 0)
        counts.append(count)
    
    # Fit line to log-log plot
    log_sizes = np.log(1 / sizes[:len(counts)])
    log_counts = np.log(counts)
    coeffs = np.polyfit(log_sizes, log_counts, 1)
    return -coeffs[0]  # Fractal dimension is the negative slope


def radial_distribution(grid):
    """
    Compute the radial density distribution of the cluster.
    
    Parameters:
      grid: 2D numpy array representing the cluster.
    
    Returns:
      radii (array): Distance bins.
      density (array): Cluster density as function of radius.
    """
    center = np.array([grid.shape[0] - 1, grid.shape[1] // 2])  # Seed location
    occupied_points = np.column_stack(np.nonzero(grid))  # Cluster points
    distances = np.linalg.norm(occupied_points - center, axis=1)
    
    max_radius = np.max(distances)
    bins = np.linspace(0, max_radius, 30)
    density, _ = np.histogram(distances, bins=bins, density=True)
    return bins[:-1], density


def radius_of_gyration(grid):
    """
    Compute the compactness of the cluster using the radius of gyration.
    
    Parameters:
      grid: 2D numpy array representing the cluster.
    
    Returns:
      float: radius of gyration.
    """
    occupied_points = np.column_stack(np.nonzero(grid))  # Cluster points
    center_of_mass = np.mean(occupied_points, axis=0)
    distances = np.linalg.norm(occupied_points - center_of_mass, axis=1)
    return np.sqrt(np.mean(distances**2))  # Radius of gyration


if __name__ == "__main__":
    # Example usage: load a cluster and compute metrics
    import matplotlib.pyplot as plt
    
    grid1 = np.load("cluster_data/cluster_ps1.npy") 
    
    # Compute metrics
    fractal_dim = fractal_dimension(grid1)
    r_values, density_values = radial_distribution(grid1)
    rg = radius_of_gyration(grid1)
    
    print(f"Fractal Dimension: {fractal_dim:.3f}")
    print(f"Radius of Gyration: {rg:.3f}")
    
    grid = np.load("cluster_data/cluster_ps02.npy") 
    
    # Compute metrics
    fractal_dim = fractal_dimension(grid)
    r_values, density_values = radial_distribution(grid)
    rg = radius_of_gyration(grid)
    
    print(f"Fractal Dimension: {fractal_dim:.3f}")
    print(f"Radius of Gyration: {rg:.3f}")

    grid = np.load("cluster_data/cluster_ps05.npy") 
    
    # Compute metrics
    fractal_dim = fractal_dimension(grid)
    r_values, density_values = radial_distribution(grid)
    rg = radius_of_gyration(grid)
    
    print(f"Fractal Dimension: {fractal_dim:.3f}")
    print(f"Radius of Gyration: {rg:.3f}")

    # Plot radial distribution
    #plt.figure(figsize=(6, 4))
    #plt.plot(r_values, density_values, marker='o')
    #plt.xlabel("Radius")
    #plt.ylabel("Density")
    #plt.title("Radial Distribution")
    #plt.show()