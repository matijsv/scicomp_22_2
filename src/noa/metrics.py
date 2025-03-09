import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import os


import numpy as np
import matplotlib.pyplot as plt

def box_count(data, box_size):
    """
    Counts the number of boxes of size `box_size` that contain at least one nonzero pixel.
    """
    num_boxes = 0
    for i in range(0, data.shape[0], box_size):
        for j in range(0, data.shape[1], box_size):
            if np.any(data[i:i+box_size, j:j+box_size]):
                num_boxes += 1
    return num_boxes

def fractal_dimension(data, min_box_size=1, max_box_size=None):
    """
    Computes the fractal dimension of a 2D numpy array using the box-counting method.
    """
    if max_box_size is None:
        max_box_size = min(data.shape) // 2  # Maximum box size should be at most half of the smallest dimension

    sizes = []
    counts = []

    box_size = min_box_size
    while box_size <= max_box_size:
        sizes.append(box_size)
        counts.append(box_count(data, box_size))
        box_size *= 2  # Increase box size exponentially

    # Fit log-log plot to estimate the fractal dimension
    sizes = np.array(sizes)
    counts = np.array(counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    
    return -coeffs[0]  # The fractal dimension is the negative slope

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