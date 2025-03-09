""" Maintains boundary conditions and fully converges SOR between every growth step"""

import numpy as np
from numba import prange, njit
import time
import numba.cuda

# calculate the maximum difference between two arrays
@njit(parallel=True)
def compute_max_diff(c_k, c_kp1):
    rows, cols = c_k.shape
    local_max = np.zeros(rows - 2)  # local max for each thread
    for i in prange(1, rows - 1):
        max_val = 0.0
        for j in range(1, cols - 1):
            diff = abs(c_kp1[i, j] - c_k[i, j])
            if diff > max_val:
                max_val = diff
        local_max[i - 1] = max_val  # store the local max for each thread
    return np.max(local_max)  # return the global max

@njit(parallel=True)
def sor_step(c_k, sink_mask, omega=1, material_conc=0):
    """
    Steps grid using SOR, direction is randomized.
    Stochastically symmetric.

    Parameters
    ----------
    c_k : _type_
        _description_
    sink_mask : _type_
        _description_
    omega : float, optional
        _description_, by default 0.9

    Returns
    -------
    _type_
        _description_
    """
    
    N, M = c_k.shape
    def leftward_update():
        for i in prange(1, N-1):
            for j in range(M-1, 0, -1):
                # Material concentration is set to a constant (TODO explore material as an isolator)
                if sink_mask[i, j] == 1:
                    c_k[i, j] = material_conc
                    continue
                # Treats top and bottom boundaries as perfect insulators
                if i == 0:
                    c_k[i, j] = omega / 4.0 * (c_k[i+1, j] + c_k[i, j] + c_k[i, (j+1) % M] + c_k[i, (j-1) % M]) + (1 - omega) * c_k[i, j]
                    continue
                if i == N-1:
                    c_k[i, j] = omega / 4.0 * (c_k[i, j] + c_k[i-1, j] + c_k[i, (j+1) % M] + c_k[i, (j-1) % M]) + (1 - omega) * c_k[i, j]
                    continue
                # Side boundaries are periodic, as per usual
                c_k[i, j] = omega / 4.0 * (c_k[i+1, j] + c_k[i-1, j] + c_k[i, (j+1) % M] + c_k[i, (j-1) % M]) + (1 - omega) * c_k[i, j]
                #if c_k[i,j] < 0 : print(c_k[i,j])
                #assert c_k[i, j] >= 0, f'concentration non-positive at {i}, {j}: {float(c_k[i, j])}'
    
    def rightward_update():
        for i in prange(1, N-1):
            for j in range(0, M):
                # Skip sinks, cause concentration loss in other cells (TODO explore setting sink concentration to a constant)
                if sink_mask[i, j] == 1:
                    c_k[i, j] = material_conc
                    continue
                # Treats top and bottom boundaries as perfect insulators
                if i == 0:
                    c_k[i, j] = omega / 4.0 * (c_k[i+1, j] + c_k[i, j] + c_k[i, (j+1) % M] + c_k[i, (j-1) % M]) + (1 - omega) * c_k[i, j]
                    continue
                if i == N-1:
                    c_k[i, j] = omega / 4.0 * (c_k[i, j] + c_k[i-1, j] + c_k[i, (j+1) % M] + c_k[i, (j-1) % M]) + (1 - omega) * c_k[i, j]
                    continue
                # Side boundaries are periodic, as per usual
                c_k[i, j] = omega / 4.0 * (c_k[i+1, j] + c_k[i-1, j] + c_k[i, (j+1) % M] + c_k[i, (j-1) % M]) + (1 - omega) * c_k[i, j]
                #if c_k[i,j] < 0 : print(c_k[i,j])
                #assert c_k[i, j] >= 0, f'concentration non-positive at {i}, {j}: {float(c_k[i, j])}'
    LR = np.random.choice(np.array([0,1]))
    if LR == 1:
        leftward_update()
    else:
        rightward_update()
    
    return c_k

@njit
def find_adjacents(grid):
    N, M = grid.shape
    adjacents = set() # use set to avoid re-adding same adjacent cells
    for j in range(M):
        for i in range(N):
            if grid[i, j] == 1:
                # If not top or bottom check top/bottom adjacents
                if i != 0 and i!=N-1:
                    if grid[i+1, j] == 0:
                        adjacents.add((i+1, j))
                    if grid[i-1, j] == 0:
                        adjacents.add((i-1, j))
                # Check side adjacents and periodic border boundaries
                if grid[i, (j-1) % M] == 0:
                    adjacents.add((i, (j-1) % M))
                if grid[i, (j+1) % M] == 0:
                    adjacents.add((i, (j+1) % M))
    return list(adjacents)

def dla(N, M, eta, iterations, max_sor_iterations=1000, epsilon=1e-5):
    # N = ROWS
    # M = COLS
    
    # Initialize material grid with center seed at bottom
    material_grid = np.zeros((N,M), dtype=int)
    material_grid[1,M//2] = 1

    # Initialize concentration grid with straight-line gradient
    concentration_grid = np.zeros((N,M), dtype=float)
    #concentration_grid[-1,:] = 1
    #concentration_grid[0,:] = 1
    for i in range(N):
        concentration_grid[i, :] = i / (N - 1)
    
    material_results = []
    concentration_results = []
    #material_results.append(concentration_grid.copy())
    start_time = time.time()
    for i in range(iterations):
        candidates = find_adjacents(material_grid)
        
        # Process neighbors and pick one in material grid
        #assert len(candidates) > 0 & i < iterations - 1, "No more candidates found."
        if candidates:
            probabilities = [(concentration_grid[i, j]**eta) for i, j in candidates]
            total_probabilites = sum(probabilities)
            
            if total_probabilites > 0:
                normalized_probabilities = probabilities / total_probabilites
                chosen_index = np.random.choice(len(candidates), p=normalized_probabilities)
                chosen_neighbor = candidates[chosen_index]
                assert material_grid[chosen_neighbor] == 0, "Already-material cell picked."
                material_grid[chosen_neighbor] = 1
            elif total_probabilites == 0:
                chosen_index = np.random.choice(len(candidates))
                chosen_neighbor = candidates[chosen_index]
                assert material_grid[chosen_neighbor] == 0, "Already-material cell picked."
                material_grid[chosen_neighbor] = 1
                
        
        # SOR STUFF 
        concentration_results.append(concentration_grid.copy())
        material_results.append(material_grid.copy())
        
        for sor_i in range(max_sor_iterations):
            
            concentration_grid_new = sor_step(concentration_grid.copy(), material_grid)
            delta = compute_max_diff(concentration_grid, concentration_grid_new)
            if delta < epsilon:
                break
            
            concentration_grid = concentration_grid_new
        assert sor_i < max_sor_iterations, "Maximum SOR iterations were surpassed, increase amount and/or check SOR omega param."
        
        print(f'Completed Simulation step {i}/{iterations}', end='\r')
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal simulation time: {total_time}s")
    print(f"Per step: {total_time/iterations}")
    return material_results, concentration_results


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    
    N_STEPS = 800
    material_results, concentration_results = dla(100, 100, 1, N_STEPS)
    print('results created, plotting...')
    """ fig, ax = plt.subplots(figsize=(6, 4))
    cax = ax.imshow(material_results[0], cmap="hot", aspect="auto", origin="lower", extent=[0, 1, 1, 0])
    fig.colorbar(cax, label="Concentration")
    ax.set_title('Concentration $c(x,y)')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show() """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    cax1 = ax1.imshow(material_results[0], cmap="hot", aspect="auto", origin="lower", extent=[0, 1, 1, 0])
    fig.colorbar(cax1, ax=ax1, label="Material")
    ax1.set_title('Material Grid')
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    
    cax2 = ax2.imshow(concentration_results[0], cmap="hot", aspect="auto", origin="lower", extent=[0, 1, 1, 0])
    fig.colorbar(cax2, ax=ax2, label="Concentration")
    ax2.set_title('Concentration Grid')
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    
    N_FRAMES = 20
    frames = np.linspace(0, len(material_results)-1, N_FRAMES).astype(int)

    def update_frame(frame):
        global N_FRAMES, N_STEPS
        step = int(frame * N_STEPS / N_FRAMES)
        print(f'\r frame {frame}/{N_FRAMES} | Simulation step: {step}/{N_STEPS}', end='\r')
        index = frames[frame]
        cax1.set_array(material_results[index])
        cax2.set_array(concentration_results[index])
        return cax1, cax2

    ani = animation.FuncAnimation(fig, update_frame, frames=N_FRAMES, interval=50, blit=False)

    ani.save('structure_evolution.mp4', writer='ffmpeg')
    plt.show()