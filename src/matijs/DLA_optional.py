import numpy as np
from numba import prange, njit, autojit
import time

@autojit
def sor_2step(c_k, sink_mask, omega=1):
    """
    Steps grid using SOR twice in either direction in random order.
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
        
    LR = np.random.choice(np.array([0,1]))
    if LR == 1:
        for i in prange(0, N, 1):
            for j in range(M-1, 0, -1):
                # Skip sinks, cause concentration loss in other cells (TODO explore setting sink concentration to a constant)
                if sink_mask[i, j] == 1:
                    c_k[i, j] = 0
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
        for i in prange(0, N, 1):
            for j in range(0, M):
                # Skip sinks, cause concentration loss in other cells (TODO explore setting sink concentration to a constant)
                if sink_mask[i, j] == 1:
                    c_k[i, j] = 0
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

    else:
        for i in prange(0, N, 1):
            for j in range(0, M):
                # Skip sinks, cause concentration loss in other cells (TODO explore setting sink concentration to a constant)
                if sink_mask[i, j] == 1:
                    c_k[i, j] = 0
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

        for i in prange(0, N, 1):
            for j in range(M-1, 0, -1):
                # Skip sinks, cause concentration loss in other cells (TODO explore setting sink concentration to a constant)
                if sink_mask[i, j] == 1:
                    c_k[i, j] = 0
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
    
    return c_k

""" @njit(parallel=True)
def find_adjacents(grid):
    N, M = grid.shape
    for j in prange(M):
        adjacents = [] # use set to avoid re-adding same adjacent cells
        for i in range(N):
            if grid[i, j] == 1:
                # If not top or bottom check top/bottom adjacents
                if i != 0 and i!=N-1:
                    if grid[i+1, j] == 0:
                        adjacents.append((i+1, j))
                    if grid[i-1, j] == 0:
                        adjacents.append((i-1, j))
                # Check side adjacents and periodic border boundaries
                if grid[i, (j-1) % M] == 0:
                    adjacents.append((i, (j-1) % M))
                if grid[i, (j+1) % M] == 0:
                    adjacents.append((i, (j+1) % M))
        
    return list(set(adjacents)) """

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

def dla(N, M, eta, iterations):
    # N = ROWS
    # M = COLS
    
    # Initialize material grid with center seed at bottom
    material_grid = np.zeros((N,M), dtype=int)
    material_grid[1,M//2] = 1

    # Initialize concentration grid with straight-line gradient
    concentration_grid = np.zeros((N,M), dtype=float)
    for i in range(N):
        concentration_grid[i, :] = i / (N - 1)
    
    material_results = []
    concentration_results = []
    #material_results.append(concentration_grid.copy())
    
    
    start_time = time.time()
    
    for i in range(iterations):
        candidates = find_adjacents(material_grid)
        # Process neighbors and pick one in material grid
        assert len(candidates) > 0 & i < iterations - 1, "No more candidates found."
        if candidates:
            probabilities = [(concentration_grid[i, j]**eta) for i, j in candidates]
            total_probabilites = sum(probabilities)
            
            if total_probabilites > 0:
                normalized_probabilities = probabilities / total_probabilites
                chosen_index = np.random.choice(len(candidates), p=normalized_probabilities)
                chosen_neighbor = candidates[chosen_index]
                assert material_grid[chosen_neighbor] == 0, "Already-material cell picked."
                material_grid[chosen_neighbor] = 1
        
        concentration_results.append(concentration_grid.copy())
        concentration_grid = sor_2step(concentration_grid, material_grid)
        material_results.append(material_grid.copy())
        
        print(f'Completed Simulation step {i+1}/{iterations}', end='\r')
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal simulation time: {total_time}s")
    print(f"Per step: {total_time/iterations}")
    
    return material_results, concentration_results


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    
    N_STEPS = 40
    N_FRAMES = min(40, N_STEPS)
    
    material_results, concentration_results = dla(1000, 1000, 1.5, N_STEPS)
    print('results created, plotting...')
    fig, ax = plt.subplots(figsize=(6, 4))
    cax = ax.imshow(material_results[-1], cmap="hot", aspect="auto", origin="lower", extent=[0, 1, 1, 0])
    fig.colorbar(cax, label="Concentration")
    ax.set_title('Concentration $c(x,y)')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()
    
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
    
    frames = np.linspace(0, len(material_results)-1, N_FRAMES).astype(int)
    
    def update_frame(frame):
        global N_FRAMES, N_STEPS
        step = int(frame * N_STEPS / N_FRAMES)
        print(f'frame {frame}/{N_FRAMES} | Simulation step: {step}/{N_STEPS}', end='\r')
        index = frames[frame]
        cax1.set_array(material_results[index])
        cax2.set_array(concentration_results[index])
        return cax1, cax2

    ani = animation.FuncAnimation(fig, update_frame, frames=N_FRAMES, interval=50, blit=False)

    ani.save('structure_evolution.mp4', writer='ffmpeg')
    plt.show()