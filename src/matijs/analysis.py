import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool

from src.matijs.DLA import dla

def plot_end(N, M, eta, omega, growth_iterations, max_sor_iterations=1000, epsilon=1e-5, material_conc = 0):
    material_results, concentration_results, sor_per_step = dla(N, M, eta, omega, growth_iterations, max_sor_iterations, epsilon, material_conc)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=300)
    
    cax1 = ax1.imshow(material_results[-1], cmap="hot", aspect="auto", origin="lower", extent=[0, 1, 1, 0])
    ax1.set_xlabel("x", fontsize=20)
    ax1.set_ylabel("y", fontsize=20)
    
    cax2 = ax2.imshow(concentration_results[-1], cmap="hot", aspect="auto", origin="lower", extent=[0, 1, 1, 0])
    fig.colorbar(cax2, ax=ax2, label="Concentration")
    ax2.set_xlabel("x", fontsize=20)
    ax2.set_ylabel("y", fontsize=20)
    
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()

def trial_omega(args):
    N, M, eta, omega, material_conc = args
    _, _, sor_per_step = dla(N, M, eta, omega, growth_iterations=100, max_sor_iterations=1000, epsilon=1e-5, material_conc=material_conc)
    mean_iterations = np.mean(sor_per_step)
    return omega, mean_iterations

def optimize_omega(N, M, eta, material_conc=0):
    omegas = np.linspace(1.75, 1.9, 25)  # Test omegas between 1.6 and 1.9
    args = [(N, M, eta, omega, material_conc) for omega in omegas]
    with Pool() as pool:
        results = pool.map(trial_omega, args)

    best_omega, best_mean_iterations = min(results, key=lambda x: x[1])
    #print(f"Best omega: {best_omega}, with mean iterations: {best_mean_iterations}")
    return best_omega, best_mean_iterations
    
def plot_best_omega_vs_eta(N, M, etas, material_conc=0):
    best_omegas = []
    for eta in etas:
        best_omega, _ = optimize_omega(N, M, eta, material_conc)
        best_omegas.append(best_omega)
    
    plt.figure(figsize=(10, 6))
    plt.plot(etas, best_omegas, marker='o')
    plt.xlabel('Eta')
    plt.ylabel('Best Omega')
    plt.title('Best Omega vs Eta')
    plt.grid(True)
    plt.show()
    plt.savefig('/fig/optimize_omega.png')