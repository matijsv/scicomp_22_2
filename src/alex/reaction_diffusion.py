import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# parameters
size = 100  # spatial size
D_u, D_v = 0.16, 0.08  # diffusion rates
f, k = 0.035, 0.060  # reaction rates
dt, dx = 1.0, 1.0  # time step, space step

# initialize U, V
U = np.ones((size, size)) * 0.5
V = np.zeros((size, size))

# center v initial condition
r = 10  # radius of the circle
center = size // 2
V[center-r:center+r, center-r:center+r] = 0.25

# noise
U += np.random.uniform(-0.01, 0.01, (size, size))
V += np.random.uniform(-0.01, 0.01, (size, size))

def laplacian_neumann(Z):
    """Compute the Laplacian of the array Z, assuming Neumann boundary conditions."""
    Z_new = np.copy(Z)
    
    Z_new[1:-1, 1:-1] = (
        Z[:-2, 1:-1] + Z[2:, 1:-1] +  # up down
        Z[1:-1, :-2] + Z[1:-1, 2:] -  # left right
        4 * Z[1:-1, 1:-1]
    ) / (dx ** 2)

    # Neumann boundary conditions: derivatives at the edges are null
    Z_new[0, :] = Z_new[1, :]
    Z_new[-1, :] = Z_new[-2, :]
    Z_new[:, 0] = Z_new[:, 1]
    Z_new[:, -1] = Z_new[:, -2]
    
    return Z_new

def laplacian_PBC(Z, delta=dx):
    '''Compute the Laplacian of the array Z.'''
    # here we use np.roll to implement the Periodic Boundary Conditions, PBC
    return (
        np.roll(Z, 1, axis=0) + np.roll(Z, -1, axis=0) +
        np.roll(Z, 1, axis=1) + np.roll(Z, -1, axis=1) - 4 * Z
    ) / (delta ** 2)

def update():
    global U, V
    
    # Laplacian
    Lu = laplacian_PBC(U)
    Lv = laplacian_PBC(V)
    
    # discretized PDE
    dUdt = D_u * Lu - U * V**2 + f * (1 - U)
    dVdt = D_v * Lv + U * V**2 - (f + k) * V
    
    U += dUdt * dt
    V += dVdt * dt

def plot_system():
    fig, ax = plt.subplots()
    im = ax.imshow(V, cmap='inferno', interpolation='bilinear')
    
    def animate(i):
        for _ in range(10):  # fast forward each 10 frames
            update()
        im.set_array(V)
        return [im]
    
    ani = animation.FuncAnimation(fig, animate, frames=200, interval=50)
    plt.colorbar(im)
    plt.show()

plot_system()
