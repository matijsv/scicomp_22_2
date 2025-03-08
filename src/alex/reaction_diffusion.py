import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
from IPython.display import HTML
import matplotlib as mpl
import imageio_ffmpeg
import os


print(imageio_ffmpeg.get_ffmpeg_exe())

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

def reset_global_UV():
    global U, V
    U = np.ones((size, size)) * 0.5
    V = np.zeros((size, size))

    V[center-r:center+r, center-r:center+r] = 0.25

    # noise
    U += np.random.uniform(-0.01, 0.01, (size, size))
    V += np.random.uniform(-0.01, 0.01, (size, size))

def laplacian_dirichlet_strong(Z, U_BC):
    """Compute the Laplacian of the array Z, assuming Dirichlet boundary conditions (Strong Imposition)."""
    Z_new = np.copy(Z)
    
    Z_new[1:-1, 1:-1] = (
        Z[:-2, 1:-1] + Z[2:, 1:-1] +  # up down
        Z[1:-1, :-2] + Z[1:-1, 2:] -  # left right
        4 * Z[1:-1, 1:-1]
    ) / (dx ** 2)

    # Dirichlet boundary conditions: boundary values are set to U_BC
    Z_new[0, :] = U_BC  # Top boundary
    Z_new[-1, :] = U_BC  # Bottom boundary
    Z_new[:, 0] = U_BC  # Left boundary
    Z_new[:, -1] = U_BC  # Right boundary
    
    return Z_new

def laplacian_dirichlet_ghost(Z, U_BC, alpha, h, dt):
    """Compute the Laplacian of the array Z using Dirichlet boundary conditions with Ghost Node Elimination."""
    Z_new = np.copy(Z)
    
    # Standard Laplacian calculation for inner points
    Z_new[1:-1, 1:-1] = (
        Z[:-2, 1:-1] + Z[2:, 1:-1] +  # up down
        Z[1:-1, :-2] + Z[1:-1, 2:] -  # left right
        4 * Z[1:-1, 1:-1]
    ) / (h ** 2)

    # Move to update function!!!
    # Dirichlet boundary conditions: boundary values are set to U_BC
    # # Ghost node elimination: update boundary using U_0 equation
    # Z_new[0, :] += dt * (2 * alpha / h**2) * (U_BC - Z[0, :])  # Top boundary
    # Z_new[-1, :] += dt * (2 * alpha / h**2) * (U_BC - Z[-1, :])  # Bottom boundary
    # Z_new[:, 0] += dt * (2 * alpha / h**2) * (U_BC - Z[:, 0])  # Left boundary
    # Z_new[:, -1] += dt * (2 * alpha / h**2) * (U_BC - Z[:, -1])  # Right boundary
    
    return Z_new

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

def update(boundary_condition='PBC', U_BC_update=0.5, alpha_update=D_v, h_update=dx, dt_update=dt, f_value = f, k_value = k):
    '''Update the state of the system.'''
    global U, V
    
    if boundary_condition == 'PBC':
        laplacian = laplacian_PBC
    elif boundary_condition == 'Dirichlet Strong':
        laplacian = lambda Z: laplacian_dirichlet_strong(Z, U_BC=U_BC_update)
    elif boundary_condition == 'Dirichlet Ghost':
        laplacian = lambda Z: laplacian_dirichlet_ghost(Z, U_BC=U_BC_update, alpha=alpha_update, h=h_update, dt=dt_update)
    elif boundary_condition == 'Neumann':
        laplacian = laplacian_neumann
    else:
        raise ValueError('Invalid boundary condition')

    # Laplacian
    Lu = laplacian(U)
    Lv = laplacian(V)
    
    # discretized PDE
    dUdt = D_u * Lu - U * V**2 + f_value * (1 - U)
    dVdt = D_v * Lv + U * V**2 - (f_value + k_value) * V
    
    U += dUdt * dt
    V += dVdt * dt

    # Only for dirichlet ghost!!!
    if boundary_condition == 'Dirichlet Ghost':
        U[0, :] += dt * (2 * alpha_update / h_update**2) * (U_BC_update - U[0, :])  # Top boundary
        U[-1, :] += dt * (2 * alpha_update / h_update**2) * (U_BC_update - U[-1, :])  # Bottom boundary
        U[:, 0] += dt * (2 * alpha_update / h_update**2) * (U_BC_update - U[:, 0])  # Left boundary
        U[:, -1] += dt * (2 * alpha_update / h_update**2) * (U_BC_update - U[:, -1])  # Right boundary

        V[0, :] += dt * (2 * alpha_update / h_update**2) * (U_BC_update - V[0, :])  # Top boundary
        V[-1, :] += dt * (2 * alpha_update / h_update**2) * (U_BC_update - V[-1, :])  # Bottom boundary
        V[:, 0] += dt * (2 * alpha_update / h_update**2) * (U_BC_update - V[:, 0])  # Left boundary
        V[:, -1] += dt * (2 * alpha_update / h_update**2) * (U_BC_update - V[:, -1])  # Right boundary

def plot_system(boundary_condition="PBC"):
    fig, ax = plt.subplots()
    im = ax.imshow(V, cmap='inferno', interpolation='bilinear')
    ax.set_title(f"Gray-Scott model Simulation\nBoundary Condition: {boundary_condition}. Initial Condition See Set_2.")

    def animate(i):
        for _ in range(10):  # fast forward each 10 frames
            update(boundary_condition = boundary_condition)
        im.set_array(V)
        return [im]
    
    ani = animation.FuncAnimation(fig, animate, frames=200, interval=50)
    
    plt.colorbar(im)
    plt.show()

# plot the animation of Gray-Scott model simulation
def plot_animation(boundary_condition="PBC", observed_variable = "U", f_value = f, k_value = k):
    reset_global_UV()
    
    # set the path to ffmpeg.exe
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    if not ffmpeg_path:
        ffmpeg_path = "C:/ffmpeg/bin/ffmpeg.exe"  # manually set the path to ffmpeg.exe, please check the path on your computer!!!
    mpl.rcParams['animation.ffmpeg_path'] = ffmpeg_path
    # mpl.rcParams['animation.embed_limit'] = 200
    print(f"Using FFmpeg at: {ffmpeg_path}")

    # pick up the observed variable
    global U, V
    observed_range = {"U": U, "V": V}
    if observed_variable not in observed_range:
        raise ValueError("Invalid observed variable")
    observed_value = observed_range[observed_variable]

    ###### MP4 version animation ######
    fig_mp4, ax_mp4 = plt.subplots()
    im_mp4 = ax_mp4.imshow(V, cmap='inferno', interpolation='bilinear')
    if boundary_condition == "PBC":
        ax_mp4.set_title(f"Gray-Scott Model Simulation\nBoundary: Periodic Boundary Condition")
    else :
        ax_mp4.set_title(f"Gray-Scott Model Simulation\nBoundary: {boundary_condition}")

    def animate_mp4(i):
        for _ in range(10):
            # core update function
            # remember to set the boundary condition, f, and k values
            update(boundary_condition=boundary_condition, f_value=f_value, k_value=k_value)
        im_mp4.set_array(observed_value)
        return [im_mp4]

    ani_mp4 = animation.FuncAnimation(fig_mp4, animate_mp4, frames=500, interval=50)

    # save the animation as an MP4 file
    writer = FFMpegWriter(fps=30, metadata=dict(artist='Your Name'), bitrate=1800)
    os.makedirs("./fig", exist_ok=True)
    ani_mp4.save(f"./fig/Gray-Scott-{boundary_condition}-observed_{observed_variable}-f_{f_value}-k_{k_value}.mp4", writer=writer)
    print("MP4 saved, Gray-Scott-model.mp4")

    plt.close(fig_mp4)  # close the figure to avoid showing it in the notebook

    ###### HTML version animation ######
    reset_global_UV()
    fig_html, ax_html = plt.subplots()
    im_html = ax_html.imshow(V, cmap='inferno', interpolation='bilinear')
    if boundary_condition == "PBC":
        ax_html.set_title(f"Gray-Scott Model Simulation\nBoundary: Periodic Boundary Condition")
    else :
        ax_html.set_title(f"Gray-Scott Model Simulation\nBoundary: {boundary_condition}")

    def animate_html(i):
        for _ in range(50):
            # core update function
            # remember to set the boundary condition, f, and k values
            update(boundary_condition=boundary_condition, f_value = f, k_value = k)
        im_html.set_array(observed_value)
        return [im_html]

    ani_html = animation.FuncAnimation(fig_html, animate_html, frames=500, interval=50)

    plt.colorbar(im_html)
    
    return HTML(ani_html.to_jshtml())  # display the animation in the notebook



if __name__ == '__main__':
    plot_system(boundary_condition="PBC")
