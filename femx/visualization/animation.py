import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import PolyCollection
from typing import Dict, Optional, Tuple
from femx.backends.numpy_backend import ndarray
from femx.core.mesh import Mesh
from femx.core.dofs import DofMap

def create_2d_field_animation(
    mesh: Mesh,
    dof_map: DofMap,
    u_history: ndarray,
    T_history: ndarray,
    time_pts: ndarray,
    deform_scale: float = 1.0,
    output_path: str = "field_animation.gif",
    fps: int = 15,
    title: str = "2D Dynamic Thermoelasticity Animation",
    dpi: int = 100
):
    """
    Create and save a 2D animation showing field solutions (Temperature & Displacement Magnitude)
    over time on both initial and deformed frames.

    Args:
        mesh: Mesh object
        dof_map: DofMap object for field lookups
        u_history: ndarray of shape (num_steps, n_dofs_u) containing displacement history
        T_history: ndarray of shape (num_steps, n_dofs_T) containing temperature history
        time_pts: ndarray of shape (num_steps,) containing time points
        deform_scale: Displacement magnification scale for deformed frame
        output_path: Output file path (.gif or .mp4)
        fps: Frames per second for animation
        title: Main title of the plot
        dpi: Resolution of animation frames
    """
    num_steps = len(time_pts)
    coords = mesh.coords
    cells = mesh.cells
    n_nodes = mesh.n_nodes
    
    # Extract nodal values per time step
    if u_history.shape[1] == 2 * n_nodes:
        u_x_hist = u_history[:, 0::2]
        u_y_hist = u_history[:, 1::2]
    else:
        u_dofs = [(dof_map.get_dof("u", i, 0), dof_map.get_dof("u", i, 1)) for i in range(n_nodes)]
        u_x_hist = u_history[:, [d[0] for d in u_dofs]]
        u_y_hist = u_history[:, [d[1] for d in u_dofs]]

    u_mag_hist = np.sqrt(u_x_hist**2 + u_y_hist**2)
    
    if T_history.shape[1] == n_nodes:
        T_val_hist = T_history
    else:
        T_dofs = [dof_map.get_dof("T", i, 0) for i in range(n_nodes)]
        T_val_hist = T_history[:, T_dofs]
    
    # Min/Max ranges for consistent colormaps with padding
    T_min, T_max = np.min(T_val_hist), np.max(T_val_hist)
    T_pad = max((T_max - T_min) * 0.05, 0.01)
    T_clim = (T_min - T_pad, T_max + T_pad)
        
    u_min, u_max = np.min(u_mag_hist), np.max(u_mag_hist)
    u_pad = max((u_max - u_min) * 0.05, 1e-6)
    u_clim = (u_min - u_pad, u_max + u_pad)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=dpi)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Initial polygons & face values for Frame 0
    deformed_coords_0 = coords + deform_scale * np.column_stack([u_x_hist[0], u_y_hist[0]])
    polys_0 = [deformed_coords_0[cell] for cell in cells]
    elem_T0 = np.mean(T_val_hist[0][cells], axis=1)
    elem_u0 = np.mean(u_mag_hist[0][cells], axis=1)
    
    # Plot on ax1 (Temperature on Deformed Frame)
    pc1 = PolyCollection(polys_0, array=elem_T0, cmap='plasma', edgecolors='black', linewidths=0.5)
    pc1.set_clim(T_clim)
    ax1.add_collection(pc1)
    cbar1 = fig.colorbar(pc1, ax=ax1)
    cbar1.set_label("Temperature T (K)")
    ax1.set_title("Temperature on Deformed Frame")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_aspect('equal')

    # Plot on ax2 (Displacement Magnitude on Deformed Frame)
    pc2 = PolyCollection(polys_0, array=elem_u0, cmap='viridis', edgecolors='black', linewidths=0.5)
    pc2.set_clim(u_clim)
    ax2.add_collection(pc2)
    cbar2 = fig.colorbar(pc2, ax=ax2)
    cbar2.set_label("Displacement ||u|| (m)")
    ax2.set_title(f"Displacement Magnitude (Scale: {deform_scale:.1f}x)")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_aspect('equal')

    # Adjust axis limits with margin based on max deformation
    x_min, y_min = np.min(coords, axis=0)
    x_max, y_max = np.max(coords, axis=0)
    max_u = np.max(u_mag_hist) * deform_scale
    margin_x = (x_max - x_min) * 0.05 + max_u
    margin_y = (y_max - y_min) * 0.15 + max_u
    
    ax1.set_xlim(x_min - margin_x * 0.1, x_max + margin_x)
    ax1.set_ylim(y_min - margin_y, y_max + margin_y)
    ax2.set_xlim(x_min - margin_x * 0.1, x_max + margin_x)
    ax2.set_ylim(y_min - margin_y, y_max + margin_y)

    time_text = fig.text(0.5, 0.02, '', ha='center', fontsize=12, fontweight='bold')

    def update(frame):
        t = time_pts[frame]
        u_x = u_x_hist[frame]
        u_y = u_y_hist[frame]
        T_val = T_val_hist[frame]
        u_mag = u_mag_hist[frame]
        
        # Deformed coordinates for this frame
        deformed_coords = coords + deform_scale * np.column_stack([u_x, u_y])
        polys_frame = [deformed_coords[cell] for cell in cells]
        
        # Element face colors (average nodal value per element)
        elem_T = np.mean(T_val[cells], axis=1)
        elem_u = np.mean(u_mag[cells], axis=1)
        
        pc1.set_verts(polys_frame)
        pc1.set_array(elem_T)
        
        pc2.set_verts(polys_frame)
        pc2.set_array(elem_u)
        
        time_text.set_text(f"Time: {t:.4f} s (Step {frame+1}/{num_steps})")
        return pc1, pc2, time_text

    anim = animation.FuncAnimation(fig, update, frames=num_steps, interval=1000/fps, blit=False)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    if output_path.endswith(".gif"):
        anim.save(output_path, writer='pillow', fps=fps)
    elif output_path.endswith(".mp4"):
        anim.save(output_path, writer='ffmpeg', fps=fps)
    else:
        anim.save(output_path, writer='pillow', fps=fps)

    plt.close(fig)
    print(f"[Animation] Successfully saved dynamic 2D field animation to {output_path}")
