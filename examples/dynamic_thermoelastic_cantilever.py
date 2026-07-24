import os
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple

from femx.backends.numpy_backend import zeros, array
from femx.backends.numpy_backend import zeros, array
from femx.core.mesh import Mesh
from femx.core.fields import FieldSpec
from femx.core.dofs import DofMap
from femx.core.assembly import assemble_system
from femx.core.tensor_assembly import assemble_system_tensor
from femx.materials.thermoelastic import LinearThermoelasticMaterial
from femx.formulations.thermoelasticity import LinearThermoelasticityFormulation
from femx.solvers.transient import solve_transient_thermoelastic
from femx.visualization.matplotlib_vis import plot_mesh
from femx.visualization.animation import create_2d_field_animation

def generate_cantilever_mesh(L: float = 4.0, H: float = 1.0, nx: int = 40, ny: int = 10) -> Mesh:
    """Generate a structured Q1 Quad mesh for a 2D rectangular cantilever beam."""
    x = np.linspace(0.0, L, nx + 1)
    y = np.linspace(-H / 2.0, H / 2.0, ny + 1)
    X, Y = np.meshgrid(x, y)
    coords = np.vstack([X.ravel(), Y.ravel()]).T

    cells = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = (j + 1) * (nx + 1) + i + 1
            n3 = n2 - 1
            cells.append([n0, n1, n2, n3])
    cells = array(cells, dtype=int)

    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    boundaries = {
        "left": np.where(np.isclose(x_coords, 0.0))[0],
        "right": np.where(np.isclose(x_coords, L))[0],
        "bottom": np.where(np.isclose(y_coords, -H / 2.0))[0],
        "top": np.where(np.isclose(y_coords, H / 2.0))[0]
    }
    return Mesh(coords=coords, cells=cells, boundaries=boundaries)

def main():
    print("=================================================================")
    print("      Dynamic 2D Multi-Physics Thermoelasticity Showcase        ")
    print("=================================================================")

    # 1. Problem Setup & Material Definition
    L = 4.0   # Beam length (m)
    H = 1.0   # Beam height (m)
    nx, ny = 40, 10
    mesh = generate_cantilever_mesh(L=L, H=H, nx=nx, ny=ny)
    
    print(f"\n[Mesh Setup] Created Cantilever Beam ({L}m x {H}m):")
    print(f"             Elements: {mesh.n_elements} | Nodes: {mesh.n_nodes}")

    # Define Material (Aluminum-like properties)
    material = LinearThermoelasticMaterial(
        rho=2700,      # Density: 2700 kg/m^3
        E=70e9,        # Young's modulus: 70 GPa
        nu=0.33,       # Poisson's ratio: 0.33
        K_th=200,      # Thermal conductivity: 200 W/(m K)
        alpha=23e-6,   # Thermal expansion coeff: 23e-6 1/K
        C_cap=900,     # Specific heat capacity: 900 J/(kg K)
        T0=293.15      # Reference temperature: 293.15 K (20 C)
    )
    
    formulation = LinearThermoelasticityFormulation(material, mode="plane_stress")
    fields = [
        FieldSpec(name="u", components=2, location="nodes", unknown=True),
        FieldSpec(name="T", components=1, location="nodes", unknown=True)
    ]
    dof_map = DofMap(fields=fields, geometry=mesh)
    print(f"[DOF Map] Total Coupled System DOFs: {dof_map.n_dofs} (u: {len(dof_map.get_field_dofs('u'))}, T: {len(dof_map.get_field_dofs('T'))})")

    # Boundary Conditions (Clamped Left Edge u_x = u_y = 0; All boundaries Adiabatic for T)
    dirichlet_bcs = {}
    left_nodes = mesh.boundaries["left"]
    for node in left_nodes:
        ux_dof = dof_map.get_dof("u", node, 0)
        uy_dof = dof_map.get_dof("u", node, 1)
        dirichlet_bcs[ux_dof] = 0.0
        dirichlet_bcs[uy_dof] = 0.0

    # 2. Performance Benchmark: Traditional vs Tensor Assembly
    print("\n-----------------------------------------------------------------")
    print("      Benchmark: Traditional Loop vs Tensor Galerkin Assembly   ")
    print("-----------------------------------------------------------------")
    
    # Traditional Assembly
    t0 = time.perf_counter()
    K_trad, M_trad, f_trad = assemble_system(dof_map, formulation)
    t_trad = (time.perf_counter() - t0) * 1000.0
    
    # PyTorch Tensor Galerkin Assembly
    t0 = time.perf_counter()
    K_tens, M_tens, f_tens, _, _ = assemble_system_tensor(dof_map, formulation)
    t_tens = (time.perf_counter() - t0) * 1000.0
    
    speedup = t_trad / max(t_tens, 1e-6)
    
    print(f"  Traditional Loop Assembly Time : {t_trad:7.2f} ms")
    print(f"  Tensor Galerkin Engine Time   : {t_tens:7.2f} ms")
    print(f"  --> Speedup Factor             : {speedup:7.2f}x Faster!")
    print("-----------------------------------------------------------------")

    # Verify Assembly Accuracy
    diff_K = np.abs((K_trad - K_tens).data).max() if K_trad.nnz > 0 else 0.0
    diff_M = np.abs((M_trad - M_tens).data).max() if M_trad.nnz > 0 else 0.0
    print(f"[Verification] Max diff between Traditional & Tensor K: {diff_K:.2e}, M: {diff_M:.2e}")

    # 3. Dynamic Solicitation Setup (Periodic Shear Force at Right Tip)
    F0 = 5.0e5  # Amplitude of total shear force (500 kN)
    freq = 5.0  # Frequency (Hz)
    omega = 2.0 * np.pi * freq
    
    right_nodes = mesh.boundaries["right"]
    # Sort right boundary nodes by Y coordinate for trapezoidal force distribution
    right_nodes_sorted = right_nodes[np.argsort(mesh.coords[right_nodes, 1])]
    n_right = len(right_nodes_sorted)
    
    # 1D Trapezoidal integration weights along the right edge
    force_weights = np.ones(n_right)
    force_weights[0] = 0.5
    force_weights[-1] = 0.5
    force_weights /= np.sum(force_weights)  # Normalized sum = 1.0


    def applied_force_func(t: float) -> float:
        if t < 0.2:
            return t * F0 / 0.2
        else:
            return 0
    
    def periodic_force_func(t: float) -> Tuple[np.ndarray, np.ndarray]:
        f_u = zeros(len(dof_map.get_field_dofs("u")))
        f_T = zeros(len(dof_map.get_field_dofs("T")))
        
        # Periodic shear force on right boundary nodes (Y-direction)
        total_F_y = applied_force_func(t)
        for i, node in enumerate(right_nodes_sorted):
            f_u[2 * node+1] += total_F_y * force_weights[i]
            
        return f_u, f_T

    # 4. Probe Locations Definition
    probe_coords = {
        "Probe 1 (Root Top x=0.1L, y=+H/2)": [0.1 * L, H / 2],
        "Probe 2 (Mid Top  x=0.5L, y=+H/2)": [0.5 * L, H / 2],
        "Probe 3 (Tip Mid  x=1.0L, y=0.0)": [1.0 * L, 0.0],
        "Probe 4 (Neutral  x=0.5L, y=0.0)": [0.5 * L, 0.0],
    }
    
    probe_nodes = {}
    for name, p_xy in probe_coords.items():
        dists = np.linalg.norm(mesh.coords - np.array(p_xy), axis=1)
        node_idx = np.argmin(dists)
        probe_nodes[name] = node_idx

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    # --- Plot 1: Problem Setup Diagram ---
    fig, ax = plt.subplots(figsize=(10, 4), dpi=100)
    plot_mesh(mesh, show_nodes=False, ax=ax)
    
    # Mark Clamped BC on left
    left_coords = mesh.coords[mesh.boundaries["left"]]
    ax.scatter(left_coords[:, 0], left_coords[:, 1], color='blue', marker='s', s=40, zorder=5, label="Clamped u=0")
    
    # Mark Right Tip Force Arrows
    ax.quiver(L * np.ones(n_right), mesh.coords[right_nodes_sorted, 1], np.zeros(n_right), np.ones(n_right)*0.5,
               color='red', scale=1, zorder=6, label="Periodic Shear Fy(t)")
               
    # Mark Probes
    for name, node_idx in probe_nodes.items():
        px, py = mesh.coords[node_idx]
        ax.scatter(px, py, color='green', s=60, zorder=7)
        ax.text(px + 0.05, py + 0.05, name.split()[0] + " " + name.split()[1], color='darkgreen', fontweight='bold', fontsize=9, zorder=7)

    ax.set_title("Problem Setup: 2D Cantilever Beam & Thermoelastic Probes", fontsize=12, fontweight='bold')
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cantilever_setup.png"))
    plt.close()
    print("[Post-Processing] Saved Problem Setup plot to 'examples/output/cantilever_setup.png'")

    # 5. Dynamic Monolithic Time Integration
    t_end = 1.0    # Simulation time (s)
    dt = 0.002     # Time step (s) - 250 steps total (100 steps per cycle)
    print(f"\n[Dynamic Simulation] Running Monolithic Time Integration over t in [0, {t_end}s] (dt={dt}s)...")
    
    t0_sol = time.perf_counter()
    time_pts, u_hist, T_hist = solve_transient_thermoelastic(
        K=K_tens,
        M=M_tens,
        dof_map=dof_map,
        dirichlet_bcs=dirichlet_bcs,
        force_func=periodic_force_func,
        t_span=(0.0, t_end),
        dt=dt,
        T0=material.get_property("T0"),
        gamma=0.5,
        beta=0.25
    )
    t_sol = time.perf_counter() - t0_sol
    print(f"[Dynamic Simulation] Completed {len(time_pts)} time steps in {t_sol:.3f} s!")

    # Add initial reference temperature T0 to relative thermal perturbation
    T_hist_abs = T_hist + material.get_property("T0")

    # --- Plot 2: Solicitation Load vs Tip Displacement Response ---
    tip_mid_node = probe_nodes["Probe 3 (Tip Mid  x=1.0L, y=0.0)"]
    tip_uy = u_hist[:, 2 * tip_mid_node + 1]
    applied_force_y = np.array([applied_force_func(t) for t in time_pts])
    
    fig, (ax_f, ax_u) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, dpi=100)
    ax_f.plot(time_pts, applied_force_y / 1e3, 'r-', linewidth=2.0)
    ax_f.set_ylabel("Applied Load Fy (kN)", fontsize=11)
    ax_f.set_title("Time-Variant Periodic Shear Force & Dynamic Structural Response", fontsize=13, fontweight='bold')
    ax_f.grid(True, linestyle='--', alpha=0.6)
    
    ax_u.plot(time_pts, tip_uy * 1000.0, 'b-', linewidth=2.0)
    ax_u.set_xlabel("Time t (s)", fontsize=11)
    ax_u.set_ylabel("Tip Deflection uy (mm)", fontsize=11)
    ax_u.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cantilever_load_response.png"))
    plt.close()
    print("[Post-Processing] Saved Load vs Response plot to 'examples/output/cantilever_load_response.png'")

    # --- Plot 3: Temperature Probes Analysis ---
    plt.figure(figsize=(10, 5), dpi=100)
    for name, node_idx in probe_nodes.items():
        T_probe = T_hist_abs[:, node_idx]
        plt.plot(time_pts, T_probe, label=name, linewidth=2.0)
        
    plt.xlabel("Time t (s)", fontsize=12)
    plt.ylabel("Absolute Temperature T (K)", fontsize=12)
    plt.title("Dynamic Thermoelasticity: Temperature Probe Histories", fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cantilever_probes.png"))
    plt.close()
    print("[Post-Processing] Saved temperature probe histories plot to 'examples/output/cantilever_probes.png'")

    # --- Plot 4: Static Deformed Frame Snapshot Plot ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), dpi=100)
    
    # Initial Frame (t=0)
    plot_mesh(mesh, show_nodes=False, ax=ax1)
    ax1.set_title("Initial Cantilever Mesh (t = 0 s)")
    
    # Deformed Frame at Peak Solicitation
    peak_step = np.argmax(np.abs(tip_uy))
    u_peak = u_hist[peak_step]
    
    n_nodes = mesh.n_nodes
    u_dof_list = [(dof_map.get_dof("u", i, 0), dof_map.get_dof("u", i, 1)) for i in range(n_nodes)]
    u_x_peak = u_peak[[d[0] for d in u_dof_list]]
    u_y_peak = u_peak[[d[1] for d in u_dof_list]]
    
    deform_scale = 30.0  # Magnification scale for 0.18mm deflection
    deformed_coords = mesh.coords + deform_scale * np.column_stack([u_x_peak, u_y_peak])
    deformed_mesh = Mesh(coords=deformed_coords, cells=mesh.cells, boundaries=mesh.boundaries)
    
    plot_mesh(deformed_mesh, show_nodes=False, ax=ax2)
    ax2.set_title(f"Deformed Mesh at Peak Deflection t = {time_pts[peak_step]:.3f} s (Scale: {deform_scale}x)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cantilever_deformed_frames.png"))
    plt.close()
    print("[Post-Processing] Saved initial vs deformed frame snapshot to 'examples/output/cantilever_deformed_frames.png'")

    # 6. Generate 2D Dynamic Animation (.gif)
    generate_animation = False
    if generate_animation:
        print("\n[Animation] Rendering dynamic 2D field animation over time...")
        create_2d_field_animation(
            mesh=mesh,
            dof_map=dof_map,
            u_history=u_hist,
            T_history=T_hist_abs,
            time_pts=time_pts,
            deform_scale=deform_scale,
            output_path=os.path.join(output_dir, "cantilever_thermoelastic_animation.gif"),
            fps=20,
            title="Dynamic Thermoelastic Cantilever Solicitation"
        )
    print("\n=================================================================")
    print("      Showcase Completed Successfully! Output Artifacts Generated ")
    print("=================================================================")

if __name__ == "__main__":
    main()
