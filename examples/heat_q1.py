import numpy as np
import matplotlib.pyplot as plt
from femx.core.mesh import Mesh
from femx.core.fields import FieldSpec
from femx.core.dofs import DofMap
from femx.core.assembly import assemble_system
from femx.materials.linear_heat import LinearHeatMaterial
from femx.formulations.heat import HeatConductionFormulation
from femx.solvers.linear import solve_system

def create_q1_grid(Lx: float, Ly: float, nx: int, ny: int) -> Mesh:
    """Create a regular grid of bilinear Q1 elements."""
    x = np.linspace(0, Lx, nx + 1)
    y = np.linspace(0, Ly, ny + 1)
    
    # Grid coordinates
    X, Y = np.meshgrid(x, y)
    coords = np.vstack([X.ravel(), Y.ravel()]).T
    
    # Grid elements (cells)
    cells = []
    for j in range(ny):
        for i in range(nx):
            # Counter-clockwise nodes:
            # 3 (top-left)  2 (top-right)
            # 0 (bot-left)  1 (bot-right)
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = (j + 1) * (nx + 1) + i + 1
            n3 = n2 - 1
            cells.append([n0, n1, n2, n3])
            
    cells = np.array(cells, dtype=int)
    
    # Store boundaries as node indices
    left_nodes = np.arange(ny + 1) * (nx + 1)
    right_nodes = np.arange(ny + 1) * (nx + 1) + nx
    bottom_nodes = np.arange(nx + 1)
    top_nodes = np.arange(nx + 1) + ny * (nx + 1)
    
    boundaries = {
        "left": left_nodes,
        "right": right_nodes,
        "bottom": bottom_nodes,
        "top": top_nodes
    }
    
    return Mesh(coords=coords, cells=cells, boundaries=boundaries)

def main():
    print("--- Solving Heat Conduction on Q1 FEM Mesh ---")
    
    # 1. Create 4x4 Q1 element grid on [0, 1]x[0, 1]
    nx, ny = 4, 4
    mesh = create_q1_grid(1.0, 1.0, nx, ny)
    print(f"Mesh generated: {mesh.n_nodes} nodes, {mesh.n_elements} elements")
    
    # 2. Setup field and DofMap
    fields = [FieldSpec(name="T", components=1, location="nodes", unknown=True)]
    dof_map = DofMap(fields=fields, geometry=mesh)
    
    # 3. Define material and formulation (rho=50, C=1.0, K=10.0)
    material = LinearHeatMaterial(rho=50.0, C=1.0, K=10.0)
    formulation = HeatConductionFormulation(material=material)
    
    # 4. Assemble system
    K, M, f = assemble_system(dof_map, formulation, field_name="T", body_load=0.0)
    print(f"System assembled. stiffness shape: {K.shape}")
    
    # 5. Apply Dirichlet BCs (representing Java buildSimpleCavity)
    # Left boundary (x=0): T varies linearly from 10 to 20
    # Right boundary (x=1): T varies linearly from 20 to 10
    dirichlet_bcs = {}
    
    # Left boundary: coords[n, 1] is y coordinate
    for node in mesh.boundaries["left"]:
        y = mesh.coords[node, 1]
        T_val = 10.0 + 10.0 * y
        dof = dof_map.get_dof("T", node, 0)
        dirichlet_bcs[dof] = T_val
        
    # Right boundary: coords[n, 1] is y coordinate
    for node in mesh.boundaries["right"]:
        y = mesh.coords[node, 1]
        T_val = 20.0 - 10.0 * y
        dof = dof_map.get_dof("T", node, 0)
        dirichlet_bcs[dof] = T_val
        
    # 6. Solve
    T_sol = solve_system(K, f, dirichlet_bcs)
    
    # Print results
    print("\nSolved Temperature at nodes:")
    print("Node ID | Coords (x, y) | Temperature")
    print("-------------------------------------")
    for node in range(mesh.n_nodes):
        x, y = mesh.coords[node]
        print(f"{node:7d} | ({x:3.2f}, {y:3.2f})   | {T_sol[node]:8.4f}")

    import os
    # 7. Visualization and VTK Export
    from femx.visualization.matplotlib_vis import (
        plot_boundary_conditions, plot_scalar_field_2d, plot_mesh
    )

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Problem Setup & BCs Plot
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    plot_boundary_conditions(mesh, dof_map, dirichlet_bcs=dirichlet_bcs, ax=ax1)
    ax1.set_title("Heat Conduction Setup & Boundary Conditions")
    fig1.tight_layout()
    fig1.savefig(os.path.join(output_dir, "heat_q1_setup.png"), dpi=200)
    print("  Saved: examples/output/heat_q1_setup.png")

    # 2. Field Solution Contour Plot
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    plot_scalar_field_2d(mesh, T_sol, title="Temperature Field T (°C)", ax=ax2)
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, "heat_q1_fields.png"), dpi=200)
    print("  Saved: examples/output/heat_q1_fields.png")

    # 3. Initial Frame View Overlaid with Mesh & Solution
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    plot_mesh(mesh, show_nodes=True, ax=ax3)
    plot_scalar_field_2d(mesh, T_sol, title="Initial Frame Mesh & Temperature Field", ax=ax3)
    fig3.tight_layout()
    fig3.savefig(os.path.join(output_dir, "heat_q1_frame.png"), dpi=200)
    print("  Saved: examples/output/heat_q1_frame.png")

    plt.close('all')

    try:
        from femx.visualization.pyvista_vis import export_to_vtk
        export_to_vtk(mesh, os.path.join(output_dir, "heat_q1_solution.vtu"), T_sol, "Temperature")
    except Exception as e:
        print(f"Skipping VTK export: {e}")

if __name__ == "__main__":
    main()
