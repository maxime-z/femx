import numpy as np
from femx.core.mesh import Mesh
from femx.core.fields import FieldSpec
from femx.core.dofs import DofMap
from femx.core.assembly import assemble_system
from femx.materials.linear_elastic import LinearElasticMaterial
from femx.formulations.elasticity import LinearElasticityFormulation
from femx.solvers.linear import solve_system

def create_cantilever_mesh(Lx: float, Ly: float, nx: int, ny: int) -> Mesh:
    """Create a regular Q1 grid for a cantilever beam."""
    x = np.linspace(0, Lx, nx + 1)
    y = np.linspace(0, Ly, ny + 1)
    
    # Grid coordinates
    X, Y = np.meshgrid(x, y)
    coords = np.vstack([X.ravel(), Y.ravel()]).T
    
    # Grid elements
    cells = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = (j + 1) * (nx + 1) + i + 1
            n3 = n2 - 1
            cells.append([n0, n1, n2, n3])
            
    cells = np.array(cells, dtype=int)
    
    # Left nodes: x = 0
    left_nodes = np.arange(ny + 1) * (nx + 1)
    # Right nodes: x = Lx
    right_nodes = np.arange(ny + 1) * (nx + 1) + nx
    
    boundaries = {
        "left": left_nodes,
        "right": right_nodes
    }
    
    return Mesh(coords=coords, cells=cells, boundaries=boundaries)

def main():
    print("--- Solving Cantilever Beam Elasticity on Q1 Mesh ---")
    
    # 1. Create a 4x1 element cantilever beam of size 4.0 x 1.0
    Lx, Ly = 4.0, 1.0
    nx, ny = 4, 1
    mesh = create_cantilever_mesh(Lx, Ly, nx, ny)
    print(f"Mesh generated: {mesh.n_nodes} nodes, {mesh.n_elements} elements")
    
    # 2. Setup field and DofMap (displacement 'u' has 2 components)
    fields = [FieldSpec(name="u", components=2, location="nodes", unknown=True)]
    dof_map = DofMap(fields=fields, geometry=mesh)
    
    # 3. Define material and formulation (rho=1.0, E=2.0e7, nu=0.3)
    material = LinearElasticMaterial(rho=1.0, E=2.0e7, nu=0.3)
    formulation = LinearElasticityFormulation(material=material)
    
    # 4. Assemble system
    K, M, f = assemble_system(dof_map, formulation, field_name="u")
    print(f"System assembled. stiffness shape: {K.shape}")
    
    # 5. Apply Dirichlet BCs: Fix left end (nodes on x=0) in both X and Y
    dirichlet_bcs = {}
    for node in mesh.boundaries["left"]:
        dof_x = dof_map.get_dof("u", node, 0)
        dof_y = dof_map.get_dof("u", node, 1)
        dirichlet_bcs[dof_x] = 0.0
        dirichlet_bcs[dof_y] = 0.0
        
    # 6. Apply tip load at the right end (nodes on x=Lx): Total force F_y = -1000.0
    # Distributed evenly among the nodes on the right boundary (2 nodes)
    F_y_node = -1000.0 / len(mesh.boundaries["right"])
    for node in mesh.boundaries["right"]:
        dof_y = dof_map.get_dof("u", node, 1)
        f[dof_y] += F_y_node
        
    # 7. Solve
    u_sol = solve_system(K, f, dirichlet_bcs)
    
    # Print results
    print("\nSolved Displacements at nodes:")
    print("Node ID | Coords (x, y) | Disp X   | Disp Y")
    print("-------------------------------------------------")
    for node in range(mesh.n_nodes):
        x, y = mesh.coords[node]
        idx_x = dof_map.get_dof("u", node, 0)
        idx_y = dof_map.get_dof("u", node, 1)
        ux = u_sol[idx_x]
        uy = u_sol[idx_y]
        print(f"{node:7d} | ({x:3.2f}, {y:3.2f})   | {ux:8.5f} | {uy:8.5f}")

    # 8. Visualization and VTK Export
    import os
    import matplotlib.pyplot as plt
    from femx.visualization.matplotlib_vis import (
        plot_boundary_conditions, plot_scalar_field_2d, plot_deformed_mesh
    )
    from femx.core.postprocessing import compute_element_stresses

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    neumann_dict = {}
    for node in mesh.boundaries["right"]:
        dof_y = dof_map.get_dof("u", node, 1)
        neumann_dict[dof_y] = F_y_node

    # 1. Problem Setup & Boundary Conditions Plot
    fig1, ax1 = plt.subplots(figsize=(8, 3))
    plot_boundary_conditions(mesh, dof_map, dirichlet_bcs=dirichlet_bcs, neumann_forces=neumann_dict, ax=ax1)
    ax1.set_title("Cantilever Beam Setup & Boundary Conditions")
    fig1.tight_layout()
    fig1.savefig(os.path.join(output_dir, "elasticity_q1_setup.png"), dpi=200)
    print("  Saved: examples/output/elasticity_q1_setup.png")

    N = mesh.n_nodes
    ux_vals = np.array([u_sol[dof_map.get_dof("u", i, 0)] for i in range(N)])
    uy_vals = np.array([u_sol[dof_map.get_dof("u", i, 1)] for i in range(N)])
    disp_norm = np.sqrt(ux_vals**2 + uy_vals**2)
    post_res = compute_element_stresses(mesh, dof_map, formulation, u_sol)

    # 2. All Field Solutions Grid Plot
    fig2, axes = plt.subplots(2, 2, figsize=(12, 6))
    plot_scalar_field_2d(mesh, ux_vals, title="Horizontal Displacement u_x", ax=axes[0, 0])
    plot_scalar_field_2d(mesh, uy_vals, title="Vertical Displacement u_y", ax=axes[0, 1])
    plot_scalar_field_2d(mesh, disp_norm, title="Displacement Magnitude ||u||", ax=axes[1, 0])
    plot_scalar_field_2d(mesh, post_res.nodal_von_mises, title="von Mises Stress sigma_vm", ax=axes[1, 1])
    fig2.suptitle("Cantilever Beam Field Solutions", fontsize=14, fontweight='bold')
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, "elasticity_q1_fields.png"), dpi=200)
    print("  Saved: examples/output/elasticity_q1_fields.png")

    # 3. Initial Frame vs Deformed Frame Plots
    fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(10, 6))
    plot_deformed_mesh(mesh, u_sol, dof_map, scale=100.0, field_values=disp_norm, field_title="Displacement Magnitude ||u||", ax=ax3a)
    plot_deformed_mesh(mesh, u_sol, dof_map, scale=100.0, field_values=post_res.nodal_von_mises, field_title="von Mises Stress", ax=ax3b)
    fig3.tight_layout()
    fig3.savefig(os.path.join(output_dir, "elasticity_q1_deformed.png"), dpi=200)
    print("  Saved: examples/output/elasticity_q1_deformed.png")

    plt.close('all')

    u_vector = u_sol.reshape((-1, 2))
    try:
        from femx.visualization.pyvista_vis import export_to_vtk
        export_to_vtk(mesh, os.path.join(output_dir, "elasticity_q1_solution.vtu"), u_vector, "Displacement")
    except Exception as e:
        print(f"Skipping VTK export: {e}")

if __name__ == "__main__":
    main()
