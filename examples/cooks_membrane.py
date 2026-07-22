import numpy as np
import matplotlib.pyplot as plt
from femx.backends.numpy_backend import array
from femx.core.mesh import Mesh
from femx.core.fields import FieldSpec
from femx.core.dofs import DofMap
from femx.core.assembly import assemble_system
from femx.materials.linear_elastic import LinearElasticMaterial
from femx.formulations.elasticity import LinearElasticityFormulation
from femx.solvers.linear import solve_system
from femx.core.boundary import integrate_neumann_traction
from femx.core.postprocessing import compute_element_stresses
from femx.visualization.pyvista_vis import export_to_vtk

def create_cooks_membrane_mesh(nx: int = 16, ny: int = 16) -> Mesh:
    """
    Generate Cook's Membrane quadrilateral mesh.
    Domain bounds:
        Left edge (x=0): y in [0, 44]
        Right edge (x=48): y in [44, 60]
    """
    # Create parametric grid [0, 1] x [0, 1]
    xi = np.linspace(0.0, 1.0, nx + 1)
    eta = np.linspace(0.0, 1.0, ny + 1)
    
    coords = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            s = xi[i]
            t = eta[j]
            # Mapping from unit square [0,1]^2 to Cook's membrane geometry:
            # x = 48 * s
            # y_bottom(s) = 44 * s
            # y_top(s) = 44 + 16 * s
            # y(s, t) = (1 - t) * y_bottom(s) + t * y_top(s)
            x = 48.0 * s
            y_bot = 44.0 * s
            y_top = 44.0 + 16.0 * s
            y = (1.0 - t) * y_bot + t * y_top
            coords.append([x, y])
            
    coords = array(coords)
    
    cells = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = (j + 1) * (nx + 1) + i + 1
            n3 = n2 - 1
            cells.append([n0, n1, n2, n3])
            
    cells = array(cells, dtype=int)
    
    # Boundary definitions
    left_nodes = np.arange(ny + 1) * (nx + 1)
    right_nodes = np.arange(ny + 1) * (nx + 1) + nx
    
    boundaries = {
        "left": left_nodes,
        "right": right_nodes
    }
    
    return Mesh(coords=coords, cells=cells, boundaries=boundaries)

def main():
    print("--- Solving Cook's Membrane Benchmark (Plane Strain Linear Elasticity) ---")
    
    # 1. Generate Cook's membrane mesh (16 x 16 elements)
    nx, ny = 16, 16
    mesh = create_cooks_membrane_mesh(nx, ny)
    print(f"Mesh generated: {mesh.n_nodes} nodes, {mesh.n_elements} elements")
    
    # 2. Setup field and DofMap
    fields = [FieldSpec(name="u", components=2, location="nodes", unknown=True)]
    dof_map = DofMap(fields=fields, geometry=mesh)
    
    # 3. Material properties: E = 1.0, nu = 0.3333333333333333, rho = 1.0
    material = LinearElasticMaterial(rho=1.0, E=1.0, nu=1.0 / 3.0)
    formulation = LinearElasticityFormulation(material=material, mode="plane_strain")
    
    # 4. Assemble stiffness matrix K and mass matrix M
    K, M, f_body = assemble_system(dof_map, formulation, field_name="u")
    
    # 5. Integrate Neumann shear traction on the right boundary (x = 48)
    # Total shear force F_y = 1.0 distributed along height L = 16 -> traction t_y = 1.0 / 16.0 = 0.0625
    t_y = 1.0 / 16.0
    f_neumann = integrate_neumann_traction(
        mesh=mesh,
        dof_map=dof_map,
        field_name="u",
        boundary_name="right",
        traction=array([0.0, t_y]),
        n_quad_pts=2
    )
    
    # Combined force vector
    f = f_body + f_neumann
    
    # 6. Apply Dirichlet BCs: Fix left end (x = 0) in both X and Y
    dirichlet_bcs = {}
    for node in mesh.boundaries["left"]:
        dof_x = dof_map.get_dof("u", node, 0)
        dof_y = dof_map.get_dof("u", node, 1)
        dirichlet_bcs[dof_x] = 0.0
        dirichlet_bcs[dof_y] = 0.0
        
    # 7. Solve linear system
    U = solve_system(K, f, dirichlet_bcs)
    
    # Top-right corner node index (last node on right boundary)
    top_right_node = mesh.boundaries["right"][-1]
    uy_top_right = U[dof_map.get_dof("u", top_right_node, 1)]
    print(f"\nTop-right corner vertical displacement u_y(48, 60): {uy_top_right:8.5f}")
    
    # 8. Postprocess stresses
    result = compute_element_stresses(mesh, dof_map, formulation, U)
    max_vm = np.max(result.nodal_von_mises)
    print(f"Maximum von Mises stress: {max_vm:8.5f}")
    
    # 9. Plot and export results
    from femx.visualization.matplotlib_vis import (
        plot_boundary_conditions, plot_scalar_field_2d, plot_deformed_mesh
    )

    # Neumann point force dictionary for visualization
    neumann_dict = {}
    for node in mesh.boundaries["right"]:
        dof_y = dof_map.get_dof("u", node, 1)
        neumann_dict[dof_y] = t_y

    # 1. Problem Setup & Boundary Conditions Plot
    fig1, ax1 = plt.subplots(figsize=(7, 6))
    plot_boundary_conditions(mesh, dof_map, dirichlet_bcs=dirichlet_bcs, neumann_forces=neumann_dict, ax=ax1)
    ax1.set_title("Cook's Membrane Setup & Boundary Conditions")
    fig1.tight_layout()
    fig1.savefig("examples/cooks_membrane_setup.png", dpi=200)
    print("Saved: examples/cooks_membrane_setup.png")

    N = mesh.n_nodes
    ux_vals = np.array([U[dof_map.get_dof("u", i, 0)] for i in range(N)])
    uy_vals = np.array([U[dof_map.get_dof("u", i, 1)] for i in range(N)])
    disp_norm = np.sqrt(ux_vals**2 + uy_vals**2)

    # 2. All Field Solutions Grid Plot
    fig2, axes = plt.subplots(2, 2, figsize=(12, 10))
    plot_scalar_field_2d(mesh, ux_vals, title="Horizontal Displacement u_x", ax=axes[0, 0])
    plot_scalar_field_2d(mesh, uy_vals, title="Vertical Displacement u_y", ax=axes[0, 1])
    plot_scalar_field_2d(mesh, disp_norm, title="Displacement Magnitude ||u||", ax=axes[1, 0])
    plot_scalar_field_2d(mesh, result.nodal_von_mises, title="von Mises Stress sigma_vm", ax=axes[1, 1])
    fig2.suptitle("Cook's Membrane Field Solutions (u, sigma)", fontsize=14, fontweight='bold')
    fig2.tight_layout()
    fig2.savefig("examples/cooks_membrane_fields.png", dpi=200)
    print("Saved: examples/cooks_membrane_fields.png")

    # 3. Initial Frame vs Deformed Frame Plots
    fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(8, 12))
    plot_deformed_mesh(mesh, U, dof_map, scale=0.5, field_values=disp_norm, field_title="Displacement Magnitude ||u||", ax=ax3a)
    plot_deformed_mesh(mesh, U, dof_map, scale=0.5, field_values=result.nodal_von_mises, field_title="von Mises Stress", ax=ax3b)
    fig3.tight_layout()
    fig3.savefig("examples/cooks_membrane_deformed.png", dpi=200)
    print("Saved: examples/cooks_membrane_deformed.png")

    plt.close('all')

    u_vector = U.reshape((-1, 2))
    try:
        export_to_vtk(mesh, "examples/cooks_membrane_solution.vtu", u_vector, "Displacement")
        print("Exported VTK file to examples/cooks_membrane_solution.vtu")
    except Exception as e:
        print(f"Skipping VTK export: {e}")

if __name__ == "__main__":
    main()
