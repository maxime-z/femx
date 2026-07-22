import time
import numpy as np
from femx.backends.numpy_backend import array
from femx.core.mesh import Mesh
from femx.core.fields import FieldSpec
from femx.core.dofs import DofMap
from femx.core.assembly import assemble_system as assemble_system_traditional
from femx.core.tensor_assembly import assemble_system_tensor
from femx.core.routing import build_routing_matrices
from femx.materials.linear_elastic import LinearElasticMaterial
from femx.formulations.elasticity import LinearElasticityFormulation
from femx.solvers.linear import solve_system

def create_grid_mesh(Lx: float, Ly: float, nx: int, ny: int) -> Mesh:
    """Create a regular nx x ny grid of Q1 quadrilateral elements."""
    x = np.linspace(0, Lx, nx + 1)
    y = np.linspace(0, Ly, ny + 1)
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
    left_nodes = np.arange(ny + 1) * (nx + 1)
    right_nodes = np.arange(ny + 1) * (nx + 1) + nx
    
    boundaries = {"left": left_nodes, "right": right_nodes}
    return Mesh(coords=coords, cells=cells, boundaries=boundaries)

def main():
    print("=== Assembly Engines Benchmark: Traditional COO vs TensorGalerkin Map-Reduce ===")
    
    # 1. Mesh setup: 50 x 50 Q1 element grid (2,500 elements, 2,601 nodes, 5,202 DoFs)
    nx, ny = 50, 50
    mesh = create_grid_mesh(10.0, 10.0, nx, ny)
    print(f"Mesh: {mesh.n_elements} elements, {mesh.n_nodes} nodes, {2 * mesh.n_nodes} DoFs")
    
    fields = [FieldSpec(name="u", components=2, location="nodes", unknown=True)]
    dof_map = DofMap(fields=fields, geometry=mesh)
    
    material = LinearElasticMaterial(rho=7800.0, E=2.1e11, nu=0.3)
    formulation = LinearElasticityFormulation(material=material, mode="plane_strain")
    
    # 2. Benchmark Traditional COO Assembly Engine
    t0 = time.perf_counter()
    K_trad, M_trad, f_trad = assemble_system_traditional(dof_map, formulation, field_name="u")
    t1 = time.perf_counter()
    time_trad = t1 - t0
    print(f"\n[1] Traditional COO Assembly Engine Time: {time_trad * 1000.0:8.2f} ms")
    
    # 3. Precompute Routing Matrices for Stage II
    t0_route = time.perf_counter()
    routing = build_routing_matrices(mesh, dof_map, field_name="u")
    t1_route = time.perf_counter()
    time_route = t1_route - t0_route
    print(f"    (Precomputing Topology Routing Matrices: {time_route * 1000.0:8.2f} ms)")
    
    # 4. Benchmark TensorGalerkin Engine (Batch-Map + SpMM)
    t0_tens = time.perf_counter()
    K_tens, M_tens, f_tens, K_local, F_local = assemble_system_tensor(
        dof_map=dof_map,
        formulation=formulation,
        field_name="u",
        routing=routing
    )
    t1_tens = time.perf_counter()
    time_tens = t1_tens - t0_tens
    print(f"[2] TensorGalerkin Map-Reduce Assembly Time: {time_tens * 1000.0:8.2f} ms")
    
    speedup = time_trad / time_tens if time_tens > 0 else 0
    print(f"\n==> TensorGalerkin Assembly Speedup: {speedup:5.2f}x faster!")
    
    # 5. Numerical Accuracy Check
    max_matrix_diff = np.max(np.abs(K_tens.toarray() - K_trad.toarray()))
    rel_matrix_diff = max_matrix_diff / np.max(np.abs(K_trad.toarray()))
    print(f"Max absolute matrix error |K_tensor - K_traditional|: {max_matrix_diff:.2e}")
    print(f"Relative matrix error: {rel_matrix_diff:.2e}")
    assert rel_matrix_diff < 1e-10
    print("Numerical validation successful! Both assembly engines produce identical global matrices.")

    # 6. Solve & Generate Visualizations
    dirichlet_bcs = {}
    for node in mesh.boundaries["left"]:
        dirichlet_bcs[dof_map.get_dof("u", node, 0)] = 0.0
        dirichlet_bcs[dof_map.get_dof("u", node, 1)] = 0.0

    F_y_node = -1e6 / len(mesh.boundaries["right"])
    for node in mesh.boundaries["right"]:
        f_tens[dof_map.get_dof("u", node, 1)] += F_y_node

    u_sol = solve_system(K_tens, f_tens, dirichlet_bcs)

    import matplotlib.pyplot as plt
    from femx.visualization.matplotlib_vis import (
        plot_boundary_conditions, plot_scalar_field_2d, plot_deformed_mesh
    )
    from femx.core.postprocessing import compute_element_stresses

    neumann_dict = {dof_map.get_dof("u", n, 1): F_y_node for n in mesh.boundaries["right"]}

    # 1. Setup Plot
    fig1, ax1 = plt.subplots(figsize=(7, 6))
    plot_boundary_conditions(mesh, dof_map, dirichlet_bcs=dirichlet_bcs, neumann_forces=neumann_dict, ax=ax1)
    ax1.set_title("Benchmark 50x50 Mesh & Boundary Conditions")
    fig1.tight_layout()
    fig1.savefig("examples/compare_engines_setup.png", dpi=200)
    print("  Saved: examples/compare_engines_setup.png")

    N = mesh.n_nodes
    ux_vals = np.array([u_sol[dof_map.get_dof("u", i, 0)] for i in range(N)])
    uy_vals = np.array([u_sol[dof_map.get_dof("u", i, 1)] for i in range(N)])
    disp_norm = np.sqrt(ux_vals**2 + uy_vals**2)
    post_res = compute_element_stresses(mesh, dof_map, formulation, u_sol)

    # 2. Field Solutions Plot
    fig2, axes = plt.subplots(2, 2, figsize=(12, 10))
    plot_scalar_field_2d(mesh, ux_vals, title="Horizontal Displacement u_x", ax=axes[0, 0])
    plot_scalar_field_2d(mesh, uy_vals, title="Vertical Displacement u_y", ax=axes[0, 1])
    plot_scalar_field_2d(mesh, disp_norm, title="Displacement Magnitude ||u||", ax=axes[1, 0])
    plot_scalar_field_2d(mesh, post_res.nodal_von_mises / 1e6, title="von Mises Stress sigma_vm (MPa)", ax=axes[1, 1])
    fig2.suptitle("Benchmark 50x50 Mesh Solution Fields", fontsize=14, fontweight='bold')
    fig2.tight_layout()
    fig2.savefig("examples/compare_engines_fields.png", dpi=200)
    print("  Saved: examples/compare_engines_fields.png")

    # 3. Deformed Mesh Plot
    fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(8, 12))
    plot_deformed_mesh(mesh, u_sol, dof_map, scale=50.0, field_values=disp_norm, field_title="Displacement Magnitude ||u||", ax=ax3a)
    plot_deformed_mesh(mesh, u_sol, dof_map, scale=50.0, field_values=post_res.nodal_von_mises / 1e6, field_title="von Mises Stress (MPa)", ax=ax3b)
    fig3.tight_layout()
    fig3.savefig("examples/compare_engines_deformed.png", dpi=200)
    print("  Saved: examples/compare_engines_deformed.png")

    plt.close('all')

if __name__ == "__main__":
    main()
