import time
import numpy as np
import scipy.sparse as sp
from femx.backends.numpy_backend import array
from femx.core.mesh import Mesh
from femx.core.fields import FieldSpec
from femx.core.dofs import DofMap
from femx.core.assembly import assemble_system as assemble_system_traditional
from femx.core.tensor_assembly import assemble_system_tensor
from femx.core.routing import build_routing_matrices
from femx.materials.thermoelastic import LinearThermoelasticMaterial
from femx.formulations.thermoelasticity import LinearThermoelasticityFormulation
from femx.solvers.linear import solve_system

def create_beam_mesh(Lx: float, Ly: float, nx: int, ny: int) -> Mesh:
    """Create a regular nx x ny grid of Q1 quadrilateral elements for a 2D beam."""
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
    print("=== Phase 3 Benchmark: Coupled Multi-Field Linear Thermoelasticity (u + T) ===")
    
    # 1. Setup beam mesh: 40 x 10 Q1 quads (400 elements, 451 nodes, 1,353 coupled DoFs)
    Lx, Ly = 2.0, 0.5
    nx, ny = 40, 10
    mesh = create_beam_mesh(Lx, Ly, nx, ny)
    print(f"Mesh: {mesh.n_elements} elements, {mesh.n_nodes} nodes")
    
    fields = [
        FieldSpec(name="u", components=2, location="nodes", unknown=True),
        FieldSpec(name="T", components=1, location="nodes", unknown=True)
    ]
    dof_map = DofMap(fields=fields, geometry=mesh)
    print(f"Total Coupled Degrees of Freedom: {dof_map.n_dofs} DoFs (u_x, u_y, T)")
    
    material = LinearThermoelasticMaterial(
        rho=7800.0,
        E=2.0e11,
        nu=0.3,
        K_th=50.0,
        alpha=1.2e-5,
        C_cap=460.0,
        T0=0.0
    )
    formulation = LinearThermoelasticityFormulation(material=material, mode="plane_stress")
    
    # 2. Benchmark Traditional COO Coupled Assembly
    t0 = time.perf_counter()
    K_trad, M_trad, f_trad = assemble_system_traditional(dof_map, formulation)
    t1 = time.perf_counter()
    time_trad = t1 - t0
    print(f"\n[1] Traditional COO Assembly Time:     {time_trad * 1000.0:8.2f} ms")
    
    # 3. Benchmark TensorGalerkin Map-Reduce Coupled Assembly
    t0_route = time.perf_counter()
    routing = build_routing_matrices(mesh, dof_map, field_name=["u", "T"])
    t1_route = time.perf_counter()
    time_route = t1_route - t0_route
    print(f"    (Precomputing Topology Routing:     {time_route * 1000.0:8.2f} ms)")
    
    t0_tens = time.perf_counter()
    K_tens, M_tens, f_tens, K_local, F_local = assemble_system_tensor(dof_map, formulation, routing=routing)
    t1_tens = time.perf_counter()
    time_tens = t1_tens - t0_tens
    print(f"[2] TensorGalerkin Map-Reduce Assembly: {time_tens * 1000.0:8.2f} ms")
    
    speedup = time_trad / time_tens if time_tens > 0 else 0.0
    print(f"==> TensorGalerkin Speedup: {speedup:5.2f}x faster!")
    
    max_diff = np.max(np.abs(K_tens.toarray() - K_trad.toarray()))
    rel_diff = max_diff / np.max(np.abs(K_trad.toarray()))
    print(f"Max matrix difference |K_tensor - K_traditional|: {max_diff:.2e} (relative {rel_diff:.2e})")
    assert rel_diff < 1e-12
    
    # 4. Solve Coupled Problem: Heated beam fixed at left edge, non-uniform temperature profile
    dirichlet_bcs = {}
    
    # Thermal BCs: T = 100°C on left edge (x=0), T = 0°C on right edge (x=L)
    for node in mesh.boundaries["left"]:
        dirichlet_bcs[dof_map.get_dof("T", node, 0)] = 100.0
    for node in mesh.boundaries["right"]:
        dirichlet_bcs[dof_map.get_dof("T", node, 0)] = 0.0
        
    # Mechanical BCs: Fixed support on left edge (u_x = 0, u_y = 0)
    for node in mesh.boundaries["left"]:
        dirichlet_bcs[dof_map.get_dof("u", node, 0)] = 0.0
        dirichlet_bcs[dof_map.get_dof("u", node, 1)] = 0.0
        
    U_sol = solve_system(K_tens, f_tens, dirichlet_bcs)
    
    # Extract temperature at center node
    center_node_idx = (ny // 2) * (nx + 1) + (nx // 2)
    T_center = U_sol[dof_map.get_dof("T", center_node_idx, 0)]
    u_x_right_center = U_sol[dof_map.get_dof("u", mesh.boundaries["right"][ny // 2], 0)]
    
    print(f"\nSolution Results:")
    print(f"  Temperature at beam center (x={Lx/2}): {T_center:.2f} °C (Linear thermal gradient = 50.00 °C)")
    print(f"  Thermal expansion u_x at right tip:  {u_x_right_center * 1000.0:.4f} mm")

    # 5. Visualizations
    import matplotlib.pyplot as plt
    from femx.visualization.matplotlib_vis import (
        plot_mesh, plot_boundary_conditions, plot_scalar_field_2d, plot_deformed_mesh
    )
    from femx.core.postprocessing import compute_element_stresses
    from femx.formulations.elasticity import LinearElasticityFormulation
    from femx.materials.linear_elastic import LinearElasticMaterial

    N = mesh.n_nodes
    T_vals = np.array([U_sol[dof_map.get_dof("T", i, 0)] for i in range(N)])
    ux_vals = np.array([U_sol[dof_map.get_dof("u", i, 0)] for i in range(N)])
    uy_vals = np.array([U_sol[dof_map.get_dof("u", i, 1)] for i in range(N)])
    disp_norm = np.sqrt(ux_vals**2 + uy_vals**2)

    # Compute von Mises stresses
    elastic_mat = LinearElasticMaterial(rho=7800.0, E=2.0e11, nu=0.3)
    elastic_form = LinearElasticityFormulation(material=elastic_mat, mode="plane_stress")
    u_dof_map = DofMap(fields=[FieldSpec(name="u", components=2, location="nodes", unknown=True)], geometry=mesh)
    u_dof_indices = [dof_map.get_dof("u", i, comp) for i in range(N) for comp in (0, 1)]
    u_sol_only = U_sol[u_dof_indices]
    
    post_res = compute_element_stresses(mesh, u_dof_map, elastic_form, u_sol_only)
    sigma_vm_nodal = post_res.nodal_von_mises

    import os
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Problem Setup & BCs Plot
    fig1, ax1 = plt.subplots(figsize=(8, 3.5))
    plot_boundary_conditions(mesh, dof_map, dirichlet_bcs=dirichlet_bcs, ax=ax1)
    ax1.set_title("Problem Definition & Boundary Conditions Setup")
    fig1.tight_layout()
    fig1.savefig(os.path.join(output_dir, "thermoelastic_setup.png"), dpi=200)
    print("  Saved: examples/output/thermoelastic_setup.png")

    # 2. All Field Solutions Grid Plot
    fig2, axes = plt.subplots(2, 2, figsize=(12, 6))
    plot_scalar_field_2d(mesh, T_vals, title="Temperature T (°C)", ax=axes[0, 0])
    plot_scalar_field_2d(mesh, ux_vals * 1000.0, title="Horizontal Displacement u_x (mm)", ax=axes[0, 1])
    plot_scalar_field_2d(mesh, uy_vals * 1000.0, title="Vertical Displacement u_y (mm)", ax=axes[1, 0])
    plot_scalar_field_2d(mesh, sigma_vm_nodal / 1e6, title="von Mises Stress sigma_vm (MPa)", ax=axes[1, 1])
    fig2.suptitle("Coupled Thermoelastic Field Solutions (u, T, sigma)", fontsize=14, fontweight='bold')
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, "thermoelastic_fields.png"), dpi=200)
    print("  Saved: examples/output/thermoelastic_fields.png")

    # 3. Initial Frame vs Deformed Frame Plots
    fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(10, 7))
    plot_deformed_mesh(mesh, U_sol, dof_map, scale=200.0, field_values=T_vals, field_title="Temperature T (°C)", ax=ax3a)
    plot_deformed_mesh(mesh, U_sol, dof_map, scale=200.0, field_values=sigma_vm_nodal / 1e6, field_title="von Mises Stress (MPa)", ax=ax3b)
    fig3.tight_layout()
    fig3.savefig(os.path.join(output_dir, "thermoelastic_deformed_frames.png"), dpi=200)
    print("  Saved: examples/output/thermoelastic_deformed_frames.png")

    plt.close('all')
    print("\nPhase 3 coupled multi-field simulation and visualizations completed successfully!")

if __name__ == "__main__":
    main()
