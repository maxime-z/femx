import numpy as np
import matplotlib.pyplot as plt
from femx.core.mesh import NurbsPatch
from femx.core.fields import FieldSpec
from femx.core.dofs import DofMap
from femx.core.assembly import assemble_system
from femx.materials.linear_heat import LinearHeatMaterial
from femx.formulations.heat import HeatConductionFormulation
from femx.solvers.linear import solve_system

def create_quadratic_nurbs_square() -> NurbsPatch:
    """Create a quadratic NURBS patch representing a unit square [0, 1]x[0, 1]."""
    # Open knot vectors for degree p=2
    knots_u = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    knots_v = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    
    # 3x3 Control points coordinate grid
    cp_x = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [1.0, 1.0, 1.0]
    ])
    cp_y = np.array([
        [0.0, 0.5, 1.0],
        [0.0, 0.5, 1.0],
        [0.0, 0.5, 1.0]
    ])
    control_points = np.stack([cp_x, cp_y], axis=-1)
    
    # Unit weights
    weights = np.ones((3, 3))
    
    # Boundary definitions as flat control point indices
    boundaries = {
        "left": np.array([0, 3, 6]),   # u_idx = 0
        "right": np.array([2, 5, 8]),  # u_idx = 2
        "bottom": np.array([0, 1, 2]), # v_idx = 0
        "top": np.array([6, 7, 8])     # v_idx = 2
    }
    
    return NurbsPatch(
        p_u=2, p_v=2,
        knots_u=knots_u, knots_v=knots_v,
        control_points=control_points,
        weights=weights,
        boundaries=boundaries
    )

def main():
    print("--- Solving Heat Conduction on Quadratic NURBS Patch ---")
    
    # 1. Create NURBS Patch
    patch = create_quadratic_nurbs_square()
    print(f"NURBS patch initialized with {patch.n_control_points} control points")
    
    # 2. Setup field and DofMap
    fields = [FieldSpec(name="T", components=1, location="control_points", unknown=True)]
    dof_map = DofMap(fields=fields, geometry=patch)
    
    # 3. Define material and formulation (rho=50, C=1.0, K=10.0)
    material = LinearHeatMaterial(rho=50.0, C=1.0, K=10.0)
    formulation = HeatConductionFormulation(material=material)
    
    # 4. Assemble system
    K, M, f = assemble_system(dof_map, formulation, field_name="T", body_load=0.0)
    print(f"System assembled. stiffness shape: {K.shape}")
    
    # 5. Apply Dirichlet BCs
    # Left boundary (x=0): T = 10.0 + 10.0 * y
    # Right boundary (x=1): T = 20.0 - 10.0 * y
    dirichlet_bcs = {}
    
    flat_cps = patch.control_points.transpose(1, 0, 2).reshape((-1, 2))
    
    # Left boundary
    for cp_idx in patch.boundaries["left"]:
        y = flat_cps[cp_idx, 1]
        T_val = 10.0 + 10.0 * y
        dof = dof_map.get_dof("T", cp_idx, 0)
        dirichlet_bcs[dof] = T_val
        
    # Right boundary
    for cp_idx in patch.boundaries["right"]:
        y = flat_cps[cp_idx, 1]
        T_val = 20.0 - 10.0 * y
        dof = dof_map.get_dof("T", cp_idx, 0)
        dirichlet_bcs[dof] = T_val
        
    # 6. Solve
    T_sol = solve_system(K, f, dirichlet_bcs)
    
    # Print results
    print("\nSolved Temperature at control points:")
    print("CP ID | Coords (x, y) | Temperature")
    print("-----------------------------------")
    for cp_idx in range(patch.n_control_points):
        x, y = flat_cps[cp_idx]
        print(f"{cp_idx:5d} | ({x:3.2f}, {y:3.2f})   | {T_sol[cp_idx]:8.4f}")

    # 7. Visualization and VTK Export
    from femx.visualization.matplotlib_vis import (
        plot_nurbs_geometry, plot_nurbs_scalar_field_2d
    )

    # 1. Problem Setup Plot
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    plot_nurbs_geometry(patch, show_control_grid=True, ax=ax1)
    ax1.set_title("NURBS Patch Geometry & Control Grid Setup")
    fig1.tight_layout()
    fig1.savefig("examples/heat_nurbs_setup.png", dpi=200)
    print("  Saved: examples/heat_nurbs_setup.png")

    # 2. Field Solution Contour Plot
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    plot_nurbs_scalar_field_2d(patch, T_sol, title="Temperature Field T (°C) (NURBS IGA)", ax=ax2)
    fig2.tight_layout()
    fig2.savefig("examples/heat_nurbs_fields.png", dpi=200)
    print("  Saved: examples/heat_nurbs_fields.png")

    plt.close('all')

    try:
        from femx.visualization.pyvista_vis import export_to_vtk
        export_to_vtk(patch, "examples/heat_nurbs_solution.vtu", T_sol, "Temperature")
    except Exception as e:
        print(f"Skipping VTK export: {e}")

if __name__ == "__main__":
    main()
