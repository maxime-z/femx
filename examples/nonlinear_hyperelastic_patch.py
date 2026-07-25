import os
import numpy as np
import matplotlib.pyplot as plt

from femx.core.mesh import Mesh
from femx.core.fields import FieldSpec
from femx.core.dofs import DofMap
from femx.core.state import State
from femx.materials.hyperelastic import NeoHookeanMaterial
from femx.formulations.hyperelasticity import HyperelasticFormulation
from femx.solvers.nonlinear import NewtonSolver, LoadStepper
from femx.visualization.matplotlib_vis import (
    plot_boundary_conditions, plot_scalar_field_2d, plot_deformed_mesh
)

def create_square_mesh(Lx: float, Ly: float, nx: int, ny: int) -> Mesh:
    """Create a structured Q1 grid for a 2D domain."""
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
            
    cells = np.array(cells, dtype=int)
    
    bottom_nodes = np.arange(nx + 1)
    top_nodes = ny * (nx + 1) + np.arange(nx + 1)
    left_nodes = np.arange(ny + 1) * (nx + 1)
    right_nodes = np.arange(ny + 1) * (nx + 1) + nx
    
    boundaries = {
        "bottom": bottom_nodes,
        "top": top_nodes,
        "left": left_nodes,
        "right": right_nodes
    }
    
    return Mesh(coords=coords, cells=cells, boundaries=boundaries)

def main():
    print("--- Solving Hyperelastic Patch Test (Simple Shear Large Deformation) ---")
    
    # 1. Mesh setup: 1.0 x 1.0 square with 4x4 elements
    Lx, Ly = 1.0, 1.0
    nx, ny = 4, 4
    mesh = create_square_mesh(Lx, Ly, nx, ny)
    print(f"Mesh generated: {mesh.n_nodes} nodes, {mesh.n_elements} elements")
    
    # 2. Setup field and DofMap
    fields = [FieldSpec(name="u", components=2, location="nodes", unknown=True)]
    dof_map = DofMap(fields=fields, geometry=mesh)
    
    # 3. Material and Formulation setup (Neo-Hookean: E=1e5, nu=0.3)
    material = NeoHookeanMaterial(rho=1.0, E=1.0e5, nu=0.3)
    formulation = HyperelasticFormulation(material=material)
    
    # 4. State initialization
    state = State()
    state.initialize_field("u", mesh.n_nodes, 2)
    
    # 5. Boundary conditions:
    # - Fix bottom (y=0) in y-direction: uy = 0
    # - Fix bottom-left node (0,0) in x-direction: ux = 0
    # - Prescribe top (y=1) with large horizontal shear displacement: ux = 0.6
    dirichlet_bcs = {}
    for node in mesh.boundaries["bottom"]:
        dof_y = dof_map.get_dof("u", node, 1)
        dirichlet_bcs[dof_y] = 0.0
        
    # Fix x-displacement at bottom-left corner
    bl_node = mesh.boundaries["bottom"][0]
    dof_x_bl = dof_map.get_dof("u", bl_node, 0)
    dirichlet_bcs[dof_x_bl] = 0.0
    
    # Large shear displacement on top boundary
    top_shear_disp = 0.6
    for node in mesh.boundaries["top"]:
        dof_x = dof_map.get_dof("u", node, 0)
        dirichlet_bcs[dof_x] = top_shear_disp
        
    # 6. Solve using LoadStepper with 6 pseudo-time steps
    print(f"\nSolving static nonlinear hyperelasticity with LoadStepper...")
    newton_solver = NewtonSolver(rtol=1e-6, atol=1e-8, max_iter=20)
    stepper = LoadStepper(n_steps=6, newton_solver=newton_solver)
    
    final_state, state_history = stepper.solve(dof_map, formulation, state, dirichlet_bcs)
    print("Nonlinear solution completed successfully!")
    
    # Unpack final vector
    u_sol = final_state.pack_vector(dof_map)
    
    # Output dir setup
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Setup & BCs
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    plot_boundary_conditions(mesh, dof_map, dirichlet_bcs=dirichlet_bcs, ax=ax1)
    ax1.set_title("Hyperelastic Simple Shear Setup")
    fig1.tight_layout()
    fig1.savefig(os.path.join(output_dir, "hyperelastic_patch_setup.png"), dpi=200)
    print("  Saved: examples/output/hyperelastic_patch_setup.png")
    
    # Plot 2: Fields and Deformed shape
    N = mesh.n_nodes
    ux_vals = np.array([u_sol[dof_map.get_dof("u", i, 0)] for i in range(N)])
    uy_vals = np.array([u_sol[dof_map.get_dof("u", i, 1)] for i in range(N)])
    disp_norm = np.sqrt(ux_vals**2 + uy_vals**2)
    
    fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_scalar_field_2d(mesh, disp_norm, title="Displacement Magnitude ||u||", ax=axes[0])
    plot_deformed_mesh(mesh, u_sol, dof_map, scale=1.0, field_values=disp_norm, field_title="Displacement Magnitude", ax=axes[1])
    fig2.suptitle("Neo-Hookean Large Deformation Solution (Simple Shear)", fontsize=14, fontweight='bold')
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, "hyperelastic_patch_solution.png"), dpi=200)
    print("  Saved: examples/output/hyperelastic_patch_solution.png")
    
    plt.close('all')

if __name__ == "__main__":
    main()
