import numpy as np
import matplotlib.pyplot as plt
from femx.core.mesh import Mesh
from femx.core.fields import FieldSpec
from femx.core.dofs import DofMap
from femx.visualization.matplotlib_vis import plot_boundary_conditions

def create_cantilever_mesh(Lx: float, Ly: float, nx: int, ny: int) -> Mesh:
    """Create a Q1 grid for a cantilever beam."""
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
    left_nodes = np.arange(ny + 1) * (nx + 1)
    right_nodes = np.arange(ny + 1) * (nx + 1) + nx
    
    boundaries = {
        "left": left_nodes,
        "right": right_nodes
    }
    return Mesh(coords=coords, cells=cells, boundaries=boundaries)

def main():
    print("--- Visualizing Cantilever Beam Setup (Mesh & BCs) ---")
    
    # 1. Create cantilever mesh
    mesh = create_cantilever_mesh(4.0, 1.0, 8, 2)
    
    # 2. Setup field and DofMap
    fields = [FieldSpec(name="u", components=2, location="nodes", unknown=True)]
    dof_map = DofMap(fields=fields, geometry=mesh)
    
    # 3. Setup Dirichlet boundary conditions: clamp left end (X and Y fixed)
    dirichlet_bcs = {}
    for node in mesh.boundaries["left"]:
        dof_x = dof_map.get_dof("u", node, 0)
        dof_y = dof_map.get_dof("u", node, 1)
        dirichlet_bcs[dof_x] = 0.0
        dirichlet_bcs[dof_y] = 0.0
        
    # 4. Setup Neumann point load: downward force F_y = -100.0 at the right tip nodes
    neumann_forces = {}
    for node in mesh.boundaries["right"]:
        dof_y = dof_map.get_dof("u", node, 1)
        neumann_forces[dof_y] = -100.0
        
    # 5. Plot setup
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_boundary_conditions(
        mesh=mesh,
        dof_map=dof_map,
        dirichlet_bcs=dirichlet_bcs,
        neumann_forces=neumann_forces,
        ax=ax
    )
    
    import os
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    image_path = os.path.join(output_dir, "mesh_setup.png")
    plt.savefig(image_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Mesh setup plot successfully saved to: examples/output/mesh_setup.png")

if __name__ == "__main__":
    main()
