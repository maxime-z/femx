import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from typing import Dict, Union, List, Tuple
from femx.backends.numpy_backend import ndarray, array
from femx.core.mesh import Mesh, NurbsPatch
from femx.core.dofs import DofMap
from femx.basis.nurbs import compute_nurbs_mapping

def plot_mesh(mesh: Mesh, show_nodes: bool = True, boundary_colors: Dict[str, str] = None, ax = None):
    """
    Plot a 2D FEM Mesh.
    Args:
        mesh: Mesh object
        show_nodes: If True, plot node locations and labels
        boundary_colors: Dict mapping boundary name to color (e.g. {'left': 'blue'})
        ax: Matplotlib axes object (optional)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        
    coords = mesh.coords
    cells = mesh.cells
    
    # Plot elements
    polys = [coords[cell] for cell in cells]
    pc = PolyCollection(polys, edgecolors='black', facecolors='#f2f5f9', linewidths=1.0, alpha=0.9)
    ax.add_collection(pc)
    
    # Plot boundary nodes/edges if requested
    if boundary_colors and mesh.boundaries:
        for name, nodes in mesh.boundaries.items():
            color = boundary_colors.get(name, 'red')
            # Mark the boundary nodes
            ax.scatter(coords[nodes, 0], coords[nodes, 1], color=color, s=40, zorder=5, label=f"BC: {name}")
            
    # Plot node markers and labels
    if show_nodes:
        ax.scatter(coords[:, 0], coords[:, 1], color='#1d3557', s=20, zorder=4)
        for i, (x, y) in enumerate(coords):
            ax.text(x + 0.02, y + 0.02, str(i), color='#1d3557', fontsize=9, zorder=4)
            
    ax.autoscale_view()
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('FEM Mesh')
    if boundary_colors:
        ax.legend()
    return ax

def plot_boundary_conditions(
    mesh: Mesh, 
    dof_map: DofMap, 
    dirichlet_bcs: Dict[int, float] = None, 
    neumann_forces: Dict[int, float] = None, 
    ax = None
):
    """
    Visualize Dirichlet constraints and Neumann force arrows.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        
    # Plot base mesh outlines first
    plot_mesh(mesh, show_nodes=False, ax=ax)
    
    # Parse Dirichlet BCs
    if dirichlet_bcs:
        # Map dof index back to coordinates
        constrained_coords = []
        for dof, val in dirichlet_bcs.items():
            for name, offset in dof_map.field_offsets.items():
                spec = dof_map.field_specs[name]
                n_ent = dof_map.field_entities[name]
                length = n_ent * spec.components
                if offset <= dof < offset + length:
                    rel_dof = dof - offset
                    node = rel_dof // spec.components
                    comp = rel_dof % spec.components
                    x, y = mesh.coords[node]
                    constrained_coords.append((x, y, comp, val))
                    
        # Plot constraint markers (X constraints as red triangles, Y as green, scalar as magenta circles)
        for x, y, comp, val in constrained_coords:
            if comp == 0:  # X or Scalar
                ax.scatter(x, y, color='#e63946', marker='X', s=70, zorder=6, label='X constraint' if 'X constraint' not in ax.get_legend_handles_labels()[1] else "")
            elif comp == 1: # Y
                ax.scatter(x, y, color='#2a9d8f', marker='^', s=70, zorder=6, label='Y constraint' if 'Y constraint' not in ax.get_legend_handles_labels()[1] else "")

    # Parse Neumann point forces
    if neumann_forces:
        for dof, val in neumann_forces.items():
            if val == 0.0:
                continue
            for name, offset in dof_map.field_offsets.items():
                spec = dof_map.field_specs[name]
                n_ent = dof_map.field_entities[name]
                length = n_ent * spec.components
                if offset <= dof < offset + length:
                    rel_dof = dof - offset
                    node = rel_dof // spec.components
                    comp = rel_dof % spec.components
                    x, y = mesh.coords[node]
                    
                    dx = val if comp == 0 else 0.0
                    dy = val if comp == 1 else 0.0
                    
                    # Normalize arrow size visually
                    scale = 0.1 / abs(val) if val != 0.0 else 1.0
                    ax.arrow(x - dx*scale, y - dy*scale, dx*scale, dy*scale, 
                             color='#e07a5f', head_width=0.04, head_length=0.06, zorder=7,
                             label='Force Load' if 'Force Load' not in ax.get_legend_handles_labels()[1] else "")
                    
    ax.legend(loc='upper right')
    ax.set_title('Mesh Boundary Conditions Setup')
    return ax

def plot_scalar_field_2d(mesh: Mesh, values: ndarray, title: str = "Scalar Field", ax = None):
    """
    Plot a smooth filled contour of a nodal scalar field (e.g. temperature).
    Splits quadrilateral elements into triangles for Matplotlib triangulation.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        
    coords = mesh.coords
    cells = mesh.cells
    
    # Split each quad cell [n0, n1, n2, n3] into two triangles [n0, n1, n2] and [n0, n2, n3]
    triangles = []
    for cell in cells:
        triangles.append([cell[0], cell[1], cell[2]])
        triangles.append([cell[0], cell[2], cell[3]])
        
    triangles = np.array(triangles)
    
    # Create triangulation
    import matplotlib.tri as tri
    triangulation = tri.Triangulation(coords[:, 0], coords[:, 1], triangles)
    
    # Plot contours
    cf = ax.tricontourf(triangulation, values.ravel(), cmap='coolwarm', levels=20)
    plt.colorbar(cf, ax=ax, label=title)
    
    # Draw element outlines
    polys = [coords[cell] for cell in cells]
    pc = PolyCollection(polys, edgecolors='black', facecolors='none', linewidths=0.5, alpha=0.5)
    ax.add_collection(pc)
    
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    return ax

def plot_deformed_mesh(
    mesh: Mesh,
    u_vector: ndarray,
    dof_map: DofMap,
    scale: float = 1.0,
    field_values: ndarray = None,
    field_title: str = "Deformed Mesh",
    cmap: str = "coolwarm",
    ax = None
):
    """
    Plot initial mesh outline (dashed) vs deformed mesh (solid), optionally overlaid with field values (e.g. von Mises stress, temperature).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
        
    coords = mesh.coords
    cells = mesh.cells
    N = len(coords)
    
    # Extract displacement vector [ux, uy] per node
    ux = np.array([u_vector[dof_map.get_dof("u", i, 0)] for i in range(N)])
    uy = np.array([u_vector[dof_map.get_dof("u", i, 1)] for i in range(N)])
    disp = np.vstack([ux, uy]).T
    deformed_coords = coords + scale * disp
    
    # 1. Plot initial mesh outline (dashed light gray)
    init_polys = [coords[cell] for cell in cells]
    init_pc = PolyCollection(init_polys, edgecolors='#a8dadc', facecolors='none', linestyles='dashed', linewidths=1.0, alpha=0.7, label='Initial Frame')
    ax.add_collection(init_pc)
    
    # 2. Plot deformed mesh
    def_polys = [deformed_coords[cell] for cell in cells]
    
    if field_values is not None:
        # Plot scalar field on deformed mesh
        import matplotlib.tri as tri
        triangles = []
        for cell in cells:
            triangles.append([cell[0], cell[1], cell[2]])
            triangles.append([cell[0], cell[2], cell[3]])
        triangulation = tri.Triangulation(deformed_coords[:, 0], deformed_coords[:, 1], np.array(triangles))
        cf = ax.tricontourf(triangulation, field_values.ravel(), cmap=cmap, levels=20)
        plt.colorbar(cf, ax=ax, label=field_title)
        def_pc = PolyCollection(def_polys, edgecolors='black', facecolors='none', linewidths=0.6, alpha=0.6, label='Deformed Frame')
        ax.add_collection(def_pc)
    else:
        def_pc = PolyCollection(def_polys, edgecolors='#1d3557', facecolors='#f1faee', linewidths=1.2, alpha=0.8, label='Deformed Frame')
        ax.add_collection(def_pc)
        
    ax.autoscale_view()
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f"{field_title} (Deformation Scale: {scale:.1f}x)")
    ax.legend(loc='upper right')
    return ax

def plot_nurbs_geometry(patch: NurbsPatch, show_control_grid: bool = True, ax = None):
    """
    Plot IGA NURBS geometry boundaries and the control grid net.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        
    n_cp_u, n_cp_v = patch.n_cp_u, patch.n_cp_v
    cps = patch.control_points
    
    # 1. Plot control point net lines
    if show_control_grid:
        lines = []
        for j in range(n_cp_v):
            for i in range(n_cp_u - 1):
                lines.append([cps[i, j], cps[i+1, j]])
        for i in range(n_cp_u):
            for j in range(n_cp_v - 1):
                lines.append([cps[i, j], cps[i, j+1]])
                
        lc = LineCollection(lines, colors='#adb5bd', linestyles='dashed', linewidths=1.0, zorder=2)
        ax.add_collection(lc)
        
        # Plot control points
        flat_cps = cps.reshape((-1, 2))
        ax.scatter(flat_cps[:, 0], flat_cps[:, 1], color='#e63946', s=40, zorder=3, label="Control Points")
        
        # Label control points with their 1D indices (Fortran-style: idx_u + idx_v * n_cp_u)
        for j in range(n_cp_v):
            for i in range(n_cp_u):
                idx_1d = i + j * n_cp_u
                x, y = cps[i, j]
                ax.text(x + 0.02, y + 0.02, str(idx_1d), color='#e63946', fontsize=9, zorder=3)
                
    # 2. Draw physical NURBS boundaries by sampling edge spans
    # Sample 50 points in each parametric direction
    samples = np.linspace(-1, 1, 30)
    spans = patch.get_element_spans()
    
    boundary_points = []
    
    # Extract outer boundaries of active element spans
    # For a simple unit patch, this corresponds to outer boundaries of parametric space
    # Let's sample the 4 physical boundaries
    for span_u, span_v in spans:
        # Determine parametric bounds of the span
        u1, u2 = patch.knots_u[span_u], patch.knots_u[span_u + 1]
        v1, v2 = patch.knots_v[span_v], patch.knots_v[span_v + 1]
        
        # We check if span lies on patch edges
        is_bottom = (v1 == patch.knots_v[patch.p_v])
        is_top = (v2 == patch.knots_v[-patch.p_v - 1])
        is_left = (u1 == patch.knots_u[patch.p_u])
        is_right = (u2 == patch.knots_u[-patch.p_u - 1])
        
        # Sample edges in physical coordinates
        for s in samples:
            # Bottom edge (v_ref = -1)
            if is_bottom:
                from femx.basis.nurbs import compute_nurbs_mapping
                _, _, _, _ = True, patch, span_u, span_v # parameters dummy
                R, _, _ = compute_nurbs_mapping(array([s, -1.0]), patch, span_u, span_v)
                cell = patch.get_element_control_points(span_u, span_v)
                coords = patch.control_points.transpose(1, 0, 2).reshape((-1, 2))[cell]
                pt = R @ coords
                boundary_points.append(pt)
            # Top edge (v_ref = 1.0)
            if is_top:
                R, _, _ = compute_nurbs_mapping(array([s, 1.0]), patch, span_u, span_v)
                cell = patch.get_element_control_points(span_u, span_v)
                coords = patch.control_points.transpose(1, 0, 2).reshape((-1, 2))[cell]
                pt = R @ coords
                boundary_points.append(pt)
            # Left edge (u_ref = -1.0)
            if is_left:
                R, _, _ = compute_nurbs_mapping(array([-1.0, s]), patch, span_u, span_v)
                cell = patch.get_element_control_points(span_u, span_v)
                coords = patch.control_points.transpose(1, 0, 2).reshape((-1, 2))[cell]
                pt = R @ coords
                boundary_points.append(pt)
            # Right edge (u_ref = 1.0)
            if is_right:
                R, _, _ = compute_nurbs_mapping(array([1.0, s]), patch, span_u, span_v)
                cell = patch.get_element_control_points(span_u, span_v)
                coords = patch.control_points.transpose(1, 0, 2).reshape((-1, 2))[cell]
                pt = R @ coords
                boundary_points.append(pt)
                
    if boundary_points:
        boundary_points = np.array(boundary_points)
        ax.scatter(boundary_points[:, 0], boundary_points[:, 1], color='#1d3557', s=4, zorder=4, label="NURBS Geometry")

    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('NURBS Patch Geometry')
    ax.legend()
    return ax

def plot_nurbs_scalar_field_2d(
    patch: NurbsPatch, 
    values: ndarray, 
    n_samples: int = 10, 
    title: str = "NURBS Scalar Field", 
    ax = None
):
    """
    Plot high-fidelity filled contours of a scalar field on a NURBS patch.
    Samples each active element span and interpolates the field values.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        
    spans = patch.get_element_spans()
    
    # Dense reference points on a single span
    xi_pts = np.linspace(-1.0, 1.0, n_samples)
    eta_pts = np.linspace(-1.0, 1.0, n_samples)
    
    dense_coords = []
    dense_vals = []
    
    flat_cps = patch.control_points.transpose(1, 0, 2).reshape((-1, 2))
    
    for span_u, span_v in spans:
        cell = patch.get_element_control_points(span_u, span_v)
        # Solutions on local control points
        local_vals = values[cell].ravel()
        
        for eta in eta_pts:
            for xi in xi_pts:
                gp = array([xi, eta])
                R, _, _ = compute_nurbs_mapping(gp, patch, span_u, span_v)
                
                # Physical position
                elem_coords = flat_cps[cell]
                pt = R @ elem_coords
                
                # Interpolated field value
                val = np.dot(R, local_vals)
                
                dense_coords.append(pt)
                dense_vals.append(val)
                
    dense_coords = np.array(dense_coords)
    dense_vals = np.array(dense_vals)
    
    # Triangulate dense sampled grid for plotting
    import matplotlib.tri as tri
    triangulation = tri.Triangulation(dense_coords[:, 0], dense_coords[:, 1])
    
    # Plot contours
    cf = ax.tricontourf(triangulation, dense_vals, cmap='coolwarm', levels=20)
    plt.colorbar(cf, ax=ax, label=title)
    
    # Overlay control point net
    plot_nurbs_geometry(patch, show_control_grid=True, ax=ax)
    
    ax.set_title(title)
    return ax
