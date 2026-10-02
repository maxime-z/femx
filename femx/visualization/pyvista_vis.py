import numpy as np
import pyvista as pv
from typing import Dict, Union, Tuple
from femx.core.mesh import Mesh, NurbsPatch
from femx.basis.nurbs import compute_nurbs_mapping

def to_pyvista_grid(geometry: Union[Mesh, NurbsPatch], values: np.ndarray = None, field_name: str = "Field") -> pv.UnstructuredGrid:
    """
    Convert a Mesh or NurbsPatch solution to a pyvista.UnstructuredGrid object.
    """
    if isinstance(geometry, Mesh):
        coords = geometry.coords
        cells = geometry.cells
        n_nodes = geometry.n_nodes
        n_elements = geometry.n_elements
        
        # PyVista/VTK coordinates must be 3D. Pad 2D coordinates with zeros in Z.
        if coords.shape[1] == 2:
            coords_3d = np.hstack([coords, np.zeros((n_nodes, 1))])
        else:
            coords_3d = coords
            
        nen = cells.shape[1]
        if nen == 4:
            size_prefix = np.full((n_elements, 1), 4, dtype=int)
            vtk_cells = np.hstack([size_prefix, cells]).ravel()
            cell_types = np.full(n_elements, 9, dtype=np.uint8)  # VTK_QUAD
            display_cells = cells
            display_coords = coords_3d
        elif nen == 8:
            size_prefix = np.full((n_elements, 1), 8, dtype=int)
            vtk_cells = np.hstack([size_prefix, cells]).ravel()
            cell_types = np.full(n_elements, 12, dtype=np.uint8)  # VTK_HEXAHEDRON
            display_cells = cells
            display_coords = coords_3d
        else:
            # Sample higher-order hexes onto corner-only linear hexes for display.
            display_coords, display_cells, values = _sample_higher_order_hex(
                coords_3d, cells, values
            )
            n_elements = len(display_cells)
            size_prefix = np.full((n_elements, 1), 8, dtype=int)
            vtk_cells = np.hstack([size_prefix, display_cells]).ravel()
            cell_types = np.full(n_elements, 12, dtype=np.uint8)
            
        grid = pv.UnstructuredGrid(vtk_cells, cell_types, display_coords)
        
        if values is not None:
            if len(np.shape(values)) > 1 and np.shape(values)[1] in (2, 3):
                vals = np.asarray(values)
                if vals.shape[1] == 2:
                    vals = np.hstack([vals, np.zeros((vals.shape[0], 1))])
                # For sampled higher-order grids, values were already remapped.
                if vals.shape[0] != display_coords.shape[0] and nen not in (4, 8):
                    pass
                grid.point_data[field_name] = vals
            else:
                grid.point_data[field_name] = np.asarray(values).ravel()
                
        return grid
        
    elif isinstance(geometry, NurbsPatch):
        if geometry.parametric_dim == 3:
            return _nurbs_volume_to_grid(geometry, values, field_name)
        # Sample the NURBS patch to create a dense quadrilateral unstructured grid
        n_samples = 12
        spans = geometry.get_element_spans()
        
        xi_pts = np.linspace(-1.0, 1.0, n_samples)
        eta_pts = np.linspace(-1.0, 1.0, n_samples)
        
        all_coords = []
        all_cells = []
        all_values = []
        
        flat_cps = geometry.flatten_control_points()
        node_offset = 0
        
        for span_u, span_v in spans:
            cell = geometry.get_element_control_points(span_u, span_v)
            local_sol = values[cell].ravel() if values is not None else None
            
            # 1. Sample coordinates and values on grid
            for j, eta in enumerate(eta_pts):
                for i, xi in enumerate(xi_pts):
                    R, _, _ = compute_nurbs_mapping(np.array([xi, eta]), geometry, span_u, span_v)
                    elem_coords = flat_cps[cell]
                    pt = R @ elem_coords
                    all_coords.append([pt[0], pt[1], 0.0] if len(pt) == 2 else list(pt))
                    
                    if local_sol is not None:
                        if np.ndim(values) > 1:
                            all_values.append(R @ values[cell])
                        else:
                            all_values.append(np.dot(R, local_sol))
                        
            # 2. Build local connectivity quad cells for this span
            for j in range(n_samples - 1):
                for i in range(n_samples - 1):
                    # Counter-clockwise quad cell
                    n0 = node_offset + j * n_samples + i
                    n1 = n0 + 1
                    n2 = n0 + n_samples + 1
                    n3 = n0 + n_samples
                    all_cells.append([n0, n1, n2, n3])
                    
            node_offset += n_samples * n_samples
            
        all_coords = np.array(all_coords)
        all_cells = np.array(all_cells)
        n_cells = len(all_cells)
        
        size_prefix = np.full((n_cells, 1), 4, dtype=int)
        vtk_cells = np.hstack([size_prefix, all_cells]).ravel()
        cell_types = np.full(n_cells, 9, dtype=np.uint8)
        
        grid = pv.UnstructuredGrid(vtk_cells, cell_types, all_coords)
        
        if values is not None:
            grid.point_data[field_name] = np.array(all_values)
            
        return grid
        
    else:
        raise TypeError("Geometry must be Mesh or NurbsPatch")


def _sample_higher_order_hex(coords_3d, cells, values):
    """Use the eight corner nodes of each higher-order hex for VTK display."""
    # Tensor-product local corners for p>1: i,j,k in {0,p}
    nen = cells.shape[1]
    p = int(round(nen ** (1.0 / 3.0))) - 1
    n_side = p + 1

    def lid(i, j, k):
        return i + n_side * j + n_side * n_side * k

    corner_local = [
        lid(0, 0, 0), lid(p, 0, 0), lid(p, p, 0), lid(0, p, 0),
        lid(0, 0, p), lid(p, 0, p), lid(p, p, p), lid(0, p, p),
    ]
    # Map to unique display nodes
    used = {}
    display_coords = []
    display_values = [] if values is not None else None
    display_cells = []
    for cell in cells:
        corners = []
        for loc in corner_local:
            g = int(cell[loc])
            if g not in used:
                used[g] = len(display_coords)
                display_coords.append(coords_3d[g])
                if values is not None:
                    display_values.append(values[g])
            corners.append(used[g])
        display_cells.append(corners)
    vals = np.asarray(display_values) if display_values is not None else None
    return np.asarray(display_coords), np.asarray(display_cells, dtype=int), vals


def _nurbs_volume_to_grid(geometry: NurbsPatch, values, field_name: str):
    """Sample a trivariate NURBS volume onto a linear hex grid for display."""
    n_samples = 4
    spans = geometry.get_element_spans()
    pts = np.linspace(-1.0, 1.0, n_samples)
    flat_cps = geometry.flatten_control_points()
    all_coords = []
    all_cells = []
    all_values = []
    node_offset = 0
    for span_u, span_v, span_w in spans:
        cell = geometry.get_element_control_points(span_u, span_v, span_w)
        for kk, zeta in enumerate(pts):
            for jj, eta in enumerate(pts):
                for ii, xi in enumerate(pts):
                    R, _, _ = compute_nurbs_mapping(
                        np.array([xi, eta, zeta]), geometry, span_u, span_v, span_w
                    )
                    pt = R @ flat_cps[cell]
                    all_coords.append(pt)
                    if values is not None:
                        all_values.append(R @ values[cell])
        for kk in range(n_samples - 1):
            for jj in range(n_samples - 1):
                for ii in range(n_samples - 1):
                    def nid(i, j, k):
                        return node_offset + i + n_samples * j + n_samples * n_samples * k
                    all_cells.append([
                        nid(ii, jj, kk), nid(ii + 1, jj, kk),
                        nid(ii + 1, jj + 1, kk), nid(ii, jj + 1, kk),
                        nid(ii, jj, kk + 1), nid(ii + 1, jj, kk + 1),
                        nid(ii + 1, jj + 1, kk + 1), nid(ii, jj + 1, kk + 1),
                    ])
        node_offset += n_samples ** 3
    all_coords = np.asarray(all_coords)
    all_cells = np.asarray(all_cells, dtype=int)
    size_prefix = np.full((len(all_cells), 1), 8, dtype=int)
    vtk_cells = np.hstack([size_prefix, all_cells]).ravel()
    cell_types = np.full(len(all_cells), 12, dtype=np.uint8)
    grid = pv.UnstructuredGrid(vtk_cells, cell_types, all_coords)
    if values is not None:
        grid.point_data[field_name] = np.asarray(all_values)
    return grid

def plot_pyvista(
    geometry: Union[Mesh, NurbsPatch], 
    values: np.ndarray = None, 
    field_name: str = "Field", 
    show_edges: bool = True
):
    """
    Open an interactive PyVista 3D plotter window.
    """
    grid = to_pyvista_grid(geometry, values, field_name)
    
    plotter = pv.Plotter()
    
    # Enable displacement warp for 2D/3D vector fields
    if values is not None and len(np.shape(values)) > 1 and np.shape(values)[1] in (2, 3):
        warped = grid.warp_by_vector(vectors=field_name, factor=1.0)
        plotter.add_mesh(warped, scalars=field_name, show_edges=show_edges, cmap='coolwarm')
        plotter.add_mesh(grid, style='wireframe', color='#adb5bd', opacity=0.5)
    else:
        plotter.add_mesh(grid, scalars=field_name if values is not None else None, show_edges=show_edges, cmap='coolwarm')
        
    plotter.show_axes()
    plotter.show()

def export_to_vtk(geometry: Union[Mesh, NurbsPatch], file_path: str, values: np.ndarray = None, field_name: str = "Field"):
    """
    Export the mesh/NURBS patch and associated fields to a VTK Unstructured (.vtu) file.
    This file can be opened directly with ParaView.
    """
    grid = to_pyvista_grid(geometry, values, field_name)
    grid.save(file_path)
    print(f"Mesh and field successfully exported to VTK: {file_path}")
