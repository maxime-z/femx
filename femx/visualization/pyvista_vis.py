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
            
        # VTK cells format: [n_points_cell0, p0, p1, ..., pN, n_points_cell1, ...]
        # Q1 quadrilateral elements have 4 points.
        size_prefix = np.full((n_elements, 1), 4, dtype=int)
        vtk_cells = np.hstack([size_prefix, cells]).ravel()
        
        # Cell types: 9 corresponds to vtk.VTK_QUAD
        cell_types = np.full(n_elements, 9, dtype=np.uint8)
        
        grid = pv.UnstructuredGrid(vtk_cells, cell_types, coords_3d)
        
        if values is not None:
            # If values is a vector field (like displacement), assign it directly as a 2D/3D array
            if len(values.shape) > 1 and values.shape[1] == 2:
                # Pad displacement values with zeros in Z for 3D visualization compatibility
                values_3d = np.hstack([values, np.zeros((n_nodes, 1))])
                grid.point_data[field_name] = values_3d
            else:
                grid.point_data[field_name] = values.ravel()
                
        return grid
        
    elif isinstance(geometry, NurbsPatch):
        # Sample the NURBS patch to create a dense quadrilateral unstructured grid
        n_samples = 12
        spans = geometry.get_element_spans()
        
        xi_pts = np.linspace(-1.0, 1.0, n_samples)
        eta_pts = np.linspace(-1.0, 1.0, n_samples)
        
        all_coords = []
        all_cells = []
        all_values = []
        
        flat_cps = geometry.control_points.transpose(1, 0, 2).reshape((-1, 2))
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
                    all_coords.append([pt[0], pt[1], 0.0])
                    
                    if local_sol is not None:
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
    
    # Enable displacement warp if it is a 2D displacement field
    if values is not None and len(values.shape) > 1 and values.shape[1] == 2:
        # Warp by displacement vector
        warped = grid.warp_by_vector(vectors=field_name, factor=1.0)
        plotter.add_mesh(warped, scalars=field_name, show_edges=show_edges, cmap='coolwarm')
        # Add wireframe of undeformed mesh as reference
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
