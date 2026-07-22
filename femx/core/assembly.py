import numpy as np
import scipy.sparse as sp
from femx.core.dofs import DofMap
from femx.core.mesh import Mesh, NurbsPatch
from femx.core.quadrature import get_quadrature_2d
from femx.formulations.base import Formulation

def assemble_system(dof_map: DofMap, formulation: Formulation, field_name: str = None, body_load=None):
    """
    Assemble the global stiffness matrix K, mass matrix M, and force vector f.
    Args:
        dof_map: DofMap mapping degrees of freedom to equations
        formulation: Physical formulation (single-field or coupled multi-field)
        field_name: Name of field (if single field) or None (if formulation specifies field_names)
        body_load: Body load/forcing term
    """
    n_dofs = dof_map.n_dofs
    geometry = dof_map.geometry

    # Determine field names list
    if hasattr(formulation, "field_names") and isinstance(formulation.field_names, list):
        field_names = formulation.field_names
    elif field_name is not None:
        field_names = [field_name]
    else:
        raise ValueError("Must provide field_name or use a formulation with field_names defined.")
    
    rows = []
    cols = []
    data = []
    
    mass_rows = []
    mass_cols = []
    mass_data = []
    
    f_global = np.zeros(n_dofs)
    
    if isinstance(geometry, Mesh):
        # Q1 FEM Mesh Assembly
        cells = geometry.cells
        coords = geometry.coords
        quad_pts, quad_wts = get_quadrature_2d(2, 2)
        
        for elem_idx, cell in enumerate(cells):
            elem_coords = coords[cell]  # shape (4, 2)
            elem_dofs = dof_map.get_element_dofs_multi(field_names, cell)
            
            # Compute element matrices
            Ke, Me, fe = formulation.compute_element_matrices(
                elem_coords, quad_pts, quad_wts, body_load=body_load
            )
            
            # Append elements to global COO coordinate buffers
            r, c = np.meshgrid(elem_dofs, elem_dofs, indexing='ij')
            rows.extend(r.ravel())
            cols.extend(c.ravel())
            data.extend(Ke.ravel())
            
            mass_rows.extend(r.ravel())
            mass_cols.extend(c.ravel())
            mass_data.extend(Me.ravel())
            
            # Accumulate elemental forces into global load vector
            np.add.at(f_global, elem_dofs, fe)
            
    elif isinstance(geometry, NurbsPatch):
        # 2D IGA NURBS Patch Assembly
        # Quadratic NURBS elements use 3x3 Gauss-Legendre quadrature
        quad_pts, quad_wts = get_quadrature_2d(geometry.p_u + 1, geometry.p_v + 1)
        spans = geometry.get_element_spans()
        
        # Flatten CP coordinates to match Fortran-like 1D indexing (u-direction contiguous):
        # (n_cp_u, n_cp_v, 2) -> transpose to (n_cp_v, n_cp_u, 2) -> reshape to (n_cp_u * n_cp_v, 2)
        flat_cp_coords = geometry.control_points.transpose(1, 0, 2).reshape((-1, 2))
        
        for span_u, span_v in spans:
            cell = geometry.get_element_control_points(span_u, span_v)
            elem_coords = flat_cp_coords[cell]
            elem_dofs = dof_map.get_element_dofs_multi(field_names, cell)
            
            # Compute element matrices (using the is_nurbs flag)
            Ke, Me, fe = formulation.compute_element_matrices(
                elem_coords=elem_coords,
                quadrature_pts=quad_pts,
                quadrature_wts=quad_wts,
                is_nurbs=True,
                patch=geometry,
                span_u=span_u,
                span_v=span_v,
                body_load=body_load if body_load is not None else 0.0
            )
            
            # Append elements to global COO coordinate buffers
            r, c = np.meshgrid(elem_dofs, elem_dofs, indexing='ij')
            rows.extend(r.ravel())
            cols.extend(c.ravel())
            data.extend(Ke.ravel())
            
            mass_rows.extend(r.ravel())
            mass_cols.extend(c.ravel())
            mass_data.extend(Me.ravel())
            
            np.add.at(f_global, elem_dofs, fe)
            
    else:
        raise TypeError("Geometry must be Mesh or NurbsPatch")
        
    K_global = sp.coo_matrix((data, (rows, cols)), shape=(n_dofs, n_dofs)).tocsr()
    M_global = sp.coo_matrix((mass_data, (mass_rows, mass_cols)), shape=(n_dofs, n_dofs)).tocsr()
    
    return K_global, M_global, f_global
