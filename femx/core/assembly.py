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
    if hasattr(formulation, "field_names") and formulation.field_names is not None:
        field_names = list(formulation.field_names)
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


def assemble_nonlinear_system(dof_map: DofMap, formulation: Formulation, state, body_load=None):
    """
    Assemble the global tangent stiffness matrix K and global residual vector R for a given state.
    Supports single-field and multi-field (u, p) formulations on Mesh and NurbsPatch.
    
    Args:
        dof_map: DofMap mapping degrees of freedom to equations
        formulation: Physical formulation with compute_element_residual_and_tangent method
        state: State object containing current field values
        body_load: Body load/forcing term
        
    Returns:
        K_global: csr_matrix of shape (n_dofs, n_dofs)
        R_global: ndarray of shape (n_dofs,)
    """
    n_dofs = dof_map.n_dofs
    geometry = dof_map.geometry
    field_names = list(formulation.field_names)
    
    rows = []
    cols = []
    data = []
    R_global = np.zeros(n_dofs)
    
    if isinstance(geometry, Mesh):
        cells = geometry.cells
        coords = geometry.coords
        quad_pts, quad_wts = get_quadrature_2d(2, 2)
        
        for elem_idx, cell in enumerate(cells):
            elem_coords = coords[cell]
            elem_dofs = dof_map.get_element_dofs_multi(field_names, cell)
            elem_u = state.values["u"][cell]
            
            kwargs = {}
            if "p" in field_names and "p" in state.values:
                kwargs["elem_p"] = state.values["p"][cell]
            
            Re, Ke = formulation.compute_element_residual_and_tangent(
                elem_coords, elem_u, quad_pts, quad_wts, body_load=body_load, **kwargs
            )
            
            r, c = np.meshgrid(elem_dofs, elem_dofs, indexing='ij')
            rows.extend(r.ravel())
            cols.extend(c.ravel())
            data.extend(Ke.ravel())
            
            np.add.at(R_global, elem_dofs, Re)
            
    elif isinstance(geometry, NurbsPatch):
        from femx.basis.nurbs import NurbsBasis
        # IGA NURBS Patch assembly for nonlinear system
        quad_pts, quad_wts = get_quadrature_2d(geometry.p_u + 1, geometry.p_v + 1)
        spans = geometry.get_element_spans()
        flat_cp_coords = geometry.control_points.transpose(1, 0, 2).reshape((-1, 2))
        
        for span_u, span_v in spans:
            cell_u = geometry.get_element_control_points(span_u, span_v)
            elem_coords = flat_cp_coords[cell_u]
            
            # Check if pressure geometry is a separate patch or same
            p_geom = dof_map.geometries.get("p", geometry)
            if p_geom is geometry:
                cell_p = cell_u
                span_u_p, span_v_p = span_u, span_v
            else:
                flat_cp_coords_p = p_geom.control_points.transpose(1, 0, 2).reshape((-1, 2))
                # Map geometric element using center parametric coordinate
                u_c = 0.5 * (geometry.knot_vectors[0].knots[span_u] + geometry.knot_vectors[0].knots[span_u+1])
                v_c = 0.5 * (geometry.knot_vectors[1].knots[span_v] + geometry.knot_vectors[1].knots[span_v+1])
                span_u_p = p_geom.knot_vectors[0].find_span(p_geom.degrees[0], u_c)
                span_v_p = p_geom.knot_vectors[1].find_span(p_geom.degrees[1], v_c)
                cell_p = p_geom.get_element_control_points(span_u_p, span_v_p)

            cell_dict = {"u": cell_u, "p": cell_p} if "p" in field_names else cell_u
            elem_dofs = dof_map.get_element_dofs_multi(field_names, cell_dict)
            
            elem_u = state.values["u"][cell_u]
            kwargs = {}
            if "p" in field_names and "p" in state.values:
                kwargs["elem_p"] = state.values["p"][cell_p]
            
            kwargs["elem_basis_u"] = NurbsBasis(geometry, span_u, span_v)
            if "p" in field_names:
                kwargs["elem_basis_p"] = NurbsBasis(p_geom, span_u_p, span_v_p)
            else:
                kwargs["elem_basis"] = NurbsBasis(geometry, span_u, span_v)
            
            # Evaluate using NurbBasis / patch
            Re, Ke = formulation.compute_element_residual_and_tangent(
                elem_coords, elem_u, quad_pts, quad_wts, body_load=body_load, **kwargs
            )
            
            r, c = np.meshgrid(elem_dofs, elem_dofs, indexing='ij')
            rows.extend(r.ravel())
            cols.extend(c.ravel())
            data.extend(Ke.ravel())
            
            np.add.at(R_global, elem_dofs, Re)
            
    else:
        raise TypeError("Geometry must be Mesh or NurbsPatch")
        
    K_global = sp.coo_matrix((data, (rows, cols)), shape=(n_dofs, n_dofs)).tocsr()
    return K_global, R_global


