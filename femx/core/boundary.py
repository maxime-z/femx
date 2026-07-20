import numpy as np
from typing import Dict, Union, List, Tuple, Callable
from femx.backends.numpy_backend import ndarray, array, zeros
from femx.core.mesh import Mesh
from femx.core.dofs import DofMap
from femx.core.quadrature import get_quadrature_1d

def compute_boundary_edges(mesh: Mesh, boundary_name: str) -> List[Tuple[int, int]]:
    """
    Extract boundary line segment node pairs [(n0, n1), (n1, n2), ...] for a given boundary.
    Supports either boundary defined as an ordered list of nodes or explicit edge pairs.
    """
    if boundary_name not in mesh.boundaries:
        raise ValueError(f"Boundary '{boundary_name}' not found in mesh. Available: {list(mesh.boundaries.keys())}")
        
    boundary_nodes = mesh.boundaries[boundary_name]
    edges = []
    
    # Check if boundary nodes are consecutiveAlong standard grid edges
    if len(boundary_nodes) < 2:
        return edges
        
    for i in range(len(boundary_nodes) - 1):
        n0 = boundary_nodes[i]
        n1 = boundary_nodes[i + 1]
        edges.append((n0, n1))
        
    return edges

def integrate_neumann_traction(
    mesh: Mesh,
    dof_map: DofMap,
    field_name: str,
    boundary_name: str,
    traction: Union[ndarray, float, Callable[[ndarray], ndarray]],
    n_quad_pts: int = 2
) -> ndarray:
    """
    Integrate Neumann traction or flux along a 2D mesh boundary.
    Args:
        mesh: Mesh object
        dof_map: DofMap mapping nodes to global equation numbers
        field_name: Name of target field (e.g. 'u' or 'T')
        boundary_name: Boundary tag (e.g. 'right' or 'top')
        traction: Constant vector/scalar, or callable function t(x, y) returning load array
        n_quad_pts: 1D Gauss-Legendre quadrature points (default=2)
    Returns:
        f_boundary: Global force vector of shape (n_dofs,) containing Neumann contributions
    """
    f_boundary = zeros(dof_map.n_dofs)
    spec = dof_map.field_specs[field_name]
    components = spec.components
    
    edges = compute_boundary_edges(mesh, boundary_name)
    pts_1d, wts_1d = get_quadrature_1d(n_quad_pts)
    
    for n0, n1 in edges:
        p0 = mesh.coords[n0]
        p1 = mesh.coords[n1]
        
        # Edge length L and Jacobian detJ_1d = L / 2
        edge_vec = p1 - p0
        L = np.linalg.norm(edge_vec)
        detJ_1d = 0.5 * L
        
        # Local DOFs for node 0 and node 1
        dofs_n0 = [dof_map.get_dof(field_name, n0, c) for c in range(components)]
        dofs_n1 = [dof_map.get_dof(field_name, n1, c) for c in range(components)]
        
        for xi, w in zip(pts_1d, wts_1d):
            # 1D linear shape functions: N0 = (1-xi)/2, N1 = (1+xi)/2
            N0 = 0.5 * (1.0 - xi)
            N1 = 0.5 * (1.0 + xi)
            
            # Physical coordinate on edge
            p_quad = N0 * p0 + N1 * p1
            
            # Evaluate traction
            if callable(traction):
                t_val = array(traction(p_quad))
            elif isinstance(traction, (int, float)):
                t_val = array([float(traction)])
            else:
                t_val = array(traction)
                
            dV = w * detJ_1d
            
            # Accumulate vector/scalar force to node DOFs
            for c in range(components):
                val_c = t_val[c] if components > 1 else t_val[0]
                f_boundary[dofs_n0[c]] += N0 * val_c * dV
                f_boundary[dofs_n1[c]] += N1 * val_c * dV
                
    return f_boundary
