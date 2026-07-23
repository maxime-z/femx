import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple
from femx.backends.numpy_backend import ndarray, zeros
from femx.core.mesh import Mesh, NurbsPatch
from femx.core.dofs import DofMap
from femx.core.quadrature import get_quadrature_2d
from femx.formulations.elasticity import LinearElasticityFormulation
from femx.basis.lagrange import LagrangeQuad

@dataclass
class PostprocessResult:
    """Container for postprocessed strains and stresses."""
    gauss_strains: ndarray     # shape (n_elements, n_gps, 3)
    gauss_stresses: ndarray    # shape (n_elements, n_gps, 3)
    gauss_von_mises: ndarray   # shape (n_elements, n_gps)
    nodal_stresses: ndarray    # shape (n_nodes, 3)
    nodal_von_mises: ndarray   # shape (n_nodes,)

def compute_von_mises_2d(sigma_xx: ndarray, sigma_yy: ndarray, tau_xy: ndarray) -> ndarray:
    """
    Compute 2D von Mises equivalent stress:
    sigma_vm = sqrt(sigma_xx^2 - sigma_xx * sigma_yy + sigma_yy^2 + 3 * tau_xy^2)
    """
    return np.sqrt(sigma_xx**2 - sigma_xx * sigma_yy + sigma_yy**2 + 3.0 * tau_xy**2)

def compute_element_stresses(
    mesh: Mesh,
    dof_map: DofMap,
    formulation: LinearElasticityFormulation,
    U: ndarray,
    field_name: str = "u"
) -> PostprocessResult:
    """
    Compute Gauss-point strains, stresses, von Mises equivalent stresses, and extrapolate to nodes.
    """
    n_elements = mesh.n_elements
    n_nodes = mesh.n_nodes
    cells = mesh.cells
    coords = mesh.coords
    
    # 2x2 Gauss quadrature
    quad_pts, quad_wts = get_quadrature_2d(2, 2)
    n_gps = len(quad_pts)
    
    basis = LagrangeQuad(p=1)
    
    D = formulation.material.get_constitutive_matrix(mode=formulation.mode)
    
    gauss_strains = zeros((n_elements, n_gps, 3))
    gauss_stresses = zeros((n_elements, n_gps, 3))
    gauss_von_mises = zeros((n_elements, n_gps))
    
    nodal_stress_sum = zeros((n_nodes, 3))
    nodal_count = zeros(n_nodes)
    
    for elem_idx, cell in enumerate(cells):
        elem_coords = coords[cell] # (4, 2)
        elem_dofs = dof_map.get_element_dofs(field_name, cell)
        u_elem = U[elem_dofs] # shape (8,)
        
        for q_idx, (gp, w) in enumerate(zip(quad_pts, quad_wts)):
            N, dN_dphys, detJ = basis.compute_mapping(gp, elem_coords)
            
            # Construct B matrix (3, 8)
            B = zeros((3, 8))
            for i in range(4):
                dN_dx = dN_dphys[0, i]
                dN_dy = dN_dphys[1, i]
                B[0, 2 * i]     = dN_dx
                B[1, 2 * i + 1] = dN_dy
                B[2, 2 * i]     = dN_dy
                B[2, 2 * i + 1] = dN_dx
                
            # Strain eps = B * u_elem
            eps = B @ u_elem
            # Stress sig = D * eps
            sig = D @ eps
            
            s_xx, s_yy, t_xy = sig[0], sig[1], sig[2]
            vm = float(compute_von_mises_2d(s_xx, s_yy, t_xy))
            
            gauss_strains[elem_idx, q_idx] = eps
            gauss_stresses[elem_idx, q_idx] = sig
            gauss_von_mises[elem_idx, q_idx] = vm
            
            # Accumulate Gauss point stresses to element nodes for nodal averaging
            for node_idx in cell:
                nodal_stress_sum[node_idx] += sig
                nodal_count[node_idx] += 1.0
                
    # Compute nodal averages
    nodal_stresses = zeros((n_nodes, 3))
    for i in range(n_nodes):
        if nodal_count[i] > 0:
            nodal_stresses[i] = nodal_stress_sum[i] / nodal_count[i]
            
    nodal_von_mises = compute_von_mises_2d(
        nodal_stresses[:, 0],
        nodal_stresses[:, 1],
        nodal_stresses[:, 2]
    )
    
    return PostprocessResult(
        gauss_strains=gauss_strains,
        gauss_stresses=gauss_stresses,
        gauss_von_mises=gauss_von_mises,
        nodal_stresses=nodal_stresses,
        nodal_von_mises=nodal_von_mises
    )
