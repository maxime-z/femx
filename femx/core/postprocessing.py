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


def compute_von_mises_3d(stress_voigt: ndarray) -> float:
    """Von Mises from Voigt stress [sxx, syy, szz, syz, sxz, sxy]."""
    sxx, syy, szz, syz, sxz, sxy = stress_voigt
    return float(np.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3.0 * (sxy ** 2 + syz ** 2 + sxz ** 2)
    ))


def element_center_von_mises_elasticity(mesh: Mesh, formulation, U: ndarray, field_name: str = "u") -> ndarray:
    """One von Mises value per hex/quad element from the element centroid Gauss point."""
    from femx.basis.lagrange import LagrangeHex, LagrangeQuad, hex_degree_from_nen
    from femx.core.quadrature import get_quadrature_3d, get_quadrature_2d

    cells = mesh.cells
    coords = mesh.coords
    dim = coords.shape[1]
    nen = cells.shape[1]
    n_comp = 3 if dim == 2 else 3
    vm = np.zeros(len(cells))

    if dim == 3:
        p = hex_degree_from_nen(nen)
        basis = LagrangeHex(p=p)
        mode = "3d"
        # Centroid of reference hex
        gp = np.array([0.0, 0.0, 0.0])
    else:
        basis = LagrangeQuad(p=1)
        mode = formulation.mode
        gp = np.array([0.0, 0.0])

    D = formulation.material.get_constitutive_matrix(mode=mode)
    for e, cell in enumerate(cells):
        elem_coords = coords[cell]
        elem_dofs = np.zeros(nen * dim, dtype=int)
        # Assume node-major packing matching DofMap
        for a, node in enumerate(cell):
            for c in range(dim):
                elem_dofs[a * dim + c] = node * dim + c
        # Prefer absolute DOFs from U layout: U is flat with node-major components
        u_elem = np.zeros(nen * dim)
        for a, node in enumerate(cell):
            u_elem[a * dim:(a + 1) * dim] = U.reshape((-1, dim))[node]

        N, dN, detJ = basis.compute_mapping(gp, elem_coords)
        if dim == 3:
            B = np.zeros((6, nen * dim))
            for i in range(nen):
                dN_dx, dN_dy, dN_dz = dN[0, i], dN[1, i], dN[2, i]
                B[0, 3 * i] = dN_dx
                B[1, 3 * i + 1] = dN_dy
                B[2, 3 * i + 2] = dN_dz
                B[3, 3 * i + 1] = dN_dz
                B[3, 3 * i + 2] = dN_dy
                B[4, 3 * i] = dN_dz
                B[4, 3 * i + 2] = dN_dx
                B[5, 3 * i] = dN_dy
                B[5, 3 * i + 1] = dN_dx
            sig = D @ (B @ u_elem)
            vm[e] = compute_von_mises_3d(sig)
        else:
            B = np.zeros((3, nen * dim))
            for i in range(nen):
                B[0, 2 * i] = dN[0, i]
                B[1, 2 * i + 1] = dN[1, i]
                B[2, 2 * i] = dN[1, i]
                B[2, 2 * i + 1] = dN[0, i]
            sig = D @ (B @ u_elem)
            vm[e] = float(compute_von_mises_2d(sig[0], sig[1], sig[2]))
    return vm


def element_center_von_mises_hyperelastic(mesh: Mesh, material, u_field: ndarray) -> ndarray:
    """Cauchy von Mises at the hex centroid for a compressible Neo-Hookean field."""
    from femx.basis.lagrange import LagrangeHex, hex_degree_from_nen

    cells = mesh.cells
    coords = mesh.coords
    nen = cells.shape[1]
    p = hex_degree_from_nen(nen)
    basis = LagrangeHex(p=p)
    gp = np.array([0.0, 0.0, 0.0])
    vm = np.zeros(len(cells))
    for e, cell in enumerate(cells):
        elem_coords = coords[cell]
        elem_u = u_field[cell]
        N, dN_dX, detJ = basis.compute_mapping(gp, elem_coords)
        H = elem_u.T @ dN_dX.T
        F = np.eye(3) + H
        P, _ = material.update(F)
        J = float(np.linalg.det(F))
        sigma = (1.0 / J) * (P @ F.T)
        sxx, syy, szz = sigma[0, 0], sigma[1, 1], sigma[2, 2]
        sxy, syz, sxz = sigma[0, 1], sigma[1, 2], sigma[0, 2]
        vm[e] = compute_von_mises_3d([sxx, syy, szz, syz, sxz, sxy])
    return vm

