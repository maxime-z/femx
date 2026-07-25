from typing import ClassVar, Tuple, Optional
import numpy as np
from femx.formulations.base import Formulation
from femx.backends.numpy_backend import ndarray, zeros, eye
from femx.materials.hyperelastic import NeoHookeanMaterial
from femx.basis.element import ElementBasis

class HyperelasticFormulation(Formulation[NeoHookeanMaterial]):
    """
    Formulation for 2D/3D hyperelasticity under finite deformation.
    Solves for displacement u using First Piola-Kirchhoff stress P and tangent C4.
    """
    field_names: ClassVar[Tuple[str, ...]] = ("u",)

    def __init__(self, material: NeoHookeanMaterial):
        super().__init__(material)

    def compute_element_residual_and_tangent(
        self,
        elem_coords: ndarray,
        elem_u: ndarray,
        quadrature_pts: ndarray,
        quadrature_wts: ndarray,
        body_load: Optional[ndarray] = None,
        elem_basis: Optional[ElementBasis] = None
    ) -> Tuple[ndarray, ndarray]:
        """
        Compute element internal force (residual) Re and tangent stiffness Ke.
        
        Args:
            elem_coords: shape (n_local, dim) - reference coordinates X
            elem_u: shape (n_local, dim) - nodal displacements u_e
            quadrature_pts: shape (n_gps, dim)
            quadrature_wts: shape (n_gps,)
            body_load: shape (dim,)
            elem_basis: element basis functions
            
        Returns:
            Re: shape (dim * n_local,) - element residual R_e = F_int - F_ext
            Ke: shape (dim * n_local, dim * n_local) - tangent stiffness matrix
        """
        n_local, dim = elem_coords.shape
        n_dofs_local = dim * n_local
        
        Re = zeros(n_dofs_local)
        Ke = zeros((n_dofs_local, n_dofs_local))
        
        if body_load is None:
            body_load = zeros(dim)
            
        if elem_basis is None:
            from femx.basis.lagrange import LagrangeQuad
            elem_basis = LagrangeQuad(p=1)
            
        for gp, w in zip(quadrature_pts, quadrature_wts):
            N, dN_dX, detJ = elem_basis.compute_mapping(gp, elem_coords)
            dV = detJ * w
            
            # Displacement gradient H_ij = d(u_i)/d(X_j) = sum_a u_{a, i} * dN_a/dX_j
            # elem_u is shape (n_local, dim), dN_dX is shape (dim, n_local)
            # H is shape (dim, dim)
            H = elem_u.T @ dN_dX.T
            
            # Deformation gradient F = I + H
            F = eye(dim) + H
            
            # Material update: First Piola-Kirchhoff stress P (dim, dim) and tangent C4 (dim, dim, dim, dim)
            P, C4 = self.material.update(F)
            
            # Internal force vector R_int[a, i] = sum_j P[i, j] * dN_dX[j, a] * dV
            for a in range(n_local):
                for i in range(dim):
                    dof_ai = a * dim + i
                    # Internal force contribution
                    f_int_ai = np.sum(P[i, :] * dN_dX[:, a]) * dV
                    # External force contribution
                    f_ext_ai = body_load[i] * N[a] * dV
                    
                    Re[dof_ai] += f_int_ai - f_ext_ai
                    
                    # Tangent stiffness K[a, i, b, k] = sum_{j, l} C4[i, j, k, l] * dN_dX[j, a] * dN_dX[l, b] * dV
                    for b in range(n_local):
                        for k in range(dim):
                            dof_bk = b * dim + k
                            k_val = np.einsum('jl,j,l->', C4[i, :, k, :], dN_dX[:, a], dN_dX[:, b]) * dV
                            Ke[dof_ai, dof_bk] += k_val
                            
        return Re, Ke

    def compute_element_matrices(
        self,
        elem_coords: ndarray,
        quadrature_pts: ndarray,
        quadrature_wts: ndarray,
        body_load: Optional[ndarray] = None,
        elem_basis: Optional[ElementBasis] = None,
        **kwargs
    ) -> Tuple[ndarray, ndarray, ndarray]:
        """Compute initial linear element matrices Ke, Me, fe around zero displacement state u=0."""
        n_local, dim = elem_coords.shape
        n_dofs_local = dim * n_local
        elem_u = zeros((n_local, dim))
        Re, Ke = self.compute_element_residual_and_tangent(
            elem_coords, elem_u, quadrature_pts, quadrature_wts, body_load=body_load, elem_basis=elem_basis
        )
        Me = zeros((n_dofs_local, n_dofs_local))
        fe = -Re
        return Ke, Me, fe

    def get_physical_tensors(self, geom, device: str = "cpu", dtype = None):
        raise NotImplementedError("Hyperelasticity relies on state-dependent nonlinear assembly.")

    def compute_batch_map(self, geom, tensors, device: str = "cpu", dtype = None):
        raise NotImplementedError("Hyperelasticity relies on state-dependent nonlinear assembly.")

