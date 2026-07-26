from typing import ClassVar, Tuple, Optional
import numpy as np
from femx.formulations.base import Formulation
from femx.backends.numpy_backend import ndarray, zeros, eye, det, inv
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


class MixedHyperelasticFormulation(Formulation):
    """
    Two-field (u, p) mixed formulation for finite strain hyperelasticity.
    Displacement u and pressure p can be defined on equal or distinct basis spaces.
    """
    field_names: ClassVar[Tuple[str, ...]] = ("u", "p")

    def __init__(self, material):
        super().__init__(material)

    def compute_element_residual_and_tangent(
        self,
        elem_coords: ndarray,
        elem_u: ndarray,
        quadrature_pts: ndarray,
        quadrature_wts: ndarray,
        body_load: Optional[ndarray] = None,
        elem_p: Optional[ndarray] = None,
        elem_basis_u: Optional[ElementBasis] = None,
        elem_basis_p: Optional[ElementBasis] = None
    ) -> Tuple[ndarray, ndarray]:
        """
        Compute element residual vector Re = [Re_u, Re_p]^T and coupled tangent Ke.

        Args:
            elem_coords: (n_u, dim)
            elem_u: (n_u, dim)
            quadrature_pts: (n_gps, dim)
            quadrature_wts: (n_gps,)
            body_load: (dim,)
            elem_p: (n_p,)
        """
        n_u, dim = elem_coords.shape
        if elem_p is None:
            elem_p = zeros(n_u)
        n_p = len(elem_p)
        n_dofs_u = dim * n_u
        n_dofs_total = n_dofs_u + n_p

        Re = zeros(n_dofs_total)
        Ke = zeros((n_dofs_total, n_dofs_total))

        if body_load is None:
            body_load = zeros(dim)

        if elem_basis_u is None:
            from femx.basis.lagrange import LagrangeQuad
            elem_basis_u = LagrangeQuad(p=1)
        if elem_basis_p is None:
            elem_basis_p = elem_basis_u

        for gp, w in zip(quadrature_pts, quadrature_wts):
            N_u, dN_u_dX, detJ = elem_basis_u.compute_mapping(gp, elem_coords)
            N_p = elem_basis_p.evaluate_shape_functions(gp)
            dV = detJ * w

            # Displacement gradient H = u_e^T @ dN_u_dX^T
            H = elem_u.T @ dN_u_dX.T
            F = eye(dim) + H

            # Pressure value at Gauss point
            p_val = float(np.squeeze(N_p @ elem_p))

            # Material update
            P, C4_uu, P_p, R_p_val, D_pu, D_pp = self.material.update_mixed(F, p_val)

            # Assemble u residual & p residual
            for a in range(n_u):
                for i in range(dim):
                    dof_ai = a * dim + i
                    f_int = np.sum(P[i, :] * dN_u_dX[:, a]) * dV
                    f_ext = body_load[i] * N_u[a] * dV
                    Re[dof_ai] += f_int - f_ext

            for a in range(n_p):
                dof_pa = n_dofs_u + a
                Re[dof_pa] += R_p_val * N_p[a] * dV

            # Assemble Tangent Blocks (K_uu, K_up, K_pu, K_pp)
            # K_uu
            for a in range(n_u):
                for i in range(dim):
                    dof_ai = a * dim + i
                    for b in range(n_u):
                        for k in range(dim):
                            dof_bk = b * dim + k
                            k_val = np.einsum('jl,j,l->', C4_uu[i, :, k, :], dN_u_dX[:, a], dN_u_dX[:, b]) * dV
                            Ke[dof_ai, dof_bk] += k_val

            # K_up
            for a in range(n_u):
                for i in range(dim):
                    dof_ai = a * dim + i
                    for b in range(n_p):
                        dof_pb = n_dofs_u + b
                        k_up_val = np.sum(P_p[i, :] * dN_u_dX[:, a]) * N_p[b] * dV
                        Ke[dof_ai, dof_pb] += k_up_val

            # K_pu
            for a in range(n_p):
                dof_pa = n_dofs_u + a
                for b in range(n_u):
                    for k in range(dim):
                        dof_bk = b * dim + k
                        k_pu_val = N_p[a] * np.sum(D_pu[k, :] * dN_u_dX[:, b]) * dV
                        Ke[dof_pa, dof_bk] += k_pu_val

            # K_pp
            for a in range(n_p):
                dof_pa = n_dofs_u + a
                for b in range(n_p):
                    dof_pb = n_dofs_u + b
                    k_pp_val = D_pp * N_p[a] * N_p[b] * dV
                    Ke[dof_pa, dof_pb] += k_pp_val

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
        """Compute initial element matrices Ke, Me, fe at zero state."""
        n_u, dim = elem_coords.shape
        n_p = kwargs.get("n_p", n_u)
        n_dofs_total = dim * n_u + n_p
        elem_u = zeros((n_u, dim))
        elem_p = zeros(n_p)
        Re, Ke = self.compute_element_residual_and_tangent(
            elem_coords, elem_u, quadrature_pts, quadrature_wts, body_load=body_load, elem_p=elem_p, **kwargs
        )
        Me = zeros((n_dofs_total, n_dofs_total))
        fe = -Re
        return Ke, Me, fe

    def get_physical_tensors(self, geom, device: str = "cpu", dtype = None):
        raise NotImplementedError("MixedHyperelasticFormulation relies on state-dependent nonlinear assembly.")

    def compute_batch_map(self, geom, tensors, device: str = "cpu", dtype = None):
        raise NotImplementedError("MixedHyperelasticFormulation relies on state-dependent nonlinear assembly.")



class FBarHyperelasticFormulation(HyperelasticFormulation):
    """
    F-bar (F-Bar) strain projection formulation for hyperelasticity.
    Projects volumetric deformation gradient J to a centroidal projected volume change J_tilde.
    """
    def compute_element_residual_and_tangent(
        self,
        elem_coords: ndarray,
        elem_u: ndarray,
        quadrature_pts: ndarray,
        quadrature_wts: ndarray,
        body_load: Optional[ndarray] = None,
        elem_basis: Optional[ElementBasis] = None
    ) -> Tuple[ndarray, ndarray]:
        n_local, dim = elem_coords.shape
        n_dofs_local = dim * n_local

        if elem_basis is None:
            from femx.basis.lagrange import LagrangeQuad
            elem_basis = LagrangeQuad(p=1)

        def compute_residual(u_mat):
            # Centroidal / Mean deformation gradient F_tilde
            gp_centroid = np.mean(quadrature_pts, axis=0)
            _, dN_dX_c, _ = elem_basis.compute_mapping(gp_centroid, elem_coords)
            H_c = u_mat.T @ dN_dX_c.T
            F_c = eye(dim) + H_c
            J_tilde = float(det(F_c))
            if J_tilde <= 0:
                J_tilde = 1.0e-5
            F_c_inv_T = inv(F_c).T

            Re = zeros(n_dofs_local)
            if body_load is None:
                b_load = zeros(dim)
            else:
                b_load = body_load

            for gp, w in zip(quadrature_pts, quadrature_wts):
                N, dN_dX, detJ = elem_basis.compute_mapping(gp, elem_coords)
                dV = detJ * w

                H = u_mat.T @ dN_dX.T
                F = eye(dim) + H
                J = float(det(F))
                if J <= 0:
                    J = 1.0e-5

                F_inv_T = inv(F).T
                alpha = (J_tilde / J) ** (1.0 / dim)
                F_bar = alpha * F

                # Update material at F_bar (only P is needed for residual)
                P, _ = self.material.update(F_bar)
                beta = (1.0 / dim) * float(np.sum(P * F))

                for a in range(n_local):
                    for i in range(dim):
                        dof_ai = a * dim + i
                        f_int_ai = (np.sum((alpha * P[i, :] - beta * F_inv_T[i, :]) * dN_dX[:, a]) +
                                    np.sum(beta * F_c_inv_T[i, :] * dN_dX_c[:, a])) * dV
                        f_ext_ai = b_load[i] * N[a] * dV
                        Re[dof_ai] += f_int_ai - f_ext_ai
            return Re

        # Base residual
        Re_base = compute_residual(elem_u)
        Ke = zeros((n_dofs_local, n_dofs_local))
        
        # Numerical element tangent (Finite Difference)
        eps = 1e-8
        for j in range(n_dofs_local):
            u_pert = elem_u.copy()
            u_pert.flat[j] += eps
            Re_pert = compute_residual(u_pert)
            Ke[:, j] = (Re_pert - Re_base) / eps

        return Re_base, Ke
