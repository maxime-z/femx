from typing import Tuple
import numpy as np
from femx.materials.base import Material
from femx.backends.numpy_backend import ndarray, zeros, eye, det, inv

class NeoHookeanMaterial(Material):
    """
    Compressible Neo-Hookean hyperelastic material model.
    Strain energy density:
        W(F) = 0.5 * mu * (tr(C) - d - 2*ln(J)) + 0.5 * lambda * (ln(J))^2
    where C = F^T * F, J = det(F), d = dimension (2 or 3).
    
    Properties:
        E: Young's modulus
        nu: Poisson's ratio
        rho: density
    """
    def __init__(self, rho: float, E: float, nu: float):
        super().__init__(rho=rho, E=E, nu=nu)

    def get_lame_parameters(self) -> Tuple[float, float]:
        """Compute Lame parameters (lambda_, mu) from Young's modulus and Poisson's ratio."""
        E = self.get_property("E")
        nu = self.get_property("nu")
        lambda_ = (nu * E) / ((1.0 + nu) * (1.0 - 2.0 * nu))
        mu = E / (2.0 * (1.0 + nu))
        return lambda_, mu

    def update(self, F: ndarray) -> Tuple[ndarray, ndarray]:
        """
        Compute First Piola-Kirchhoff stress P and material tangent C4 = dP/dF for a given F.
        
        Args:
            F: Deformation gradient tensor of shape (dim, dim) or (n_elems, n_gps, dim, dim).
            
        Returns:
            P: First Piola-Kirchhoff stress tensor of same shape as F.
            C4: Tangent stiffness tensor dP_ij / dF_kl of shape (*F.shape, dim, dim).
        """
        lambda_, mu = self.get_lame_parameters()
        
        if F.ndim == 2:
            dim = F.shape[0]
            J = float(det(F))
            if J <= 0:
                raise ValueError(f"Inverted element: J = {J} <= 0")
            
            F_inv_T = inv(F).T
            lnJ = np.log(J)
            
            # P_ij = mu * F_ij + (lambda * ln(J) - mu) * F^{-T}_ij
            P = mu * F + (lambda_ * lnJ - mu) * F_inv_T
            
            # Tangent tensor C4[i, j, k, l] = dP_ij / dF_kl
            # C4 = mu * delta_ik * delta_jl + lambda * F_inv_T_ij * F_inv_T_kl + (mu - lambda * lnJ) * F_inv_T_kj * F_inv_T_il
            delta = eye(dim)
            C4 = mu * np.einsum('ik,jl->ijkl', delta, delta) + \
                 lambda_ * np.einsum('ij,kl->ijkl', F_inv_T, F_inv_T) + \
                 (mu - lambda_ * lnJ) * np.einsum('kj,il->ijkl', F_inv_T, F_inv_T)
                 
            return P, C4
        else:
            # Batched calculation over (n_elems, n_gps, dim, dim)
            dim = F.shape[-1]
            J = np.linalg.det(F) # shape (n_elems, n_gps)
            if np.any(J <= 0):
                raise ValueError("Inverted element detected: J <= 0")
            
            F_inv_T = np.swapaxes(np.linalg.inv(F), -1, -2) # shape (n_elems, n_gps, dim, dim)
            lnJ = np.log(J)[..., None, None] # shape (n_elems, n_gps, 1, 1)
            
            P = mu * F + (lambda_ * lnJ - mu) * F_inv_T
            
            delta = eye(dim)
            term1 = mu * np.einsum('ik,jl->ijkl', delta, delta)
            term2 = lambda_ * np.einsum('...ij,...kl->...ijkl', F_inv_T, F_inv_T)
            term3 = (mu - lambda_ * lnJ[..., 0]) * np.einsum('...kj,...il->...ijkl', F_inv_T, F_inv_T)
            
            C4 = term1 + term2 + term3
            return P, C4
