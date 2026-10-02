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
            # Batched calculation over (..., dim, dim), typically (n_elems, n_gps, dim, dim)
            dim = F.shape[-1]
            J = np.linalg.det(F)  # shape (...)
            if np.any(J <= 0):
                raise ValueError("Inverted element detected: J <= 0")

            F_inv_T = np.swapaxes(np.linalg.inv(F), -1, -2)
            lnJ = np.log(J)
            lnJ_broad = lnJ[..., None, None]
            P = mu * F + (lambda_ * lnJ_broad - mu) * F_inv_T

            delta = eye(dim)
            term1 = mu * np.einsum('ik,jl->ijkl', delta, delta)
            term2 = lambda_ * np.einsum('...ij,...kl->...ijkl', F_inv_T, F_inv_T)
            coeff = (mu - lambda_ * lnJ)[..., None, None, None, None]
            term3 = coeff * np.einsum('...kj,...il->...ijkl', F_inv_T, F_inv_T)

            C4 = term1 + term2 + term3
            return P, C4


def neohookean_stress_tangent_torch(F, lambda_: float, mu: float):
    """Batched First Piola stress and analytic ``C4 = dP/dF`` (torch).

    ``F`` may be ``(dim, dim)`` or batched ``(..., dim, dim)``. Matches
    :meth:`NeoHookeanMaterial.update`.

    Returns:
        P: same shape as ``F``
        C4: shape ``(*F.shape[:-2], dim, dim, dim, dim)``
    """
    import torch

    J = torch.linalg.det(F.contiguous())
    if torch.any(J <= 0):
        raise ValueError("Inverted element detected: J <= 0")
    F_inv_T = torch.linalg.inv(F).transpose(-1, -2)
    lnJ = torch.log(J)
    lnJ_broad = lnJ
    while lnJ_broad.ndim < F.ndim:
        lnJ_broad = lnJ_broad.unsqueeze(-1)
    P = mu * F + (lambda_ * lnJ_broad - mu) * F_inv_T

    dim = F.shape[-1]
    delta = torch.eye(dim, dtype=F.dtype, device=F.device)
    term1 = mu * torch.einsum("ik,jl->ijkl", delta, delta)
    term2 = lambda_ * torch.einsum("...ij,...kl->...ijkl", F_inv_T, F_inv_T)
    # coeff shape (...); expand to (..., 1, 1, 1, 1) for C4 broadcast
    coeff = mu - lambda_ * lnJ
    for _ in range(4):
        coeff = coeff.unsqueeze(-1)
    term3 = coeff * torch.einsum("...kj,...il->...ijkl", F_inv_T, F_inv_T)
    C4 = term1 + term2 + term3
    return P, C4


def first_piola_torch(F, lambda_: float, mu: float):
    """First Piola-Kirchhoff stress for compressible Neo-Hookean (torch).

    ``F`` may be ``(dim, dim)`` or batched ``(..., dim, dim)``. Matches
    :meth:`NeoHookeanMaterial.update` without returning the analytic tangent.
    """
    P, _ = neohookean_stress_tangent_torch(F, lambda_, mu)
    return P

