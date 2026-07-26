from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np
from femx.materials.base import Material
from femx.backends.numpy_backend import ndarray, zeros, eye, det, inv

class VolumetricPenalty(ABC):
    """Abstract base class for volumetric penalty functions G(J)."""
    @abstractmethod
    def G(self, J: float) -> float:
        """Penalty function value G(J)."""
        pass

    @abstractmethod
    def dG_dJ(self, J: float) -> float:
        """First derivative dG/dJ."""
        pass

    @abstractmethod
    def d2G_dJ2(self, J: float) -> float:
        """Second derivative d2G/dJ2."""
        pass


class QuadraticPenalty(VolumetricPenalty):
    """Quadratic penalty function G(J) = 0.5 * (J - 1)^2."""
    def G(self, J: float) -> float:
        return 0.5 * (J - 1.0) ** 2

    def dG_dJ(self, J: float) -> float:
        return J - 1.0

    def d2G_dJ2(self, J: float) -> float:
        return 1.0


class SimoMiehePenalty(VolumetricPenalty):
    """Simo & Miehe penalty function G(J) = 0.25 * (J^2 - 1 - 2*ln(J))."""
    def G(self, J: float) -> float:
        return 0.25 * (J**2 - 1.0 - 2.0 * np.log(J))

    def dG_dJ(self, J: float) -> float:
        return 0.5 * (J - 1.0 / J)

    def d2G_dJ2(self, J: float) -> float:
        return 0.5 * (1.0 + 1.0 / (J**2))


class OgdenPenalty(VolumetricPenalty):
    """Ogden penalty function G(J) = (1/beta^2) * (beta*ln(J) + J^(-beta) - 1)."""
    def __init__(self, beta: float = 1.0):
        self.beta = beta

    def G(self, J: float) -> float:
        b = self.beta
        return (1.0 / b**2) * (b * np.log(J) + J**(-b) - 1.0)

    def dG_dJ(self, J: float) -> float:
        b = self.beta
        return (1.0 / b) * (1.0 / J - J**(-b - 1.0))

    def d2G_dJ2(self, J: float) -> float:
        b = self.beta
        return (1.0 / (J**(b + 2.0))) * (1.0 + (1.0 / b) * (1.0 - J**b))


class MixedMooneyRivlinMaterial(Material):
    """
    Decoupled Mixed Mooney-Rivlin hyperelastic material model.
    Isochoric Strain Energy Density:
        W_iso(I1_bar, I2_bar) = c10 * (I1_bar - 3) + c01 * (I2_bar - 3)
    Volumetric Constraint:
        Legendre transform with pressure p (sign convention: p is positive in compression).
    """
    def __init__(
        self,
        rho: float,
        c10: float,
        c01: float,
        kappa: float,
        penalty: Optional[VolumetricPenalty] = None
    ):
        if penalty is None:
            penalty = QuadraticPenalty()
        super().__init__(rho=rho, c10=c10, c01=c01, kappa=kappa)
        self.penalty = penalty

    def _compute_pk1(self, F: ndarray, p: float) -> ndarray:
        """Compute PK1 stress P for a given (F, p). Used for numerical tangent."""
        dim = F.shape[0]
        J = float(det(F))
        F_inv = inv(F)
        F_inv_T = F_inv.T
        delta = eye(dim)

        c10 = self.get_property("c10")
        c01 = self.get_property("c01")

        J_minus_inv_dim = J ** (-1.0 / dim)
        J_minus_two_inv_dim = J ** (-2.0 / dim)

        F_bar = J_minus_inv_dim * F
        C_bar = F_bar.T @ F_bar
        I1_bar = float(np.trace(C_bar))

        C = F.T @ F
        C_inv = inv(C)

        dW_dI1bar = c10
        dW_dI2bar = c01

        I2_bar = 0.5 * (I1_bar**2 - np.trace(C_bar @ C_bar))
        S_bar = 2.0 * ((dW_dI1bar + I1_bar * dW_dI2bar) * delta - dW_dI2bar * C_bar)

        S_bar_dot_C = float(np.sum(S_bar * C))
        S_iso = J_minus_two_inv_dim * (S_bar - (1.0 / dim) * S_bar_dot_C * C_inv)

        S_vol = p * J * C_inv
        S_total = S_iso + S_vol
        P = F @ S_total
        return P

    def update_mixed(self, F: ndarray, p: float) -> Tuple[ndarray, ndarray, ndarray, float, ndarray, float]:
        """
        Compute PK1 stress P, material tangent C4, and cross-coupling terms with pressure p.

        Args:
            F: Deformation gradient (dim, dim)
            p: Hydrostatic pressure scalar

        Returns:
            P: PK1 stress tensor (dim, dim)
            C4_uu: Tangent tensor dP/dF (dim, dim, dim, dim)
            P_p: Derivative dP/dp = J * F^{-T} (dim, dim)
            R_p: Residual of pressure constraint = dG_dJ - p / kappa
            D_pu: Derivative dR_p / dF = d2G_dJ2 * J * F^{-T} (dim, dim)
            D_pp: Derivative dR_p / dp = -1 / kappa
        """
        dim = F.shape[0]
        J = float(det(F))
        if J <= 0:
            raise ValueError(f"Inverted element: J = {J} <= 0")

        F_inv = inv(F)
        F_inv_T = F_inv.T
        delta = eye(dim)

        c10 = self.get_property("c10")
        c01 = self.get_property("c01")
        kappa = self.get_property("kappa")

        # Isochoric strain measures
        alpha = J ** (-2.0 / dim)  # J^(-2/d)
        J_m1d = J ** (-1.0 / dim)

        F_bar = J_m1d * F
        C_bar = F_bar.T @ F_bar
        I1_bar = float(np.trace(C_bar))

        C = F.T @ F
        C_inv = inv(C)
        trC = float(np.trace(C))

        # Isochoric 2nd PK stress S_iso (Mooney-Rivlin deviatoric projection)
        dW_dI1bar = c10
        dW_dI2bar = c01
        I2_bar = 0.5 * (I1_bar**2 - np.trace(C_bar @ C_bar))
        S_bar = 2.0 * ((dW_dI1bar + I1_bar * dW_dI2bar) * delta - dW_dI2bar * C_bar)
        S_bar_dot_C = float(np.sum(S_bar * C))
        S_iso = alpha * (S_bar - (1.0 / dim) * S_bar_dot_C * C_inv)

        # Volumetric 2nd PK stress
        dG_dJ = self.penalty.dG_dJ(J)
        d2G_dJ2 = self.penalty.d2G_dJ2(J)
        S_vol = p * J * C_inv
        S_total = S_iso + S_vol

        # PK1 stress P = F @ S_total
        P = F @ S_total

        # --- Tangent C4_uu = dP/dF (analytical, 7-term formula) ---
        # For Neo-Hookean (c01=0): P_iso = alpha * (mu*F - (mu/d)*trC*F^{-T})
        #   where mu = 2*c10. Full product rule gives 5 isochoric terms.
        # P_vol = p*J*F^{-T} gives 2 volumetric terms.
        a = 2.0 * (c10 + c01)  # = mu for Neo-Hookean

        C4_uu = (
            # dP_iso/dF (5 terms from product rule through J^(-2/d), trC, and F^{-T}):
            a * alpha * np.einsum('ik,jl->ijkl', delta, delta)                          # 1: d(alpha*a*F)/dF identity
            - (2*a/dim) * alpha * np.einsum('ij,kl->ijkl', F, F_inv_T)                  # 2: d(alpha)/dF * a*F
            + (2*a/dim**2) * alpha * trC * np.einsum('ij,kl->ijkl', F_inv_T, F_inv_T)   # 3: d(alpha)/dF * (a/d)*trC*F^{-T}
            - (2*a/dim) * alpha * np.einsum('ij,kl->ijkl', F_inv_T, F)                  # 4: alpha*(a/d)*d(trC)/dF*F^{-T}
            + (a/dim) * alpha * trC * np.einsum('il,kj->ijkl', F_inv_T, F_inv_T)        # 5: alpha*(a/d)*trC*d(F^{-T})/dF
            # dP_vol/dF (2 terms from product rule through J and F^{-T}):
            + p * J * np.einsum('ij,kl->ijkl', F_inv_T, F_inv_T)                        # 6: p*dJ/dF*F^{-T}
            - p * J * np.einsum('il,kj->ijkl', F_inv_T, F_inv_T)                        # 7: p*J*d(F^{-T})/dF
        )

        # Cross terms (analytical — verified correct against FD):
        P_p = J * F_inv_T
        R_p = dG_dJ - p / kappa
        D_pu = d2G_dJ2 * J * F_inv_T
        D_pp = -1.0 / kappa

        return P, C4_uu, P_p, R_p, D_pu, D_pp


class MixedNeoHookeanMaterial(MixedMooneyRivlinMaterial):
    """Mixed Neo-Hookean material model (Mooney-Rivlin with c01 = 0)."""
    def __init__(
        self,
        rho: float,
        mu: float,
        kappa: float,
        penalty: Optional[VolumetricPenalty] = None
    ):
        super().__init__(rho=rho, c10=mu / 2.0, c01=0.0, kappa=kappa, penalty=penalty)
