from typing import Tuple
from numpy import eye, zeros, einsum
from femx.materials.base import Material
from femx.backends.numpy_backend import ndarray

class LinearThermoelasticMaterial(Material):
    """
    Isotropic linear thermoelastic material model.
    Properties:
        rho: mass density
        E: Young's modulus
        nu: Poisson's ratio
        K_th: thermal conductivity
        alpha: secant thermal expansion coefficient
        C_cap: specific heat capacity (default 1.0)
        T0: reference stress-free temperature (default 0.0)
    """
    def __init__(self, rho: float, E: float, nu: float, K_th: float, alpha: float, C_cap: float = 1.0, T0: float = 0.0):
        super().__init__(rho=rho, E=E, nu=nu, K_th=K_th, alpha=alpha, C_cap=C_cap, T0=T0)

    def get_lame_parameters(self) -> Tuple[float, float]:
        """Compute Lamé parameters (lambda_, mu) from E and nu."""
        E = self.get_property("E")
        nu = self.get_property("nu")
        lambda_ = (nu * E) / ((1.0 + nu) * (1.0 - 2.0 * nu))
        mu = E / (2.0 * (1.0 + nu))
        return lambda_, mu

    def get_constitutive_matrix(self, mode: str = "plane_strain") -> ndarray:
        """Return 3x3 mechanical constitutive matrix D."""
        E = self.get_property("E")
        nu = self.get_property("nu")
        lambda_, mu = self.get_lame_parameters()

        if mode == "plane_strain":
            D = zeros((3, 3))
            D[0, 0] = lambda_ + 2.0 * mu
            D[0, 1] = lambda_
            D[1, 0] = lambda_
            D[1, 1] = lambda_ + 2.0 * mu
            D[2, 2] = mu
            return D
        elif mode == "plane_stress":
            factor = E / (1.0 - nu**2)
            D = zeros((3, 3))
            D[0, 0] = factor
            D[0, 1] = factor * nu
            D[1, 0] = factor * nu
            D[1, 1] = factor
            D[2, 2] = factor * (1.0 - nu) / 2.0
            return D
        else:
            raise ValueError(f"Unknown elastic mode: {mode}")

    def get_elasticity_tensor_4th(self, mode: str = "plane_strain", dim: int = 2) -> ndarray:
        """Return 4th-order elasticity tensor C4 of shape (dim, dim, dim, dim)."""
        E = self.get_property("E")
        nu = self.get_property("nu")
        lambda_, mu = self.get_lame_parameters()

        if mode == "plane_stress":
            lambda_star = (E * nu) / (1.0 - nu**2)
        else:
            lambda_star = lambda_

        delta = eye(dim)
        C4 = einsum('ij,kl->ijkl', delta, delta) * lambda_star + (
            einsum('ik,jl->ijkl', delta, delta) + einsum('il,jk->ijkl', delta, delta)
        ) * mu
        return C4

    def get_thermal_coupling_vector(self, mode: str = "plane_strain") -> ndarray:
        """
        Return 3x1 Voigt thermal stress coefficient vector m_th = C : I_th.
        In Voigt notation: sigma_th = m_th * (T - T0).
        """
        E = self.get_property("E")
        nu = self.get_property("nu")
        alpha = self.get_property("alpha")
        lambda_, mu = self.get_lame_parameters()

        m_th = zeros(3)
        if mode == "plane_strain":
            beta = (3.0 * lambda_ + 2.0 * mu) * alpha
        elif mode == "plane_stress":
            beta = (E * alpha) / (1.0 - nu)
        else:
            raise ValueError(f"Unknown elastic mode: {mode}")

        m_th[0] = beta
        m_th[1] = beta
        m_th[2] = 0.0
        return m_th

    def get_thermal_coupling_tensor(self, mode: str = "plane_strain", dim: int = 2) -> ndarray:
        """
        Return 2nd-order thermal coupling tensor M_th of shape (dim, dim).
        M_th_ij = beta * delta_ij.
        """
        E = self.get_property("E")
        nu = self.get_property("nu")
        alpha = self.get_property("alpha")
        lambda_, mu = self.get_lame_parameters()

        if mode == "plane_strain":
            beta = (3.0 * lambda_ + 2.0 * mu) * alpha
        elif mode == "plane_stress":
            beta = (E * alpha) / (1.0 - nu)
        else:
            raise ValueError(f"Unknown elastic mode: {mode}")

        return beta * eye(dim)
