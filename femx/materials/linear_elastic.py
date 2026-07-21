from typing import Tuple
from femx.materials.base import Material
from femx.backends.numpy_backend import ndarray, zeros

class LinearElasticMaterial(Material):
    """
    Isotropic linear elastic material model.
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

    def get_constitutive_matrix(self, mode: str = "plane_strain") -> ndarray:
        """
        Return constitutive matrix D:
        - "plane_strain": 3x3 matrix based on Lame parameters
        - "plane_stress": 3x3 matrix based on E and nu
        - "3d": 6x6 matrix for 3D isotropic linear elasticity
        """
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
        elif mode == "3d":
            D = zeros((6, 6))
            # Normal components (xx, yy, zz)
            for i in range(3):
                for j in range(3):
                    D[i, j] = lambda_
                D[i, i] += 2.0 * mu
            # Shear components (yz, zx, xy)
            D[3, 3] = mu
            D[4, 4] = mu
            D[5, 5] = mu
            return D
        else:
            raise ValueError(f"Unknown elastic mode: {mode}. Must be 'plane_strain', 'plane_stress', or '3d'.")

    def get_elasticity_tensor_4th(self, mode: str = "plane_strain", dim: int = 2) -> ndarray:
        """
        Return the true 4th-order elasticity tensor C4 of shape (dim, dim, dim, dim).
        C4[i, j, k, l] = lambda_star * delta_ij * delta_kl + mu * (delta_ik * delta_jl + delta_il * delta_jk)
        """
        E = self.get_property("E")
        nu = self.get_property("nu")
        lambda_, mu = self.get_lame_parameters()

        if mode == "plane_stress":
            lambda_star = (E * nu) / (1.0 - nu**2)
        else:
            lambda_star = lambda_

        from numpy import eye, einsum
        delta = eye(dim)
        
        C4 = einsum('ij,kl->ijkl', delta, delta) * lambda_star + (
            einsum('ik,jl->ijkl', delta, delta) + einsum('il,jk->ijkl', delta, delta)
        ) * mu
        
        return C4
