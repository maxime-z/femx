from femx.materials.base import Material
from femx.backends.numpy_backend import ndarray, eye

class LinearHeatMaterial(Material):
    """
    Isotropic linear heat conduction material.
    Properties:
        rho: density
        C: capacity
        K: conductivity
    """
    def __init__(self, rho: float, C: float, K: float):
        super().__init__(rho=rho, C=C, K=K)

    def get_constitutive_matrix(self, dim: int) -> ndarray:
        """
        Return the thermal conductivity matrix D of shape (dim, dim).
        For isotropic material, D = K * I
        """
        K = self.get_property("K")
        return K * eye(dim)
