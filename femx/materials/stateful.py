"""Minimal stateful material for testing Gauss-point commit/restore."""
from typing import Tuple
import numpy as np
from femx.materials.base import Material
from femx.materials.hyperelastic import NeoHookeanMaterial
from femx.backends.numpy_backend import ndarray, eye


class AccumulatingDeformationMaterial(Material):
    """
    Neo-Hookean stress response plus a scalar Gauss history
    alpha_trial = alpha_n + ||F - I||_F.
    """

    def __init__(self, rho: float, E: float, nu: float):
        super().__init__(rho=rho, E=E, nu=nu)
        self._elastic = NeoHookeanMaterial(rho=rho, E=E, nu=nu)

    def update(self, F: ndarray, alpha_n: float = 0.0) -> Tuple[ndarray, ndarray, float]:
        P, C4 = self._elastic.update(F)
        dim = F.shape[-1]
        alpha_trial = float(alpha_n) + float(np.linalg.norm(F - eye(dim)))
        return P, C4, alpha_trial
