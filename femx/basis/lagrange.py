import numpy as np
from typing import Tuple
from femx.backends.numpy_backend import ndarray, array, invert_matrix, determinant
from femx.basis.element import ElementBasis
from femx.core.quadrature import get_quadrature_2d, get_quadrature_triangle

class LagrangeQuad(ElementBasis):
    """
    Arbitrary-order 2D Quadrilateral Lagrange Element (Q1, Q2, Q3...).
    Reference domain [-1, 1]^2.
    """
    def __init__(self, p: int = 1):
        if p < 1:
            raise ValueError(f"Polynomial degree p must be >= 1, got {p}")
        self.p = p
        self._nodes_1d = np.linspace(-1.0, 1.0, p + 1)
        
    @property
    def n_dofs_per_element(self) -> int:
        return (self.p + 1) ** 2

    @property
    def dim(self) -> int:
        return 2

    def evaluate_shape_functions(self, ref_coords: ndarray) -> ndarray:
        xi, eta = ref_coords[0], ref_coords[1]
        if self.p == 1:
            # Standard counter-clockwise Q1 ordering:
            # 0: (-1, -1), 1: (+1, -1), 2: (+1, +1), 3: (-1, +1)
            return 0.25 * array([
                (1.0 - xi) * (1.0 - eta),
                (1.0 + xi) * (1.0 - eta),
                (1.0 + xi) * (1.0 + eta),
                (1.0 - xi) * (1.0 + eta)
            ])
        else:
            L_xi = self._eval_1d_lagrange(xi)
            L_eta = self._eval_1d_lagrange(eta)
            return np.outer(L_eta, L_xi).ravel()

    def evaluate_shape_derivatives(self, ref_coords: ndarray) -> ndarray:
        xi, eta = ref_coords[0], ref_coords[1]
        if self.p == 1:
            # Standard counter-clockwise Q1 derivatives:
            # Row 0: dN/dxi, Row 1: dN/deta
            return 0.25 * array([
                [-(1.0 - eta),  (1.0 - eta), (1.0 + eta), -(1.0 + eta)],
                [-(1.0 - xi),  -(1.0 + xi),  (1.0 + xi),   (1.0 - xi)]
            ])
        else:
            L_xi = self._eval_1d_lagrange(xi)
            dL_xi = self._eval_1d_lagrange_derivatives(xi)
            L_eta = self._eval_1d_lagrange(eta)
            dL_eta = self._eval_1d_lagrange_derivatives(eta)
            
            dN_dxi = np.outer(L_eta, dL_xi).ravel()
            dN_deta = np.outer(dL_eta, L_xi).ravel()
            return np.vstack([dN_dxi, dN_deta])

    def compute_mapping(self, ref_coords: ndarray, elem_coords: ndarray) -> Tuple[ndarray, ndarray, float]:
        N = self.evaluate_shape_functions(ref_coords)
        dN_dref = self.evaluate_shape_derivatives(ref_coords)
        
        J = dN_dref @ elem_coords
        detJ = float(determinant(J))
        if detJ <= 0.0:
            raise ValueError(f"Jacobian determinant is non-positive: {detJ}")
            
        invJ = invert_matrix(J)
        dN_dphys = invJ @ dN_dref
        return N, dN_dphys, detJ

    def get_default_quadrature(self) -> Tuple[ndarray, ndarray]:
        n_pts = self.p + 1
        return get_quadrature_2d(n_pts, n_pts)

    def _eval_1d_lagrange(self, x: float) -> ndarray:
        nodes = self._nodes_1d
        n = len(nodes)
        vals = np.ones(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    vals[i] *= (x - nodes[j]) / (nodes[i] - nodes[j])
        return vals

    def _eval_1d_lagrange_derivatives(self, x: float) -> ndarray:
        nodes = self._nodes_1d
        n = len(nodes)
        dvals = np.zeros(n)
        for i in range(n):
            for k in range(n):
                if k != i:
                    term = 1.0 / (nodes[i] - nodes[k])
                    for j in range(n):
                        if j != i and j != k:
                            term *= (x - nodes[j]) / (nodes[i] - nodes[j])
                    dvals[i] += term
        return dvals


class LagrangeTriangle(ElementBasis):
    """
    2D Triangular Lagrange Element (T1, T2).
    Reference domain: xi >= 0, eta >= 0, xi + eta <= 1.
    """
    def __init__(self, p: int = 1):
        if p not in (1, 2):
            raise ValueError(f"LagrangeTriangle currently supports p=1 or p=2, got {p}")
        self.p = p

    @property
    def n_dofs_per_element(self) -> int:
        return 3 if self.p == 1 else 6

    @property
    def dim(self) -> int:
        return 2

    def evaluate_shape_functions(self, ref_coords: ndarray) -> ndarray:
        xi, eta = ref_coords[0], ref_coords[1]
        l3 = 1.0 - xi - eta
        l1 = xi
        l2 = eta
        if self.p == 1:
            return array([l3, l1, l2])
        else:
            return array([
                l3 * (2.0 * l3 - 1.0),
                l1 * (2.0 * l1 - 1.0),
                l2 * (2.0 * l2 - 1.0),
                4.0 * l1 * l3,
                4.0 * l1 * l2,
                4.0 * l2 * l3
            ])

    def evaluate_shape_derivatives(self, ref_coords: ndarray) -> ndarray:
        xi, eta = ref_coords[0], ref_coords[1]
        if self.p == 1:
            return array([
                [-1.0,  1.0,  0.0],
                [-1.0,  0.0,  1.0]
            ])
        else:
            l3 = 1.0 - xi - eta
            l1 = xi
            l2 = eta
            dN_dxi = array([
                -4.0 * l3 + 1.0,
                4.0 * l1 - 1.0,
                0.0,
                4.0 * (l3 - l1),
                4.0 * l2,
                -4.0 * l2
            ])
            dN_deta = array([
                -4.0 * l3 + 1.0,
                0.0,
                4.0 * l2 - 1.0,
                -4.0 * l1,
                4.0 * l1,
                4.0 * (l3 - l2)
            ])
            return np.vstack([dN_dxi, dN_deta])

    def compute_mapping(self, ref_coords: ndarray, elem_coords: ndarray) -> Tuple[ndarray, ndarray, float]:
        N = self.evaluate_shape_functions(ref_coords)
        dN_dref = self.evaluate_shape_derivatives(ref_coords)
        
        J = dN_dref @ elem_coords
        detJ = float(determinant(J))
        if detJ <= 0.0:
            raise ValueError(f"Jacobian determinant non-positive: {detJ}")
            
        invJ = invert_matrix(J)
        dN_dphys = invJ @ dN_dref
        return N, dN_dphys, detJ

    def get_default_quadrature(self) -> Tuple[ndarray, ndarray]:
        n_pts = 1 if self.p == 1 else 3
        return get_quadrature_triangle(n_pts)


