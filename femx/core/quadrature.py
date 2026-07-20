import numpy as np
from typing import Tuple
from femx.backends.numpy_backend import ndarray, array

def get_quadrature_1d(n_points: int) -> Tuple[ndarray, ndarray]:
    """
    Get Gauss-Legendre quadrature points and weights in 1D for interval [-1, 1].
    """
    points, weights = np.polynomial.legendre.leggauss(n_points)
    return points, weights

def get_quadrature_2d(n_points_u: int, n_points_v: int) -> Tuple[ndarray, ndarray]:
    """
    Get tensor-product Gauss-Legendre quadrature points and weights in 2D for [-1, 1]^2.
    Returns:
        points_2d: array of shape (n_points_u * n_points_v, 2)
        weights_2d: array of shape (n_points_u * n_points_v,)
    """
    pts_u, w_u = get_quadrature_1d(n_points_u)
    pts_v, w_v = get_quadrature_1d(n_points_v)
    
    pts_2d = []
    w_2d = []
    # Row-major ordering matching standard tensor products (outer v, inner u)
    for pv, wv in zip(pts_v, w_v):
        for pu, wu in zip(pts_u, w_u):
            pts_2d.append([pu, pv])
            w_2d.append(wu * wv)
            
    return array(pts_2d), array(w_2d)

def get_quadrature_triangle(n_points: int = 1) -> Tuple[ndarray, ndarray]:
    """
    Get quadrature points and weights for reference triangle (xi >= 0, eta >= 0, xi + eta <= 1).
    Args:
        n_points: 1 or 3 integration points
    Returns:
        points: array of shape (n_points, 2)
        weights: array of shape (n_points,)
    """
    if n_points == 1:
        # Centroid rule (exact for linear polynomials)
        pts = array([[1.0 / 3.0, 1.0 / 3.0]])
        wts = array([0.5])
        return pts, wts
    elif n_points == 3:
        # 3-point mid-edge rule (exact for quadratics)
        pts = array([
            [1.0 / 6.0, 1.0 / 6.0],
            [2.0 / 3.0, 1.0 / 6.0],
            [1.0 / 6.0, 2.0 / 3.0]
        ])
        wts = array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        return pts, wts
    else:
        raise ValueError("Triangle quadrature currently supports n_points=1 or n_points=3.")
