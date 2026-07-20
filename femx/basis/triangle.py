import numpy as np
from typing import Tuple
from femx.backends.numpy_backend import ndarray, array, invert_matrix, determinant, zeros

def evaluate_t1_shape_functions(xi: float, eta: float) -> ndarray:
    """
    Evaluate 2D Linear Triangle (T1) shape functions at reference point (xi, eta).
    Area coordinates / reference triangle domain: xi >= 0, eta >= 0, xi + eta <= 1.
    """
    N0 = 1.0 - xi - eta
    N1 = xi
    N2 = eta
    return array([N0, N1, N2])

def evaluate_t1_shape_derivatives_ref() -> ndarray:
    """
    Evaluate T1 reference derivatives dN/dref.
    Returns:
        dN_dref: array of shape (2, 3) where:
            row 0 is dN/dxi = [-1, 1, 0]
            row 1 is dN/deta = [-1, 0, 1]
    """
    return array([
        [-1.0,  1.0,  0.0],
        [-1.0,  0.0,  1.0]
    ])

def compute_t1_mapping(gp: ndarray, elem_coords: ndarray) -> Tuple[ndarray, ndarray, float]:
    """
    Compute T1 physical mapping at Gauss point gp = [xi, eta].
    Args:
        gp: Reference point [xi, eta]
        elem_coords: Element node coordinates of shape (3, 2)
    Returns:
        N: Shape function values (shape (3,))
        dN_dphys: Derivatives wrt physical coordinates [x, y] (shape (2, 3))
        detJ: Jacobian determinant (scalar)
    """
    xi, eta = gp[0], gp[1]
    N = evaluate_t1_shape_functions(xi, eta)
    dN_dref = evaluate_t1_shape_derivatives_ref()
    
    # Jacobian matrix J = dN_dref @ elem_coords (shape (2, 2))
    J = dN_dref @ elem_coords
    detJ = float(determinant(J))
    if detJ <= 0.0:
        raise ValueError(f"Non-positive Jacobian determinant det(J) = {detJ} in T1 element.")
        
    invJ = invert_matrix(J)
    # dN_dphys = inv(J) @ dN_dref (shape (2, 3))
    dN_dphys = invJ @ dN_dref
    
    return N, dN_dphys, detJ
