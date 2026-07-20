import numpy as np
from typing import Tuple
from femx.backends.numpy_backend import ndarray, array, invert_matrix, determinant

def evaluate_q1_shape_functions(xi: float, eta: float) -> ndarray:
    """
    Evaluate bilinear Q1 shape functions at reference point (xi, eta).
    Returns a 1D array of shape (4,).
    """
    N = 0.25 * array([
        (1.0 - xi) * (1.0 - eta),
        (1.0 + xi) * (1.0 - eta),
        (1.0 + xi) * (1.0 + eta),
        (1.0 - xi) * (1.0 + eta)
    ])
    return N

def evaluate_q1_shape_derivatives_ref(xi: float, eta: float) -> ndarray:
    """
    Evaluate derivatives of bilinear Q1 shape functions with respect to
    reference coordinates (xi, eta).
    Returns a 2D array of shape (2, 4), where row 0 is dN/dxi and row 1 is dN/deta.
    """
    dN_dref = 0.25 * array([
        [-(1.0 - eta),  (1.0 - eta), (1.0 + eta), -(1.0 + eta)], # dN/dxi
        [-(1.0 - xi),  -(1.0 + xi),  (1.0 + xi),   (1.0 - xi) ] # dN/deta
    ])
    return dN_dref

def compute_q1_mapping(gp: ndarray, elem_coords: ndarray) -> Tuple[ndarray, ndarray, float]:
    """
    Compute reference-to-physical mapping for a Q1 element.
    Args:
        gp: Reference point [xi, eta] (array of shape (2,))
        elem_coords: Node physical coordinates (array of shape (4, 2))
    Returns:
        N: Shape function values (shape (4,))
        dN_dphys: Physical derivatives dN/dx, dN/dy (shape (2, 4))
        detJ: Jacobian determinant (float)
    """
    xi, eta = gp[0], gp[1]
    N = evaluate_q1_shape_functions(xi, eta)
    dN_dref = evaluate_q1_shape_derivatives_ref(xi, eta)
    
    # Jacobian matrix J = dN_dref @ elem_coords of shape (2, 2)
    # J = [[dx/dxi,  dy/dxi],
    #      [dx/deta, dy/deta]]
    J = dN_dref @ elem_coords
    
    detJ = determinant(J)
    if detJ <= 0.0:
        raise ValueError(f"Jacobian determinant is non-positive: {detJ}")
        
    invJ = invert_matrix(J)
    
    # dN_dphys = invJ @ dN_dref of shape (2, 4)
    # Row 0: dN/dx, Row 1: dN/dy
    dN_dphys = invJ @ dN_dref
    
    return N, dN_dphys, detJ
