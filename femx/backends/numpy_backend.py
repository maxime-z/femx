import numpy as np
import scipy.sparse as sp

# Expose standard numpy functions and types for general use
from numpy import (
    array, zeros, ones, arange, eye, linspace, 
    dot, matmul, transpose, sum, sqrt, cos, sin, pi,
    abs, min, max, ndarray, float64, int32,
    outer, inner, cross, shape, reshape, zeros_like,
    ones_like, block, concatenate, stack
)

from numpy.linalg import det, inv

def invert_matrix(m):
    """Compute the inverse of a square matrix."""
    return np.linalg.inv(m)

def determinant(m):
    """Compute the determinant of a square matrix."""
    return np.linalg.det(m)

def solve_linear(A, b):
    """Solve the linear system A * x = b, handling both sparse and dense matrices."""
    if sp.issparse(A):
        from scipy.sparse.linalg import spsolve
        return spsolve(A.tocsr(), b)
    return np.linalg.solve(A, b)
