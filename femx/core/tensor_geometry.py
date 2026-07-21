import numpy as np
import torch
from typing import Tuple
from femx.core.mesh import Mesh
from femx.core.quadrature import get_quadrature_2d, get_quadrature_triangle
from femx.basis.lagrange import evaluate_q1_shape_functions, evaluate_q1_shape_derivatives_ref
from femx.basis.triangle import evaluate_t1_shape_functions, evaluate_t1_shape_derivatives_ref

class BatchedGeometry:
    """Container for physics-independent batched mesh geometric quantities."""
    def __init__(
        self,
        X: torch.Tensor,
        J: torch.Tensor,
        detJ: torch.Tensor,
        G: torch.Tensor,
        W_hat: torch.Tensor,
        B_hat: torch.Tensor,
        E: int,
        nen: int,
        dim: int,
        Q: int
    ):
        self.X = X          # Element coordinates (E, nen, dim)
        self.J = J          # Batched Jacobians (E, Q, dim, dim)
        self.detJ = detJ    # Determinants (E, Q)
        self.G = G          # Physical shape gradients (E, Q, nen, dim)
        self.W_hat = W_hat  # Quadrature weights (Q,)
        self.B_hat = B_hat  # Reference shape values (Q, nen)
        self.E = E
        self.nen = nen
        self.dim = dim
        self.Q = Q

def evaluate_batched_geometry(mesh: Mesh, device: str = "cpu", dtype: torch.dtype = torch.float64) -> BatchedGeometry:
    """
    Evaluates physics-independent batched geometric quantities across all elements.
    Returns BatchedGeometry object.
    """
    coords = torch.tensor(mesh.coords, dtype=dtype, device=device) # (N, dim)
    cells = torch.tensor(mesh.cells, dtype=torch.int64, device=device)     # (E, nen)
    E = mesh.n_elements
    nen = cells.shape[1]
    dim = coords.shape[1]
    
    X = coords[cells] # (E, nen, dim)
    
    if nen == 4:
        # Q1 Quad: 2x2 quadrature (Q=4)
        pts_np, wts_np = get_quadrature_2d(2, 2)
        Q = len(pts_np)
        B_hat_list = [evaluate_q1_shape_functions(pt[0], pt[1]) for pt in pts_np]
        dB_hat_list = [evaluate_q1_shape_derivatives_ref(pt[0], pt[1]) for pt in pts_np]
        dB_hat_q = torch.tensor(np.array([dB.T for dB in dB_hat_list]), dtype=dtype, device=device) # (Q, nen, dim)
    else:
        # T1 Triangle: 1-point centroid rule (Q=1)
        pts_np, wts_np = get_quadrature_triangle(1)
        Q = len(pts_np)
        B_hat_list = [evaluate_t1_shape_functions(pt[0], pt[1]) for pt in pts_np]
        dB_hat_list = [evaluate_t1_shape_derivatives_ref() for pt in pts_np]
        dB_hat_q = torch.tensor(np.array([dB.T for dB in dB_hat_list]), dtype=dtype, device=device) # (Q, nen, dim)
        
    B_hat = torch.tensor(np.array(B_hat_list), dtype=dtype, device=device) # (Q, nen)
    W_hat = torch.tensor(wts_np, dtype=dtype, device=device)               # (Q,)
    
    # Batched Jacobians J[e, q, c, d] = sum_a ( X[e, a, c] * dB_hat_q[q, a, d] )
    J = torch.einsum('eac,qad->eqcd', X, dB_hat_q)
    detJ = torch.linalg.det(J)
    J_inv_T = torch.linalg.inv(J).transpose(-1, -2)
    
    # Physical shape gradients G[e, q, a, c] = sum_d ( J_inv_T[e, q, c, d] * dB_hat_q[q, a, d] )
    G = torch.einsum('eqcd,qad->eqac', J_inv_T, dB_hat_q)
    
    return BatchedGeometry(
        X=X, J=J, detJ=detJ, G=G, W_hat=W_hat, B_hat=B_hat, E=E, nen=nen, dim=dim, Q=Q
    )
