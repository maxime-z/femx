import numpy as np
import torch
from typing import Tuple
from femx.core.mesh import Mesh
from femx.core.quadrature import get_quadrature_2d, get_quadrature_triangle, get_quadrature_3d
from femx.basis.lagrange import LagrangeQuad, LagrangeTriangle

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
    from femx.basis.lagrange import LagrangeHex

    coords = torch.tensor(mesh.coords, dtype=dtype, device=device) # (N, dim)
    cells = torch.tensor(mesh.cells, dtype=torch.int64, device=device)     # (E, nen)
    E = mesh.n_elements
    nen = cells.shape[1]
    dim = coords.shape[1]
    
    X = coords[cells] # (E, nen, dim)
    
    if nen == 8 and dim == 3:
        pts_np, wts_np = get_quadrature_3d(2, 2, 2)
        Q = len(pts_np)
        basis = LagrangeHex(p=1)
        B_hat_list = [basis.evaluate_shape_functions(pt) for pt in pts_np]
        dB_hat_list = [basis.evaluate_shape_derivatives(pt) for pt in pts_np]
        dB_hat_q = torch.tensor(np.array([dB.T for dB in dB_hat_list]), dtype=dtype, device=device)
    elif nen == 4:
        # Q1 Quad: 2x2 quadrature (Q=4)
        pts_np, wts_np = get_quadrature_2d(2, 2)
        Q = len(pts_np)
        basis = LagrangeQuad(p=1)
        B_hat_list = [basis.evaluate_shape_functions(pt) for pt in pts_np]
        dB_hat_list = [basis.evaluate_shape_derivatives(pt) for pt in pts_np]
        dB_hat_q = torch.tensor(np.array([dB.T for dB in dB_hat_list]), dtype=dtype, device=device) # (Q, nen, dim)
    else:
        # T1 Triangle: 1-point centroid rule (Q=1)
        pts_np, wts_np = get_quadrature_triangle(1)
        Q = len(pts_np)
        basis = LagrangeTriangle(p=1)
        B_hat_list = [basis.evaluate_shape_functions(pt) for pt in pts_np]
        dB_hat_list = [basis.evaluate_shape_derivatives(pt) for pt in pts_np]
        dB_hat_q = torch.tensor(np.array([dB.T for dB in dB_hat_list]), dtype=dtype, device=device) # (Q, nen, dim)
        
    B_hat = torch.tensor(np.array(B_hat_list), dtype=dtype, device=device) # (Q, nen)
    W_hat = torch.tensor(wts_np, dtype=dtype, device=device)               # (Q,)
    
    # Batched Jacobians J[e, q, c, d] = sum_a ( X[e, a, c] * dB_hat_q[q, a, d] )
    J = torch.einsum('eac,qad->eqcd', X, dB_hat_q)
    detJ = torch.linalg.det(J.contiguous())
    J_inv_T = torch.linalg.inv(J).transpose(-1, -2)
    
    # Physical shape gradients G[e, q, a, c] = sum_d ( J_inv_T[e, q, c, d] * dB_hat_q[q, a, d] )
    G = torch.einsum('eqcd,qad->eqac', J_inv_T, dB_hat_q)
    
    return BatchedGeometry(
        X=X, J=J, detJ=detJ, G=G, W_hat=W_hat, B_hat=B_hat, E=E, nen=nen, dim=dim, Q=Q
    )


def quadrature_coordinates(geom: BatchedGeometry) -> np.ndarray:
    """Physical coordinates of every Gauss point, shape ``(n_elem, n_q, dim)``."""
    xyz = torch.einsum("qa,ead->eqd", geom.B_hat, geom.X)
    return xyz.detach().cpu().numpy()


def integrate_nodal_source(geom: BatchedGeometry, f_gp: torch.Tensor) -> torch.Tensor:
    """Integrate ``N_a * f * dV`` into node-major element forces, shape ``(E, nen * n_comp)``."""
    nodal = torch.einsum("q,eq,eqc,qa->eac", geom.W_hat, geom.detJ, f_gp, geom.B_hat)
    return nodal.reshape(geom.E, geom.nen * f_gp.shape[-1])
