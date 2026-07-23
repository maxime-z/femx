import numpy as np
from typing import Tuple
from femx.backends.numpy_backend import ndarray, array, invert_matrix, determinant
from femx.basis.element import ElementBasis
from femx.core.mesh import NurbsPatch
from femx.core.quadrature import get_quadrature_2d

def find_span(n: int, p: int, u: float, knots: ndarray) -> int:
    """Find the knot span index for coordinate u."""
    if u >= knots[n + 1]:
        return n
        
    low = p
    high = n + 1
    mid = (low + high) // 2
    while u < knots[mid] or u >= knots[mid + 1]:
        if u < knots[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2
    return mid

def ders_basis_functions(i: int, u: float, p: int, n: int, knots: ndarray) -> ndarray:
    """Compute B-spline basis functions and derivatives up to order n."""
    ndu = np.zeros((p + 1, p + 1))
    ndu[0, 0] = 1.0
    left = np.zeros(p + 1)
    right = np.zeros(p + 1)
    
    for j in range(1, p + 1):
        left[j] = u - knots[i + 1 - j]
        right[j] = knots[i + j] - u
        saved = 0.0
        for r in range(j):
            ndu[j, r] = right[r + 1] + left[j - r]
            temp = ndu[r, j - 1] / ndu[j, r]
            ndu[r, j] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        ndu[j, j] = saved
        
    ders = np.zeros((n + 1, p + 1))
    for j in range(p + 1):
        ders[0, j] = ndu[j, p]
        
    a = np.zeros((2, p + 1))
    for r in range(p + 1):
        s1 = 0
        s2 = 1
        a[0, 0] = 1.0
        for k in range(1, n + 1):
            d = 0.0
            rk = r - k
            pk = p - k
            if r >= k:
                a[s2, 0] = a[s1, 0] / ndu[pk + 1, rk]
                d = a[s2, 0] * ndu[rk, pk]
            
            j1 = 1 if rk >= -1 else -rk
            j2 = k - 1 if r - 1 <= pk else p - r
            
            for j in range(j1, j2 + 1):
                a[s2, j] = (a[s1, j] - a[s1, j - 1]) / ndu[pk + 1, rk + j]
                d += a[s2, j] * ndu[rk + j, pk]
                
            if r <= pk:
                a[s2, k] = -a[s1, k - 1] / ndu[pk + 1, r]
                d += a[s2, k] * ndu[r, pk]
                
            ders[k, r] = d
            s1, s2 = s2, s1
            
    r_factor = p
    for k in range(1, n + 1):
        for j in range(p + 1):
            ders[k, j] *= r_factor
        r_factor *= (p - k)
        
    return ders

class NurbsBasis(ElementBasis):
    """
    NURBS Isogeometric Element Basis.
    Wraps a NurbsPatch and active knot span (span_u, span_v).
    """
    def __init__(self, patch: NurbsPatch, span_u: int = 0, span_v: int = 0):
        self.patch = patch
        self.span_u = span_u
        self.span_v = span_v

    @property
    def n_dofs_per_element(self) -> int:
        return (self.patch.p_u + 1) * (self.patch.p_v + 1)

    @property
    def dim(self) -> int:
        return 2

    def evaluate_shape_functions(self, ref_coords: ndarray) -> ndarray:
        R, _, _ = compute_nurbs_mapping(ref_coords, self.patch, self.span_u, self.span_v)
        return R

    def evaluate_shape_derivatives(self, ref_coords: ndarray) -> ndarray:
        _, dR_dphys, _ = compute_nurbs_mapping(ref_coords, self.patch, self.span_u, self.span_v)
        return dR_dphys

    def compute_mapping(self, ref_coords: ndarray, elem_coords: ndarray = None) -> Tuple[ndarray, ndarray, float]:
        return compute_nurbs_mapping(ref_coords, self.patch, self.span_u, self.span_v)

    def get_default_quadrature(self) -> Tuple[ndarray, ndarray]:
        nu = self.patch.p_u + 1
        nv = self.patch.p_v + 1
        return get_quadrature_2d(nu, nv)

def compute_nurbs_mapping(gp_ref: ndarray, patch: NurbsPatch, span_u: int, span_v: int) -> Tuple[ndarray, ndarray, float]:
    """Compute reference-to-physical mapping for a NURBS element span."""
    u1, u2 = patch.knots_u[span_u], patch.knots_u[span_u + 1]
    v1, v2 = patch.knots_v[span_v], patch.knots_v[span_v + 1]
    
    hu = 0.5 * (u2 - u1)
    hv = 0.5 * (v2 - v1)
    
    u = hu * gp_ref[0] + 0.5 * (u2 + u1)
    v = hv * gp_ref[1] + 0.5 * (v2 + v1)
    
    detJ_PR = hu * hv
    
    ders_u = ders_basis_functions(span_u, u, patch.p_u, 1, patch.knots_u)
    ders_v = ders_basis_functions(span_v, v, patch.p_v, 1, patch.knots_v)
    
    n_local = (patch.p_u + 1) * (patch.p_v + 1)
    B = np.zeros(n_local)
    dB_du = np.zeros(n_local)
    dB_dv = np.zeros(n_local)
    
    w_local = np.zeros(n_local)
    elem_coords = np.zeros((n_local, 2))
    
    for j in range(patch.p_v + 1):
        idx_v = span_v - patch.p_v + j
        for i in range(patch.p_u + 1):
            idx_u = span_u - patch.p_u + i
            local_idx = j * (patch.p_u + 1) + i
            
            N_u = ders_u[0, i]
            dN_du = ders_u[1, i]
            N_v = ders_v[0, j]
            dN_dv = ders_v[1, j]
            
            B[local_idx] = N_u * N_v
            dB_du[local_idx] = dN_du * N_v
            dB_dv[local_idx] = N_u * dN_dv
            
            w_local[local_idx] = patch.weights[idx_u, idx_v]
            elem_coords[local_idx] = patch.control_points[idx_u, idx_v]
            
    W = np.dot(w_local, B)
    dW_du = np.dot(w_local, dB_du)
    dW_dv = np.dot(w_local, dB_dv)
    
    R = (w_local * B) / W
    
    dR_du = w_local * (dB_du * W - B * dW_du) / (W * W)
    dR_dv = w_local * (dB_dv * W - B * dW_dv) / (W * W)
    
    dR_dparam = np.vstack([dR_du, dR_dv])
    J_PP = dR_dparam @ elem_coords
    
    detJ_PP = determinant(J_PP)
    detJ = detJ_PP * detJ_PR
    
    if detJ <= 0.0:
        raise ValueError(f"Jacobian determinant is non-positive: {detJ}")
        
    invJ_PP = invert_matrix(J_PP)
    dR_dphys = invJ_PP @ dR_dparam
    
    return R, dR_dphys, detJ
