from typing import ClassVar, Tuple, Optional
from femx.formulations.base import Formulation
from femx.backends.numpy_backend import ndarray, zeros, outer
from femx.materials.linear_heat import LinearHeatMaterial
from femx.basis.element import ElementBasis

class HeatConductionFormulation(Formulation[LinearHeatMaterial]):
    """
    Formulation for linear heat conduction (Laplace / Poisson solver).
    """
    field_names: ClassVar[Tuple[str, ...]] = ("T",)

    def __init__(self, material: LinearHeatMaterial):
        super().__init__(material)

    def compute_element_matrices(
        self,
        elem_coords: ndarray,
        quadrature_pts: ndarray,
        quadrature_wts: ndarray,
        elem_basis: Optional[ElementBasis] = None,
        is_nurbs: bool = False,
        patch = None,
        span_u: int = 0,
        span_v: int = 0,
        body_load: float = 0.0
    ):
        """Compute Ke, Me, and fe for a single element."""
        n_local = elem_coords.shape[0]
        Ke = zeros((n_local, n_local))
        Me = zeros((n_local, n_local))
        fe = zeros(n_local)

        D = self.material.get_constitutive_matrix(dim=2)
        rho = self.material.get_property("rho")
        C = self.material.get_property("C")

        if body_load is None:
            body_load = 0.0

        if elem_basis is None and not is_nurbs:
            from femx.basis.lagrange import LagrangeQuad
            elem_basis = LagrangeQuad(p=1)

        for gp, w in zip(quadrature_pts, quadrature_wts):
            if is_nurbs and getattr(elem_basis, 'compute_mapping', None) is None: # handle NURBS
                from femx.basis.nurbs import compute_nurbs_mapping
                N, dN_dphys, detJ = compute_nurbs_mapping(gp, patch, span_u, span_v)
            else:
                N, dN_dphys, detJ = elem_basis.compute_mapping(gp, elem_coords)

            dV = detJ * w

            # B matrix is shape (2, n_local)
            B = dN_dphys

            Ke += (B.T @ D @ B) * dV
            Me += (rho * C * outer(N, N)) * dV
            fe += (body_load * N) * dV

        return Ke, Me, fe

    def get_physical_tensors(self, geom, device: str = "cpu", dtype = None):
        import torch
        if dtype is None:
            dtype = torch.float64
            
        K_np = self.material.get_constitutive_matrix(dim=geom.dim)
        K_2nd = torch.tensor(K_np, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0).expand(geom.E, geom.Q, geom.dim, geom.dim)
        
        rho = self.material.get_property("rho")
        C_cap = self.material.get_property("C")
        M_0th = torch.full((geom.E, geom.Q), fill_value=rho * C_cap, dtype=dtype, device=device)
        f_body = None
        
        return K_2nd, M_0th, f_body

    def compute_batch_map(self, geom, tensors, device: str = "cpu", dtype=None):
        import torch
        K_2nd, M_0th, f_body = tensors
        # 2nd-order physical tensor contraction (Scalar Heat/Diffusion)
        K_local = torch.einsum('q,eq,eqai,eqij,eqbj->eab', geom.W_hat, geom.detJ, geom.G, K_2nd, geom.G)
        M_local = torch.einsum('q,eq,eq,qa,qb->eab', geom.W_hat, geom.detJ, M_0th, geom.B_hat, geom.B_hat)
        F_local = torch.zeros((geom.E, geom.nen), dtype=dtype, device=device)
        return K_local, M_local, F_local
