from typing import ClassVar, Tuple
from femx.formulations.base import Formulation
from femx.backends.numpy_backend import ndarray, zeros, outer
from femx.materials.linear_heat import LinearHeatMaterial
from femx.basis.element import ElementBasis
from femx.core.loads import sample_body_load, sample_body_load_batch
from femx.core.tensor_geometry import integrate_nodal_source, quadrature_coordinates

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
        elem_basis: ElementBasis,
        body_load: float = 0.0,
        elem_idx: int = 0,
        **kwargs
    ):
        """Compute Ke, Me, and fe for a single element."""
        n_local = elem_coords.shape[0]
        Ke = zeros((n_local, n_local))
        Me = zeros((n_local, n_local))
        fe = zeros(n_local)

        D = self.material.get_constitutive_matrix(dim=elem_coords.shape[1])
        rho = self.material.get_property("rho")
        C = self.material.get_property("C")

        for q, (gp, w) in enumerate(zip(quadrature_pts, quadrature_wts)):
            N, dN_dphys, detJ = elem_basis.compute_mapping(gp, elem_coords)
            dV = detJ * w
            B = dN_dphys
            xyz = N @ elem_coords
            f_gp = sample_body_load(body_load, xyz, elem_idx=elem_idx, q_idx=q, n_comp=1)
            Ke += (B.T @ D @ B) * dV
            Me += (rho * C * outer(N, N)) * dV
            fe += (f_gp[0] * N) * dV

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

    def compute_batch_map(self, geom, tensors, device: str = "cpu", dtype=None, body_load=None):
        import torch
        K_2nd, M_0th, f_body = tensors
        K_local = torch.einsum('q,eq,eqai,eqij,eqbj->eab', geom.W_hat, geom.detJ, geom.G, K_2nd, geom.G)
        M_local = torch.einsum('q,eq,eq,qa,qb->eab', geom.W_hat, geom.detJ, M_0th, geom.B_hat, geom.B_hat)
        f_gp = torch.tensor(
            sample_body_load_batch(body_load, quadrature_coordinates(geom), n_comp=1),
            dtype=dtype, device=device,
        )
        F_local = integrate_nodal_source(geom, f_gp)
        return K_local, M_local, F_local
