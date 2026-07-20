from femx.formulations.base import Formulation
from femx.backends.numpy_backend import ndarray, zeros, outer
from femx.materials.linear_heat import LinearHeatMaterial

class HeatConductionFormulation(Formulation):
    """
    Formulation for linear heat conduction (Laplace / Poisson solver).
    """
    def __init__(self, material: LinearHeatMaterial):
        self.material = material

    def compute_element_matrices(
        self,
        elem_coords: ndarray,
        quadrature_pts: ndarray,
        quadrature_wts: ndarray,
        is_nurbs: bool = False,
        patch = None,
        span_u: int = 0,
        span_v: int = 0,
        body_load: float = 0.0
    ):
        """
        Compute Ke, Me, and fe for a single element.
        """
        n_local = elem_coords.shape[0]
        Ke = zeros((n_local, n_local))
        Me = zeros((n_local, n_local))
        fe = zeros(n_local)

        D = self.material.get_constitutive_matrix(dim=2)
        rho = self.material.get_property("rho")
        C = self.material.get_property("C")

        if body_load is None:
            body_load = 0.0

        for gp, w in zip(quadrature_pts, quadrature_wts):
            if is_nurbs:
                from femx.basis.nurbs import compute_nurbs_mapping
                N, dN_dphys, detJ = compute_nurbs_mapping(gp, patch, span_u, span_v)
            else:
                from femx.basis.lagrange import compute_q1_mapping
                N, dN_dphys, detJ = compute_q1_mapping(gp, elem_coords)

            dV = detJ * w

            # B matrix is shape (2, n_local)
            B = dN_dphys

            Ke += (B.T @ D @ B) * dV
            Me += (rho * C * outer(N, N)) * dV
            fe += (body_load * N) * dV

        return Ke, Me, fe
