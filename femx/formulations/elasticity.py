from typing import ClassVar, Tuple
from femx.formulations.base import Formulation
from femx.backends.numpy_backend import ndarray, zeros
from femx.materials.linear_elastic import LinearElasticMaterial
from femx.basis.element import ElementBasis
from femx.core.loads import sample_body_load, sample_body_load_batch
from femx.core.tensor_geometry import integrate_nodal_source, quadrature_coordinates

class LinearElasticityFormulation(Formulation[LinearElasticMaterial]):
    """
    Formulation for 2D/3D linear elasticity.
    Modes: plane_strain, plane_stress (2D), or 3d.
    """
    field_names: ClassVar[Tuple[str, ...]] = ("u",)

    def __init__(self, material: LinearElasticMaterial, mode: str = "plane_strain"):
        super().__init__(material)
        self.mode = mode

    def compute_element_matrices(
        self,
        elem_coords: ndarray,
        quadrature_pts: ndarray,
        quadrature_wts: ndarray,
        elem_basis: ElementBasis,
        body_load: ndarray = None,
        elem_idx: int = 0,
        **kwargs
    ):
        """Compute element stiffness Ke, mass Me, and load force vector fe."""
        n_local = elem_coords.shape[0]
        spatial_dim = elem_coords.shape[1]
        n_dofs_local = spatial_dim * n_local

        Ke = zeros((n_dofs_local, n_dofs_local))
        Me = zeros((n_dofs_local, n_dofs_local))
        fe = zeros(n_dofs_local)

        mode = self.mode
        if spatial_dim == 3:
            mode = "3d"
        D = self.material.get_constitutive_matrix(mode=mode)
        rho = self.material.get_property("rho")

        for q, (gp, w) in enumerate(zip(quadrature_pts, quadrature_wts)):
            N, dN_dphys, detJ = elem_basis.compute_mapping(gp, elem_coords)
            dV = detJ * w
            xyz = N @ elem_coords
            f_gp = sample_body_load(
                body_load, xyz, elem_idx=elem_idx, q_idx=q, n_comp=spatial_dim
            )

            if spatial_dim == 2:
                B = zeros((3, n_dofs_local))
                for i in range(n_local):
                    dN_dx = dN_dphys[0, i]
                    dN_dy = dN_dphys[1, i]
                    B[0, 2 * i]     = dN_dx
                    B[1, 2 * i + 1] = dN_dy
                    B[2, 2 * i]     = dN_dy
                    B[2, 2 * i + 1] = dN_dx
            else:
                B = zeros((6, n_dofs_local))
                for i in range(n_local):
                    dN_dx = dN_dphys[0, i]
                    dN_dy = dN_dphys[1, i]
                    dN_dz = dN_dphys[2, i]
                    B[0, 3 * i]     = dN_dx
                    B[1, 3 * i + 1] = dN_dy
                    B[2, 3 * i + 2] = dN_dz
                    B[3, 3 * i + 1] = dN_dz
                    B[3, 3 * i + 2] = dN_dy
                    B[4, 3 * i]     = dN_dz
                    B[4, 3 * i + 2] = dN_dx
                    B[5, 3 * i]     = dN_dy
                    B[5, 3 * i + 1] = dN_dx

            Ke += (B.T @ D @ B) * dV

            for i in range(n_local):
                for c in range(spatial_dim):
                    fe[spatial_dim * i + c] += f_gp[c] * N[i] * dV
                for j in range(n_local):
                    mass_val = rho * N[i] * N[j] * dV
                    for c in range(spatial_dim):
                        Me[spatial_dim * i + c, spatial_dim * j + c] += mass_val

        return Ke, Me, fe

    def get_physical_tensors(self, geom, device: str = "cpu", dtype = None):
        import torch
        if dtype is None:
            dtype = torch.float64

        C4_np = self.material.get_elasticity_tensor_4th(mode=self.mode, dim=geom.dim)
        C_4th = torch.tensor(C4_np, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0).expand(
            geom.E, geom.Q, geom.dim, geom.dim, geom.dim, geom.dim
        )

        rho = self.material.get_property("rho")
        M_0th = torch.full((geom.E, geom.Q), fill_value=rho, dtype=dtype, device=device)
        f_body = None

        return C_4th, M_0th, f_body

    def compute_batch_map(self, geom, tensors, device: str = "cpu", dtype=None, body_load=None):
        import torch
        C_4th, M_0th, f_body = tensors
        k_dofs = geom.nen * geom.dim
        K_tensor = torch.einsum('q,eq,eqaj,eqijkl,eqbl->eaibk', geom.W_hat, geom.detJ, geom.G, C_4th, geom.G)
        K_local = K_tensor.reshape(geom.E, k_dofs, k_dofs)

        M_scalar = torch.einsum('q,eq,eq,qa,qb->eab', geom.W_hat, geom.detJ, M_0th, geom.B_hat, geom.B_hat)
        I_dim = torch.eye(geom.dim, dtype=dtype, device=device)
        M_tensor = torch.einsum('eab,ij->eaibj', M_scalar, I_dim)
        M_local = M_tensor.reshape(geom.E, k_dofs, k_dofs)

        f_gp = torch.tensor(
            sample_body_load_batch(body_load, quadrature_coordinates(geom), n_comp=geom.dim),
            dtype=dtype, device=device,
        )
        F_local = integrate_nodal_source(geom, f_gp)
        return K_local, M_local, F_local
