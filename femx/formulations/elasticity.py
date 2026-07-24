from typing import ClassVar, Tuple, Optional
from femx.formulations.base import Formulation
from femx.backends.numpy_backend import ndarray, zeros
from femx.materials.linear_elastic import LinearElasticMaterial
from femx.basis.element import ElementBasis

class LinearElasticityFormulation(Formulation[LinearElasticMaterial]):
    """
    Formulation for 2D linear elasticity (Plane Strain or Plane Stress).
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
        body_load: ndarray = None,
        elem_basis: Optional[ElementBasis] = None,
        is_nurbs: bool = False,
        patch = None,
        span_u: int = 0,
        span_v: int = 0
    ):
        """Compute element stiffness Ke, mass Me, and load force vector fe."""
        n_local = elem_coords.shape[0]
        n_dofs_local = 2 * n_local
        
        Ke = zeros((n_dofs_local, n_dofs_local))
        Me = zeros((n_dofs_local, n_dofs_local))
        fe = zeros(n_dofs_local)

        D = self.material.get_constitutive_matrix(mode=self.mode)
        rho = self.material.get_property("rho")
        
        if body_load is None:
            body_load = zeros(2)

        if elem_basis is None and not is_nurbs:
            from femx.basis.lagrange import LagrangeQuad
            elem_basis = LagrangeQuad(p=1)

        for gp, w in zip(quadrature_pts, quadrature_wts):
            if is_nurbs and getattr(elem_basis, 'compute_mapping', None) is None:
                from femx.basis.nurbs import compute_nurbs_mapping
                N, dN_dphys, detJ = compute_nurbs_mapping(gp, patch, span_u, span_v)
            else:
                N, dN_dphys, detJ = elem_basis.compute_mapping(gp, elem_coords)
            
            dV = detJ * w
            
            # Construct strain-displacement B matrix of shape (3, 2 * n_local)
            B = zeros((3, n_dofs_local))
            for i in range(n_local):
                dN_dx = dN_dphys[0, i]
                dN_dy = dN_dphys[1, i]
                B[0, 2 * i]     = dN_dx      # epsilon_xx due to u_x
                B[1, 2 * i + 1] = dN_dy      # epsilon_yy due to u_y
                B[2, 2 * i]     = dN_dy      # gamma_xy due to u_x
                B[2, 2 * i + 1] = dN_dx      # gamma_xy due to u_y
                
            Ke += (B.T @ D @ B) * dV
            
            for i in range(n_local):
                fe[2 * i]     += body_load[0] * N[i] * dV
                fe[2 * i + 1] += body_load[1] * N[i] * dV
                
                for j in range(n_local):
                    mass_val = rho * N[i] * N[j] * dV
                    Me[2 * i, 2 * j]         += mass_val
                    Me[2 * i + 1, 2 * j + 1] += mass_val
                    
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

    def compute_batch_map(self, geom, tensors, device: str = "cpu", dtype=None):
        import torch
        C_4th, M_0th, f_body = tensors
        k_dofs = geom.nen * geom.dim
        # 4th-order physical tensor contraction (Vector Elasticity/Mechanics)
        K_tensor = torch.einsum('q,eq,eqaj,eqijkl,eqbl->eaibk', geom.W_hat, geom.detJ, geom.G, C_4th, geom.G)
        K_local = K_tensor.reshape(geom.E, k_dofs, k_dofs)
        
        M_scalar = torch.einsum('q,eq,eq,qa,qb->eab', geom.W_hat, geom.detJ, M_0th, geom.B_hat, geom.B_hat)
        I_dim = torch.eye(geom.dim, dtype=dtype, device=device)
        M_tensor = torch.einsum('eab,ij->eaibj', M_scalar, I_dim)
        M_local = M_tensor.reshape(geom.E, k_dofs, k_dofs)
        
        F_local = torch.zeros((geom.E, k_dofs), dtype=dtype, device=device)
        return K_local, M_local, F_local
