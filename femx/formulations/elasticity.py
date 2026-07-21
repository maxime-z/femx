from femx.formulations.base import Formulation
from femx.backends.numpy_backend import ndarray, zeros
from femx.materials.linear_elastic import LinearElasticMaterial

class LinearElasticityFormulation(Formulation):
    """
    Formulation for 2D linear elasticity (Plane Strain or Plane Stress).
    """
    def __init__(self, material: LinearElasticMaterial, mode: str = "plane_strain"):
        self.material = material
        self.mode = mode

    def compute_element_matrices(
        self,
        elem_coords: ndarray,
        quadrature_pts: ndarray,
        quadrature_wts: ndarray,
        body_load: ndarray = None,
        is_nurbs: bool = False,
        patch = None,
        span_u: int = 0,
        span_v: int = 0
    ):
        """
        Compute element stiffness Ke, mass Me, and load force vector fe.
        """
        n_local = elem_coords.shape[0]
        n_dofs_local = 2 * n_local
        
        Ke = zeros((n_dofs_local, n_dofs_local))
        Me = zeros((n_dofs_local, n_dofs_local))
        fe = zeros(n_dofs_local)

        D = self.material.get_constitutive_matrix(mode=self.mode)
        rho = self.material.get_property("rho")
        
        if body_load is None:
            body_load = zeros(2)

        for gp, w in zip(quadrature_pts, quadrature_wts):
            from femx.basis.lagrange import compute_q1_mapping
            N, dN_dphys, detJ = compute_q1_mapping(gp, elem_coords)
            
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
                
            # Ke += (B^T D B) * dV
            Ke += (B.T @ D @ B) * dV
            
            # Me += rho * N_i * N_j * dV (block diagonal)
            for i in range(n_local):
                # Body load contribution
                fe[2 * i]     += body_load[0] * N[i] * dV
                fe[2 * i + 1] += body_load[1] * N[i] * dV
                
                for j in range(n_local):
                    mass_val = rho * N[i] * N[j] * dV
                    Me[2 * i, 2 * j]         += mass_val
                    Me[2 * i + 1, 2 * j + 1] += mass_val
                    
        return Ke, Me, fe

    def get_physical_tensors(self, geom, device: str = "cpu", dtype = None):
        """
        Return true 4th-order physical elasticity tensor for linear elasticity:
        - C_4th: 4th-order elasticity tensor of shape (E, Q, dim, dim, dim, dim)
        - M_0th: 0th-order density scalar of shape (E, Q)
        - f_body: body force vector of shape (E, Q, dim)
        """
        import torch
        if dtype is None:
            dtype = torch.float64
            
        C4_np = self.material.get_elasticity_tensor_4th(mode=self.mode, dim=geom.dim) # (dim, dim, dim, dim)
        C_4th = torch.tensor(C4_np, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0).expand(
            geom.E, geom.Q, geom.dim, geom.dim, geom.dim, geom.dim
        )
        
        rho = self.material.get_property("rho")
        M_0th = torch.full((geom.E, geom.Q), fill_value=rho, dtype=dtype, device=device)
        f_body = None
        
        return C_4th, M_0th, f_body
