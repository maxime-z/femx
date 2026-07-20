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
