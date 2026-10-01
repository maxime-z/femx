from typing import ClassVar, Tuple, Optional, List
from femx.formulations.base import Formulation
from femx.backends.numpy_backend import ndarray, zeros, outer, eye
from femx.materials.thermoelastic import LinearThermoelasticMaterial
from femx.basis.element import ElementBasis
from femx.core.loads import sample_body_load, sample_body_load_batch, split_coupled_body_load
from femx.core.tensor_geometry import integrate_nodal_source, quadrature_coordinates

class LinearThermoelasticityFormulation(Formulation[LinearThermoelasticMaterial]):
    """
    Coupled formulation for Linear Thermoelasticity (u + T).
    Combines 2D/3D linear elasticity and steady-state thermal conduction.
    """
    field_names: ClassVar[Tuple[str, ...]] = ("u", "T")

    def __init__(self, material: LinearThermoelasticMaterial, mode: str = "plane_strain"):
        super().__init__(material)
        self.mode = mode

    def compute_element_matrices(
        self,
        elem_coords: ndarray,
        quadrature_pts: ndarray,
        quadrature_wts: ndarray,
        elem_basis: ElementBasis,
        body_load=None,
        elem_idx: int = 0,
        **kwargs
    ):
        """
        Compute coupled local element stiffness Ke, mass Me, and forcing vector fe.
        """
        n_local = elem_coords.shape[0]
        n_u_dofs = 2 * n_local
        n_T_dofs = n_local
        n_tot = n_u_dofs + n_T_dofs
        
        Ke = zeros((n_tot, n_tot))
        Me = zeros((n_tot, n_tot))
        fe = zeros(n_tot)
        
        D_mech = self.material.get_constitutive_matrix(mode=self.mode)
        K_th = self.material.get_thermal_conductivity_matrix(dim=2)
        m_th = self.material.get_thermal_coupling_vector(mode=self.mode)
        
        rho = self.material.get_property("rho")
        C_cap = self.material.get_property("C_cap")
        T0 = self.material.get_property("T0")
        load_u, load_T = split_coupled_body_load(body_load)
        spatial_dim = 2

        for q, (gp, w) in enumerate(zip(quadrature_pts, quadrature_wts)):
            N, dN_dphys, detJ = elem_basis.compute_mapping(gp, elem_coords)
                
            dV = detJ * w
            xyz = N @ elem_coords
            f_u = sample_body_load(load_u, xyz, elem_idx=elem_idx, q_idx=q, n_comp=spatial_dim)
            f_T = sample_body_load(load_T, xyz, elem_idx=elem_idx, q_idx=q, n_comp=1)
            
            # 1. Mechanical Strain-Displacement Matrix B_u: shape (3, n_u_dofs)
            B_u = zeros((3, n_u_dofs))
            for i in range(n_local):
                dN_dx = dN_dphys[0, i]
                dN_dy = dN_dphys[1, i]
                B_u[0, 2 * i]     = dN_dx
                B_u[1, 2 * i + 1] = dN_dy
                B_u[2, 2 * i]     = dN_dy
                B_u[2, 2 * i + 1] = dN_dx
                
            # 2. Thermal Gradient Matrix B_T: shape (2, n_T_dofs)
            B_T = dN_dphys
            
            # K_uu block (n_u_dofs x n_u_dofs)
            Ke[0:n_u_dofs, 0:n_u_dofs] += (B_u.T @ D_mech @ B_u) * dV
            
            # K_uT block (n_u_dofs x n_T_dofs): - B_u^T * m_th * N_T
            # m_th is shape (3,), B_u is shape (3, n_u_dofs)
            m_N = outer(m_th, N) # shape (3, n_T_dofs)
            K_uT_local = - (B_u.T @ m_N) * dV
            Ke[0:n_u_dofs, n_u_dofs:n_tot] += K_uT_local
            
            # K_TT block (n_T_dofs x n_T_dofs)
            Ke[n_u_dofs:n_tot, n_u_dofs:n_tot] += (B_T.T @ K_th @ B_T) * dV
            
            # Mass block M_uu (density rho)
            for i in range(n_local):
                for j in range(n_local):
                    m_val_u = rho * N[i] * N[j] * dV
                    Me[2 * i, 2 * j]         += m_val_u
                    Me[2 * i + 1, 2 * j + 1] += m_val_u
                    
                    m_val_T = rho * C_cap * N[i] * N[j] * dV
                    Me[n_u_dofs + i, n_u_dofs + j] += m_val_T

            for i in range(n_local):
                fe[2 * i] += f_u[0] * N[i] * dV
                fe[2 * i + 1] += f_u[1] * N[i] * dV
                fe[n_u_dofs + i] += f_T[0] * N[i] * dV
                    
            # Forcing offset term due to T0: f_u = - K_uT * T0_vec
            if T0 != 0.0:
                T0_vec = zeros(n_T_dofs)
                T0_vec[:] = T0
                fe[0:n_u_dofs] -= K_uT_local @ T0_vec
                
        return Ke, Me, fe

    def get_physical_tensors(self, geom, device: str = "cpu", dtype = None):
        import torch
        if dtype is None:
            dtype = torch.float64
            
        C4_np = self.material.get_elasticity_tensor_4th(mode=self.mode, dim=geom.dim)
        C_4th = torch.tensor(C4_np, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0).expand(
            geom.E, geom.Q, geom.dim, geom.dim, geom.dim, geom.dim
        )
        
        M_th_np = self.material.get_thermal_coupling_tensor_2nd(mode=self.mode, dim=geom.dim)
        M_th = torch.tensor(M_th_np, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0).expand(
            geom.E, geom.Q, geom.dim, geom.dim
        )
        
        K_th_np = self.material.get_thermal_conductivity_matrix(dim=geom.dim)
        K_th = torch.tensor(K_th_np, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0).expand(
            geom.E, geom.Q, geom.dim, geom.dim
        )
        
        rho = self.material.get_property("rho")
        C_cap = self.material.get_property("C_cap")
        
        M_0th_u = torch.full((geom.E, geom.Q), fill_value=rho, dtype=dtype, device=device)
        M_0th_T = torch.full((geom.E, geom.Q), fill_value=rho * C_cap, dtype=dtype, device=device)
        
        T0 = self.material.get_property("T0")
        
        return C_4th, M_th, K_th, M_0th_u, M_0th_T, T0

    def compute_batch_map(self, geom, tensors, device: str = "cpu", dtype=None, body_load=None):
        import torch
        C_4th, M_th, K_th, M_0th_u, M_0th_T, T0 = tensors
        u_dofs = geom.nen * geom.dim
        T_dofs = geom.nen
        total_dofs = u_dofs + T_dofs

        # 1. K_uu (4th-order mechanical contraction)
        K_uu_tensor = torch.einsum('q,eq,eqaj,eqijkl,eqbl->eaibk', geom.W_hat, geom.detJ, geom.G, C_4th, geom.G)
        K_uu = K_uu_tensor.reshape(geom.E, u_dofs, u_dofs)

        # 2. K_uT (2nd-order thermal coupling contraction: G_aj * M_th_ij * N_b)
        K_uT_tensor = - torch.einsum('q,eq,eqaj,eqij,qb->eaib', geom.W_hat, geom.detJ, geom.G, M_th, geom.B_hat)
        K_uT = K_uT_tensor.reshape(geom.E, u_dofs, T_dofs)

        # 3. K_TT (2nd-order thermal conductivity contraction)
        K_TT = torch.einsum('q,eq,eqai,eqij,eqbj->eab', geom.W_hat, geom.detJ, geom.G, K_th, geom.G)

        # 4. M_uu and M_TT (Mass matrices)
        M_uu_scalar = torch.einsum('q,eq,eq,qa,qb->eab', geom.W_hat, geom.detJ, M_0th_u, geom.B_hat, geom.B_hat)
        M_TT = torch.einsum('q,eq,eq,qa,qb->eab', geom.W_hat, geom.detJ, M_0th_T, geom.B_hat, geom.B_hat)
        
        I_dim = torch.eye(geom.dim, dtype=dtype, device=device)
        M_uu = torch.einsum('eab,ij->eaibj', M_uu_scalar, I_dim).reshape(geom.E, u_dofs, u_dofs)

        # Assemble block matrix
        K_local = torch.zeros((geom.E, total_dofs, total_dofs), dtype=dtype, device=device)
        K_local[:, 0:u_dofs, 0:u_dofs] = K_uu
        K_local[:, 0:u_dofs, u_dofs:total_dofs] = K_uT
        K_local[:, u_dofs:total_dofs, u_dofs:total_dofs] = K_TT

        M_local = torch.zeros((geom.E, total_dofs, total_dofs), dtype=dtype, device=device)
        M_local[:, 0:u_dofs, 0:u_dofs] = M_uu
        M_local[:, u_dofs:total_dofs, u_dofs:total_dofs] = M_TT

        F_local = torch.zeros((geom.E, total_dofs), dtype=dtype, device=device)
        load_u, load_T = split_coupled_body_load(body_load)
        xyz = quadrature_coordinates(geom)
        f_u = torch.tensor(
            sample_body_load_batch(load_u, xyz, n_comp=geom.dim), dtype=dtype, device=device,
        )
        f_T = torch.tensor(
            sample_body_load_batch(load_T, xyz, n_comp=1), dtype=dtype, device=device,
        )
        F_local[:, 0:u_dofs] += integrate_nodal_source(geom, f_u)
        F_local[:, u_dofs:total_dofs] += integrate_nodal_source(geom, f_T)
        if abs(T0) > 1e-12:
            T0_vec = torch.full((geom.E, T_dofs), fill_value=T0, dtype=dtype, device=device)
            F_local[:, 0:u_dofs] -= torch.einsum('eij,ej->ei', K_uT, T0_vec)
            
        return K_local, M_local, F_local
