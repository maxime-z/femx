from typing import List, Tuple
from femx.formulations.base import Formulation
from femx.backends.numpy_backend import ndarray, zeros, outer, eye
from femx.materials.thermoelastic import LinearThermoelasticMaterial

class LinearThermoelasticityFormulation(Formulation):
    """
    Coupled formulation for Linear Thermoelasticity (u + T).
    Combines 2D/3D linear elasticity and steady-state thermal conduction.
    """
    def __init__(self, material: LinearThermoelasticMaterial, mode: str = "plane_strain"):
        self.material = material
        self.mode = mode
        self.field_names: List[str] = ["u", "T"]

    def compute_element_matrices(
        self,
        elem_coords: ndarray,
        quadrature_pts: ndarray,
        quadrature_wts: ndarray,
        is_nurbs: bool = False,
        patch = None,
        span_u: int = 0,
        span_v: int = 0,
        body_load = None
    ) -> Tuple[ndarray, ndarray, ndarray]:
        """
        Compute block elemental matrices Ke (12x12), Me (12x12), fe (12x1) for Q1 elements.
        """
        nen = elem_coords.shape[0]  # 4 for Q1 quad, 3 for T1 tri
        u_dofs = 2 * nen
        T_dofs = nen
        total_dofs = u_dofs + T_dofs

        Ke = zeros((total_dofs, total_dofs))
        Me = zeros((total_dofs, total_dofs))
        fe = zeros(total_dofs)

        K_uu = zeros((u_dofs, u_dofs))
        K_uT = zeros((u_dofs, T_dofs))
        K_TT = zeros((T_dofs, T_dofs))

        M_uu = zeros((u_dofs, u_dofs))
        M_TT = zeros((T_dofs, T_dofs))

        f_u = zeros(u_dofs)
        f_T = zeros(T_dofs)

        D_mech = self.material.get_constitutive_matrix(mode=self.mode)  # (3, 3)
        m_th = self.material.get_thermal_coupling_vector(mode=self.mode)  # (3,)
        K_th_matrix = self.material.get_property("K_th") * eye(2)  # (2, 2)
        rho = self.material.get_property("rho")
        C_cap = self.material.get_property("C_cap")
        T0 = self.material.get_property("T0")

        # Parse body_load: can be a tuple/list (body_force_u, heat_source_q) or None
        if body_load is None:
            body_force = zeros(2)
            heat_source = 0.0
        elif isinstance(body_load, (list, tuple)) and len(body_load) == 2:
            body_force = body_load[0] if body_load[0] is not None else zeros(2)
            heat_source = body_load[1] if body_load[1] is not None else 0.0
        else:
            body_force = zeros(2)
            heat_source = 0.0

        for gp, w in zip(quadrature_pts, quadrature_wts):
            if is_nurbs:
                from femx.basis.nurbs import compute_nurbs_mapping
                N, dN_dphys, detJ = compute_nurbs_mapping(gp, patch, span_u, span_v)
            else:
                from femx.basis.lagrange import compute_q1_mapping
                N, dN_dphys, detJ = compute_q1_mapping(gp, elem_coords)

            dV = detJ * w

            # 1. Mechanical B matrix (3, 2*nen)
            B_u = zeros((3, u_dofs))
            for i in range(nen):
                dN_dx = dN_dphys[0, i]
                dN_dy = dN_dphys[1, i]
                B_u[0, 2 * i]     = dN_dx
                B_u[1, 2 * i + 1] = dN_dy
                B_u[2, 2 * i]     = dN_dy
                B_u[2, 2 * i + 1] = dN_dx

            # K_uu = B_u^T @ D_mech @ B_u * dV
            K_uu += (B_u.T @ D_mech @ B_u) * dV

            # K_uT = - B_u^T @ m_th @ N_T * dV
            # B_u.T @ m_th is (2*nen,), N is (nen,) -> outer product is (2*nen, nen)
            K_uT -= outer(B_u.T @ m_th, N) * dV

            # 2. Thermal conductivity K_TT = dN_dphys.T @ K_th @ dN_dphys * dV
            K_TT += (dN_dphys.T @ K_th_matrix @ dN_dphys) * dV

            # 3. Mass matrices
            for i in range(nen):
                f_u[2 * i]     += body_force[0] * N[i] * dV
                f_u[2 * i + 1] += body_force[1] * N[i] * dV
                f_T[i]         += heat_source * N[i] * dV

                for j in range(nen):
                    val_u = rho * N[i] * N[j] * dV
                    M_uu[2 * i, 2 * j]         += val_u
                    M_uu[2 * i + 1, 2 * j + 1] += val_u

                    val_T = rho * C_cap * N[i] * N[j] * dV
                    M_TT[i, j] += val_T

        # Offset for initial reference temperature T0: f_u_initial = - K_uT @ T0_vec
        if abs(T0) > 1e-12:
            T0_vec = T0 * ones(T_dofs)
            f_u -= K_uT @ T0_vec

        # Assemble block matrices
        Ke[0:u_dofs, 0:u_dofs] = K_uu
        Ke[0:u_dofs, u_dofs:total_dofs] = K_uT
        # K_Tu remains zero for uncoupled steady-state thermal reaction
        Ke[u_dofs:total_dofs, u_dofs:total_dofs] = K_TT

        Me[0:u_dofs, 0:u_dofs] = M_uu
        Me[u_dofs:total_dofs, u_dofs:total_dofs] = M_TT

        fe[0:u_dofs] = f_u
        fe[u_dofs:total_dofs] = f_T

        return Ke, Me, fe

    def get_physical_tensors(self, geom, device: str = "cpu", dtype = None):
        """
        Return true physical material tensors for coupled thermoelasticity:
        - C_4th: 4th-order elasticity tensor (E, Q, dim, dim, dim, dim)
        - M_th: 2nd-order thermal coupling tensor (E, Q, dim, dim)
        - K_th: 2nd-order thermal conductivity tensor (E, Q, dim, dim)
        - M_0th: 0th-order capacity scalar (E, Q)
        - T0: reference temperature float
        """
        import torch
        if dtype is None:
            dtype = torch.float64

        C4_np = self.material.get_elasticity_tensor_4th(mode=self.mode, dim=geom.dim)
        C_4th = torch.tensor(C4_np, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0).expand(
            geom.E, geom.Q, geom.dim, geom.dim, geom.dim, geom.dim
        )

        M_th_np = self.material.get_thermal_coupling_tensor(mode=self.mode, dim=geom.dim)
        M_th = torch.tensor(M_th_np, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0).expand(
            geom.E, geom.Q, geom.dim, geom.dim
        )

        K_th_np = self.material.get_property("K_th") * eye(geom.dim)
        K_th = torch.tensor(K_th_np, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0).expand(
            geom.E, geom.Q, geom.dim, geom.dim
        )

        rho = self.material.get_property("rho")
        M_0th = torch.full((geom.E, geom.Q), fill_value=rho, dtype=dtype, device=device)
        T0 = self.material.get_property("T0")

        return C_4th, M_th, K_th, M_0th, T0
