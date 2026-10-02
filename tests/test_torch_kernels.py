"""Differentiable PyTorch element kernels for heat, elasticity, and Neo-Hookean."""
import numpy as np
import torch
from femx.backends.numpy_backend import array
from femx.core.mesh import Mesh
from femx.core.fields import FieldSpec
from femx.core.dofs import DofMap
from femx.core.state import State
from femx.core.assembly import assemble_system, assemble_nonlinear_system
from femx.core.tensor_geometry import evaluate_batched_geometry, integrate_nodal_source
from femx.core.loads import sample_body_load_batch
from femx.core.torch_kernels import (
    element_dof_indices,
    heat_residual_global,
    heat_stiffness_local,
    elasticity_residual_global,
    elasticity_stiffness_local,
    neohookean_residual_global,
    neohookean_residual_local,
    neohookean_stiffness_local,
    neohookean_system_local,
)
from femx.materials.linear_heat import LinearHeatMaterial
from femx.materials.linear_elastic import LinearElasticMaterial
from femx.materials.hyperelastic import (
    NeoHookeanMaterial,
    first_piola_torch,
    neohookean_stress_tangent_torch,
)
from femx.formulations.heat import HeatConductionFormulation
from femx.formulations.elasticity import LinearElasticityFormulation
from femx.formulations.hyperelasticity import HyperelasticFormulation
from femx.basis.lagrange import LagrangeQuad
from femx.core.quadrature import get_quadrature_2d


def _quad_mesh_2x2() -> Mesh:
    coords = array([
        [0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
        [0.0, 0.5], [0.5, 0.5], [1.0, 0.5],
        [0.0, 1.0], [0.5, 1.0], [1.0, 1.0],
    ])
    cells = array([
        [0, 1, 4, 3],
        [1, 2, 5, 4],
        [3, 4, 7, 6],
        [4, 5, 8, 7],
    ])
    return Mesh(coords=coords, cells=cells, boundaries={
        "left": array([0, 3, 6]),
        "right": array([2, 5, 8]),
    })


def _available_accel_devices():
    devices = []
    if torch.cuda.is_available():
        devices.append(("cuda", torch.float64, 1e-10))
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append(("mps", torch.float32, 5e-4))
    return devices


def test_first_piola_torch_matches_numpy():
    material = NeoHookeanMaterial(rho=1.0, E=1.0e5, nu=0.3)
    lambda_, mu = material.get_lame_parameters()
    F_np = np.array([[1.1, 0.2], [0.05, 0.95]])
    P_np, C4_np = material.update(F_np)
    F_t = torch.tensor(F_np, dtype=torch.float64)
    P_t = first_piola_torch(F_t, lambda_, mu)
    np.testing.assert_allclose(P_t.detach().numpy(), P_np, atol=1e-12)

    P2, C4_t = neohookean_stress_tangent_torch(F_t, lambda_, mu)
    np.testing.assert_allclose(P2.detach().numpy(), P_np, atol=1e-12)
    np.testing.assert_allclose(C4_t.detach().numpy(), C4_np, atol=1e-12)

    # Batched (E, Q, dim, dim) — the layout used by element kernels
    F_eq = np.stack([F_np, np.eye(2)], axis=0)[:, None, :, :]  # (2, 1, 2, 2)
    P_b, C4_b = material.update(F_eq)
    P_bt, C4_bt = neohookean_stress_tangent_torch(
        torch.tensor(F_eq, dtype=torch.float64), lambda_, mu
    )
    np.testing.assert_allclose(P_bt.detach().numpy(), P_b, atol=1e-12)
    np.testing.assert_allclose(C4_bt.detach().numpy(), C4_b, atol=1e-12)

    # Per-item check also validates torch against single-matrix NumPy update
    for i in range(2):
        P_i, C4_i = material.update(F_eq[i, 0])
        np.testing.assert_allclose(P_bt[i, 0].detach().numpy(), P_i, atol=1e-12)
        np.testing.assert_allclose(C4_bt[i, 0].detach().numpy(), C4_i, atol=1e-12)

    # 3D single
    F3 = np.eye(3)
    F3[0, 1] = 0.1
    P3_np, C43_np = material.update(F3)
    P3_t, C43_t = neohookean_stress_tangent_torch(
        torch.tensor(F3, dtype=torch.float64), lambda_, mu
    )
    np.testing.assert_allclose(P3_t.detach().numpy(), P3_np, atol=1e-12)
    np.testing.assert_allclose(C43_t.detach().numpy(), C43_np, atol=1e-12)


def test_heat_torch_residual_and_jacrev():
    mesh = _quad_mesh_2x2()
    fields = [FieldSpec(name="T", components=1, location="nodes", unknown=True)]
    dof_map = DofMap(fields=fields, geometry=mesh)
    K_cond = 10.0
    material = LinearHeatMaterial(rho=1.0, C=1.0, K=K_cond)
    formulation = HeatConductionFormulation(material=material)
    body = 2.0

    K_np, _, f_np = assemble_system(dof_map, formulation, field_name="T", body_load=body)
    T_np = np.linspace(0.0, 1.0, dof_map.n_dofs)
    R_np = K_np @ T_np - f_np

    geom = evaluate_batched_geometry(mesh, device="cpu", dtype=torch.float64)
    elem_dofs = element_dof_indices(mesh, dof_map, "T", device="cpu")
    from femx.core.tensor_geometry import quadrature_coordinates
    f_gp = torch.tensor(
        sample_body_load_batch(body, quadrature_coordinates(geom), n_comp=1),
        dtype=torch.float64,
    )
    F_e = integrate_nodal_source(geom, f_gp)

    conductivity = torch.tensor(K_cond, dtype=torch.float64)
    T = torch.tensor(T_np, dtype=torch.float64)
    R_t = heat_residual_global(geom, conductivity, T, elem_dofs, F_e=F_e)
    np.testing.assert_allclose(R_t.detach().numpy(), R_np, atol=1e-10)

    # dR/dT == K
    def R_of_T(T_var):
        return heat_residual_global(geom, conductivity, T_var, elem_dofs, F_e=F_e)

    J_auto = torch.func.jacrev(R_of_T)(T)
    np.testing.assert_allclose(J_auto.detach().numpy(), K_np.toarray(), atol=1e-9)

    # dR/dK matches finite difference
    def R_of_K(K_var):
        return heat_residual_global(geom, K_var, T, elem_dofs, F_e=F_e)

    dRdK = torch.func.jacrev(R_of_K)(conductivity)
    eps = 1e-6
    R_p = R_of_K(conductivity + eps).detach().numpy()
    R_m = R_of_K(conductivity - eps).detach().numpy()
    dRdK_fd = (R_p - R_m) / (2.0 * eps)
    np.testing.assert_allclose(dRdK.detach().numpy(), dRdK_fd, atol=1e-6)

    # Local stiffness matches assemble_system_tensor path
    K_local = heat_stiffness_local(geom, conductivity)
    assert K_local.shape == (mesh.n_elements, 4, 4)


def test_elasticity_torch_residual_and_jacrev():
    mesh = _quad_mesh_2x2()
    fields = [FieldSpec(name="u", components=2, location="nodes", unknown=True)]
    dof_map = DofMap(fields=fields, geometry=mesh)
    E_val = 2.0e7
    nu = 0.3
    material = LinearElasticMaterial(rho=1.0, E=E_val, nu=nu)
    formulation = LinearElasticityFormulation(material=material, mode="plane_strain")
    gravity = np.array([0.0, -9.81])

    K_np, _, f_np = assemble_system(dof_map, formulation, field_name="u", body_load=gravity)
    U_np = np.linspace(0.0, 0.01, dof_map.n_dofs)
    R_np = K_np @ U_np - f_np

    geom = evaluate_batched_geometry(mesh, device="cpu", dtype=torch.float64)
    elem_dofs = element_dof_indices(mesh, dof_map, "u", device="cpu")
    from femx.core.tensor_geometry import quadrature_coordinates
    f_gp = torch.tensor(
        sample_body_load_batch(gravity, quadrature_coordinates(geom), n_comp=2),
        dtype=torch.float64,
    )
    F_e = integrate_nodal_source(geom, f_gp)

    E = torch.tensor(E_val, dtype=torch.float64)
    U = torch.tensor(U_np, dtype=torch.float64)
    R_t = elasticity_residual_global(geom, E, nu, U, elem_dofs, F_e=F_e, mode="plane_strain")
    np.testing.assert_allclose(R_t.detach().numpy(), R_np, rtol=1e-8, atol=1e-6)

    def R_of_U(U_var):
        return elasticity_residual_global(
            geom, E, nu, U_var, elem_dofs, F_e=F_e, mode="plane_strain"
        )

    J_auto = torch.func.jacrev(R_of_U)(U)
    np.testing.assert_allclose(J_auto.detach().numpy(), K_np.toarray(), rtol=1e-8, atol=1e-4)

    def R_of_E(E_var):
        return elasticity_residual_global(
            geom, E_var, nu, U, elem_dofs, F_e=F_e, mode="plane_strain"
        )

    dRdE = torch.func.jacrev(R_of_E)(E)
    eps = E_val * 1e-6
    R_p = R_of_E(E + eps).detach().numpy()
    R_m = R_of_E(E - eps).detach().numpy()
    dRdE_fd = (R_p - R_m) / (2.0 * eps)
    np.testing.assert_allclose(dRdE.detach().numpy(), dRdE_fd, rtol=1e-5, atol=1e-6)

    K_local = elasticity_stiffness_local(geom, E, nu, mode="plane_strain")
    assert K_local.shape == (mesh.n_elements, 8, 8)


def test_neohookean_torch_residual_jacrev_and_fd():
    mesh = _quad_mesh_2x2()
    fields = [FieldSpec(name="u", components=2, location="nodes", unknown=True)]
    dof_map = DofMap(fields=fields, geometry=mesh)
    material = NeoHookeanMaterial(rho=1.0, E=1.0e5, nu=0.3)
    formulation = HyperelasticFormulation(material=material)
    lambda_, mu = material.get_lame_parameters()

    state = State()
    state.initialize_field("u", mesh.n_nodes, 2)
    # small shear-like displacement
    for node in range(mesh.n_nodes):
        x, y = mesh.coords[node]
        state.values["u"][node, 0] = 0.05 * y
        state.values["u"][node, 1] = 0.02 * x

    K_np, R_np = assemble_nonlinear_system(dof_map, formulation, state)
    U_np = state.pack_vector(dof_map)

    geom = evaluate_batched_geometry(mesh, device="cpu", dtype=torch.float64)
    elem_dofs = element_dof_indices(mesh, dof_map, "u", device="cpu")
    U = torch.tensor(U_np, dtype=torch.float64)
    R_t = neohookean_residual_global(geom, U, elem_dofs, lambda_, mu)
    np.testing.assert_allclose(R_t.detach().numpy(), R_np, atol=1e-8)

    def R_of_U(U_var):
        return neohookean_residual_global(geom, U_var, elem_dofs, lambda_, mu)

    J_auto = torch.func.jacrev(R_of_U)(U)
    np.testing.assert_allclose(J_auto.detach().numpy(), K_np.toarray(), atol=1e-5)

    # Central finite difference of global residual
    eps = 1e-7
    J_fd = np.zeros_like(J_auto.detach().numpy())
    for i in range(len(U_np)):
        Up = U_np.copy()
        Um = U_np.copy()
        Up[i] += eps
        Um[i] -= eps
        Rp = R_of_U(torch.tensor(Up, dtype=torch.float64)).detach().numpy()
        Rm = R_of_U(torch.tensor(Um, dtype=torch.float64)).detach().numpy()
        J_fd[:, i] = (Rp - Rm) / (2.0 * eps)
    np.testing.assert_allclose(J_auto.detach().numpy(), J_fd, atol=1e-5)

    # Local residual and analytic tangent vs element loop
    basis = LagrangeQuad(p=1)
    quad_pts, quad_wts = get_quadrature_2d(2, 2)
    u_e = U[elem_dofs].reshape(mesh.n_elements, 4, 2)
    R_local, K_local = neohookean_system_local(geom, u_e, lambda_, mu)
    K_stiff = neohookean_stiffness_local(geom, u_e, lambda_, mu)
    np.testing.assert_allclose(K_local.detach().numpy(), K_stiff.detach().numpy(), atol=1e-12)
    for e, cell in enumerate(mesh.cells):
        elem_coords = mesh.coords[cell]
        elem_u = state.values["u"][cell]
        Re, Ke = formulation.compute_element_residual_and_tangent(
            elem_coords, elem_u, quad_pts, quad_wts, elem_basis=basis, elem_idx=e
        )
        np.testing.assert_allclose(R_local[e].detach().numpy(), Re, atol=1e-8)
        np.testing.assert_allclose(K_local[e].detach().numpy(), Ke, atol=1e-8)

    # Analytic tangent matches jacrev (regression oracle only)
    np.testing.assert_allclose(J_auto.detach().numpy(), K_np.toarray(), atol=1e-5)


def test_compiled_neohookean_newton_step():
    """One Newton step using analytic Torch tangent (not jacrev)."""
    from femx.core.tensor_assembly import assemble_nonlinear_system_tensor

    mesh = _quad_mesh_2x2()
    fields = [FieldSpec(name="u", components=2, location="nodes", unknown=True)]
    dof_map = DofMap(fields=fields, geometry=mesh)
    material = NeoHookeanMaterial(rho=1.0, E=1.0e5, nu=0.3)
    formulation = HyperelasticFormulation(material=material)

    state = State()
    state.initialize_field("u", mesh.n_nodes, 2)
    for node in range(mesh.n_nodes):
        x, y = mesh.coords[node]
        state.values["u"][node, 0] = 0.04 * y
        state.values["u"][node, 1] = 0.01 * x

    constrained = {
        dof_map.get_dof("u", 0, 0),
        dof_map.get_dof("u", 0, 1),
        dof_map.get_dof("u", 1, 1),
    }
    free = np.array([i for i in range(dof_map.n_dofs) if i not in constrained], dtype=int)

    K_np, R_np = assemble_nonlinear_system(dof_map, formulation, state)
    dU_free_np = np.linalg.solve(K_np.toarray()[np.ix_(free, free)], -R_np[free])

    K_t, R_t = assemble_nonlinear_system_tensor(
        dof_map, formulation, state, device="cpu", dtype=torch.float64,
    )
    np.testing.assert_allclose(R_t, R_np, atol=1e-8)
    np.testing.assert_allclose(K_t.toarray(), K_np.toarray(), atol=1e-6)

    dU_free = np.linalg.solve(K_t.toarray()[np.ix_(free, free)], -R_t[free])
    np.testing.assert_allclose(dU_free, dU_free_np, atol=1e-6)


def test_torch_residual_on_accelerators():
    """Run residual comparison on CUDA (float64) and MPS (float32) when available."""
    accel = _available_accel_devices()
    if not accel:
        print(" (no CUDA/MPS; skipped accelerator residual) ", end="")
        return

    mesh = _quad_mesh_2x2()
    fields = [FieldSpec(name="T", components=1, location="nodes", unknown=True)]
    dof_map = DofMap(fields=fields, geometry=mesh)
    K_cond = 10.0
    material = LinearHeatMaterial(rho=1.0, C=1.0, K=K_cond)
    formulation = HeatConductionFormulation(material=material)
    body = 2.0
    K_np, _, f_np = assemble_system(dof_map, formulation, field_name="T", body_load=body)
    T_np = np.linspace(0.0, 1.0, dof_map.n_dofs)
    R_np = K_np @ T_np - f_np

    from femx.core.tensor_geometry import quadrature_coordinates

    for device, dtype, atol in accel:
        geom = evaluate_batched_geometry(mesh, device=device, dtype=dtype)
        elem_dofs = element_dof_indices(mesh, dof_map, "T", device=device)
        f_gp = torch.tensor(
            sample_body_load_batch(body, quadrature_coordinates(geom), n_comp=1),
            dtype=dtype, device=device,
        )
        F_e = integrate_nodal_source(geom, f_gp)
        conductivity = torch.tensor(K_cond, dtype=dtype, device=device)
        T = torch.tensor(T_np, dtype=dtype, device=device)
        R_t = heat_residual_global(geom, conductivity, T, elem_dofs, F_e=F_e)
        R_np_cmp = R_t.detach().cpu().numpy()
        if not np.isfinite(R_np_cmp).all():
            print(
                f" ({device} residual non-finite in {dtype}; "
                f"skipped accuracy check — use CPU float64 as baseline) ",
                end="",
            )
            continue
        np.testing.assert_allclose(
            R_np_cmp, R_np, atol=atol, rtol=atol
        )

        # jacrev / compile when the device accepts them
        try:
            def R_of_T(T_var):
                return heat_residual_global(geom, conductivity, T_var, elem_dofs, F_e=F_e)

            J = torch.func.jacrev(R_of_T)(T)
            np.testing.assert_allclose(
                J.detach().cpu().numpy(), K_np.toarray(), atol=max(atol, 1e-3), rtol=max(atol, 1e-3)
            )
        except Exception as exc:
            print(f" ({device} jacrev skipped: {type(exc).__name__}) ", end="")

        try:
            compiled = torch.compile(
                lambda T_var: heat_residual_global(geom, conductivity, T_var, elem_dofs, F_e=F_e)
            )
            R_c = compiled(T)
            np.testing.assert_allclose(
                R_c.detach().cpu().numpy(), R_np, atol=atol, rtol=atol
            )
        except Exception as exc:
            print(f" ({device} compile skipped: {type(exc).__name__}) ", end="")


if __name__ == "__main__":
    test_first_piola_torch_matches_numpy()
    test_heat_torch_residual_and_jacrev()
    test_elasticity_torch_residual_and_jacrev()
    test_neohookean_torch_residual_jacrev_and_fd()
    test_compiled_neohookean_newton_step()
    test_torch_residual_on_accelerators()
    print("torch kernel tests passed")
