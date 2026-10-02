"""NumPy vs PyTorch timing for Q1 hex elasticity assembly and Neo-Hookean residual."""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from femx.core.fields import FieldSpec
from femx.core.dofs import DofMap
from femx.core.assembly import assemble_system, assemble_nonlinear_system
from femx.core.state import State
from femx.core.tensor_geometry import evaluate_batched_geometry
from femx.core.tensor_assembly import assemble_system_tensor, assemble_nonlinear_system_tensor
from femx.core.torch_kernels import (
    element_dof_indices,
    elasticity_residual_global,
    neohookean_residual_global,
)
from femx.materials.linear_elastic import LinearElasticMaterial
from femx.materials.hyperelastic import NeoHookeanMaterial
from femx.formulations.elasticity import LinearElasticityFormulation
from femx.formulations.hyperelasticity import HyperelasticFormulation
from examples.twisting_column.problem import create_lagrange_column


def _median_time(fn, repeats: int = 3) -> float:
    times = []
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.synchronize()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=4)
    parser.add_argument("--ny", type=int, default=20)
    parser.add_argument("--nz", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    # Correctness on a coarse mesh
    mesh_c = create_lagrange_column(1, 2, 1, p=1)
    fields = [FieldSpec(name="u", components=3, location="nodes", unknown=True)]
    dof_c = DofMap(fields=fields, geometry=mesh_c)
    mat_e = LinearElasticMaterial(rho=1.0, E=1.0e5, nu=0.3)
    form_e = LinearElasticityFormulation(mat_e, mode="3d")
    K_np, _, f_np = assemble_system(dof_c, form_e, field_name="u")
    K_t, _, f_t, _, _ = assemble_system_tensor(dof_c, form_e, field_name="u")
    assert np.allclose(K_t.toarray(), K_np.toarray(), atol=1e-8)
    assert np.allclose(f_t, f_np, atol=1e-8)
    print("elasticity tensor assembly matches NumPy on coarse mesh")

    mat_n = NeoHookeanMaterial(rho=1.0, E=1.0e5, nu=0.3)
    form_n = HyperelasticFormulation(mat_n)
    state = State()
    state.initialize_field("u", mesh_c.n_nodes, 3)
    for node in range(mesh_c.n_nodes):
        x, y, z = mesh_c.coords[node]
        state.values["u"][node] = [0.02 * y, 0.0, -0.01 * y]
    K_np, R_np = assemble_nonlinear_system(dof_c, form_n, state)
    geom = evaluate_batched_geometry(mesh_c, device="cpu", dtype=torch.float64)
    elem_dofs = element_dof_indices(mesh_c, dof_c, "u", device="cpu")
    U = torch.tensor(state.pack_vector(dof_c), dtype=torch.float64)
    lam, mu = mat_n.get_lame_parameters()
    R_t = neohookean_residual_global(geom, U, elem_dofs, lam, mu)
    assert np.allclose(R_t.detach().numpy(), R_np, atol=1e-7)
    print("Neo-Hookean torch residual matches NumPy on coarse mesh")

    K_tt, R_tt = assemble_nonlinear_system_tensor(
        dof_c, form_n, state, device="cpu", dtype=torch.float64,
    )
    assert np.allclose(R_tt, R_np, atol=1e-7)
    assert np.allclose(K_tt.toarray(), K_np.toarray(), atol=1e-5)
    print("Neo-Hookean torch analytic K+R matches NumPy on coarse mesh")

    # Timing on a refined mesh
    mesh = create_lagrange_column(args.nx, args.ny, args.nz, p=1)
    dof = DofMap(fields=fields, geometry=mesh)
    print(f"Timing mesh: {mesh.n_elements} hexes, {dof.n_dofs} dofs")

    def numpy_elasticity():
        assemble_system(dof, form_e, field_name="u")

    def torch_cpu_elasticity():
        assemble_system_tensor(dof, form_e, field_name="u", device="cpu", dtype=torch.float64)

    t_np = _median_time(numpy_elasticity, args.repeats)
    t_cpu = _median_time(torch_cpu_elasticity, args.repeats)
    print(f"elasticity assemble  NumPy float64 : {t_np*1e3:.2f} ms")
    print(f"elasticity assemble  Torch CPU f64 : {t_cpu*1e3:.2f} ms  (speedup {t_np/max(t_cpu,1e-12):.2f}x)")

    state_r = State()
    state_r.initialize_field("u", mesh.n_nodes, 3)
    for node in range(mesh.n_nodes):
        x, y, z = mesh.coords[node]
        state_r.values["u"][node] = [0.01 * y, 0.0, -0.005 * y]

    def numpy_residual():
        assemble_nonlinear_system(dof, form_n, state_r)

    geom_cpu = evaluate_batched_geometry(mesh, device="cpu", dtype=torch.float64)
    ed_cpu = element_dof_indices(mesh, dof, "u", device="cpu")
    U_cpu = torch.tensor(state_r.pack_vector(dof), dtype=torch.float64)

    def torch_cpu_residual():
        neohookean_residual_global(geom_cpu, U_cpu, ed_cpu, lam, mu)

    def torch_cpu_system():
        assemble_nonlinear_system_tensor(
            dof, form_n, state_r, device="cpu", dtype=torch.float64,
        )

    t_r_np = _median_time(numpy_residual, args.repeats)
    t_r_cpu = _median_time(torch_cpu_residual, args.repeats)
    t_sys_cpu = _median_time(torch_cpu_system, args.repeats)
    print(f"neo residual         NumPy float64 : {t_r_np*1e3:.2f} ms")
    print(f"neo residual         Torch CPU f64 : {t_r_cpu*1e3:.2f} ms  (speedup {t_r_np/max(t_r_cpu,1e-12):.2f}x)")
    print(f"neo K+R              Torch CPU f64 : {t_sys_cpu*1e3:.2f} ms  (vs NumPy K+R {t_r_np/max(t_sys_cpu,1e-12):.2f}x)")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        geom_mps = evaluate_batched_geometry(mesh, device="mps", dtype=torch.float32)
        ed_mps = element_dof_indices(mesh, dof, "u", device="mps")
        U_mps = torch.tensor(state_r.pack_vector(dof), dtype=torch.float32, device="mps")
        # Coarse float32 check
        geom_c = evaluate_batched_geometry(mesh_c, device="mps", dtype=torch.float32)
        ed_c = element_dof_indices(mesh_c, dof_c, "u", device="mps")
        U_c = torch.tensor(state.pack_vector(dof_c), dtype=torch.float32, device="mps")
        R_mps = neohookean_residual_global(geom_c, U_c, ed_c, lam, mu)
        err = np.max(np.abs(R_mps.detach().cpu().numpy() - R_np))
        print(f"MPS float32 residual max abs error vs NumPy on coarse mesh: {err:.3e}")

        def torch_mps_residual():
            neohookean_residual_global(geom_mps, U_mps, ed_mps, lam, mu)

        t_r_mps = _median_time(torch_mps_residual, args.repeats)
        print(f"neo residual         Torch MPS f32 : {t_r_mps*1e3:.2f} ms  (speedup vs NumPy {t_r_np/max(t_r_mps,1e-12):.2f}x)")
    else:
        print("MPS not available; skipped Apple GPU timing")


if __name__ == "__main__":
    main()
