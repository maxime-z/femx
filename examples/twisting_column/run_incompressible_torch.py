"""Twisting-column nonlinear solve with PyTorch Neo-Hookean assembly when possible.

Mixed u-p incompressible assembly still has no torch kernel, so that solve stays on
NumPy. For Lagrange Q1 hexes, compressible Neo-Hookean residual and analytic tangent
are evaluated with batched torch kernels (CPU float64, or MPS float32 when available).
Dirichlet BCs and the sparse linear solve remain on SciPy/NumPy. Wall-clock times are
printed against pure-NumPy incompressible and Neo-Hookean runs on the same mesh and angle.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from femx.core.fields import FieldSpec
from femx.core.dofs import DofMap
from femx.core.state import State
from femx.core.mesh import Mesh
from femx.core.assembly import assemble_nonlinear_system
from femx.backends.numpy_backend import solve_linear
from femx.solvers.linear import apply_dirichlet_bcs
from femx.materials.hyperelastic import NeoHookeanMaterial
from femx.formulations.hyperelasticity import HyperelasticFormulation
from examples.twisting_column.problem import (
    add_common_args,
    build_dirichlet_bcs,
    default_n_steps,
    make_geometry,
)
from examples.twisting_column.run_incompressible import run_incompressible
from examples.twisting_column.run_neohookean import run_neohookean


def _select_torch_device(requested: str) -> Tuple[str, object]:
    """Device/dtype for the hybrid Newton assembly.

    ``auto`` and ``cpu`` use float64 on CPU (stable for Newton). ``mps`` uses
    float32 on Apple GPU; prefer CPU for production solves.
    """
    import torch

    if requested == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print(
                "Warning: MPS float32 assembly can be less stable for Newton; "
                "use --device cpu for reliable solves"
            )
            return "mps", torch.float32
        print("MPS requested but unavailable; falling back to CPU float64")
        return "cpu", torch.float64
    return "cpu", torch.float64


def _mps_available() -> bool:
    import torch

    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def _torch_residual_supported(geometry, p: int, discretization: str) -> bool:
    return (
        discretization == "lagrange"
        and p == 1
        and isinstance(geometry, Mesh)
        and geometry.coords.shape[1] == 3
        and geometry.cells.shape[1] == 8
    )


def _spmm_supported(device: str) -> bool:
    """Return True if torch.sparse.mm works for CSR routing on this device."""
    import torch

    if device == "cpu":
        return True
    try:
        crow = torch.tensor([0, 1], dtype=torch.int64, device=device)
        col = torch.tensor([0], dtype=torch.int64, device=device)
        vals = torch.tensor([1.0], dtype=torch.float32, device=device)
        S = torch.sparse_csr_tensor(crow, col, vals, size=(1, 1), device=device)
        x = torch.ones((1, 1), dtype=torch.float32, device=device)
        _ = torch.sparse.mm(S, x)
        return True
    except Exception:
        return False


class HybridTorchNeoNewton:
    """Newton with Torch analytic Neo-Hookean residual+tangent and SciPy solve."""

    def __init__(self, rtol: float = 1e-8, atol: float = 1e-10, max_iter: int = 25):
        self.rtol = rtol
        self.atol = atol
        self.max_iter = max_iter
        self.time_residual = 0.0
        self.time_tangent = 0.0
        self.time_assembly = 0.0
        self.n_residual_evals = 0
        self.n_tangent_evals = 0
        self.n_assembly_evals = 0

    def solve(
        self,
        dof_map: DofMap,
        formulation: HyperelasticFormulation,
        state: State,
        dirichlet_bcs: Dict[int, float],
        residual_fn,
        system_fn=None,
        sync_fn=None,
    ) -> Tuple[State, List[float]]:
        U = state.pack_vector(dof_map)
        for dof, val in dirichlet_bcs.items():
            U[dof] = val
        state.unpack_vector(U, dof_map)

        all_dofs = np.arange(dof_map.n_dofs)
        constrained = (
            np.fromiter(dirichlet_bcs.keys(), dtype=np.intp)
            if dirichlet_bcs
            else np.array([], dtype=np.intp)
        )
        free = np.setdiff1d(all_dofs, constrained)
        history: List[float] = []
        r0 = None

        for _ in range(self.max_iter):
            state.copy_committed_to_trial()

            if system_fn is not None:
                t0 = time.perf_counter()
                K, R = system_fn(state)
                if sync_fn:
                    sync_fn()
                self.time_assembly += time.perf_counter() - t0
                self.n_assembly_evals += 1
                self.n_tangent_evals += 1
                self.n_residual_evals += 1
            else:
                t0 = time.perf_counter()
                K, _ = assemble_nonlinear_system(dof_map, formulation, state)
                if sync_fn:
                    sync_fn()
                self.time_tangent += time.perf_counter() - t0
                self.n_tangent_evals += 1

                t0 = time.perf_counter()
                R = residual_fn(state.pack_vector(dof_map))
                if sync_fn:
                    sync_fn()
                self.time_residual += time.perf_counter() - t0
                self.n_residual_evals += 1

            r_norm = float(np.linalg.norm(R[free]))
            history.append(r_norm)
            if r0 is None:
                r0 = r_norm if r_norm > 0 else 1.0
            tol = self.rtol * r0 + self.atol
            if r_norm <= tol or r_norm <= self.atol:
                state.commit_gauss()
                return state, history

            delta_bcs = {dof: 0.0 for dof in constrained}
            K_eff, neg_R = apply_dirichlet_bcs(K, -R, delta_bcs, preserve_symmetry=False)
            dU = solve_linear(K_eff, neg_R)

            U_committed = U.copy()
            step = 1.0
            accepted = False
            for _ls in range(5):
                U_trial = U_committed + step * dU
                state.restore_nodal(U_trial, dof_map)
                state.copy_committed_to_trial()
                try:
                    t0 = time.perf_counter()
                    R_test = residual_fn(U_trial)
                    if sync_fn:
                        sync_fn()
                    self.time_residual += time.perf_counter() - t0
                    self.n_residual_evals += 1
                    r_test = float(np.linalg.norm(R_test[free]))
                    if r_test < r_norm or step <= 0.125:
                        U = U_trial
                        accepted = True
                        break
                except ValueError:
                    step *= 0.5
                    state.restore_nodal(U_committed, dof_map)
                    continue
                step *= 0.5

            if not accepted:
                U = U_committed + step * dU
                state.restore_nodal(U, dof_map)

        raise RuntimeError(
            f"Hybrid torch Newton failed after {self.max_iter} iterations. "
            f"Final residual: {history[-1]:.4e}"
        )


def _make_torch_assembly(
    mesh: Mesh,
    dof_map: DofMap,
    material: NeoHookeanMaterial,
    device: str,
    dtype,
):
    """Build cached Torch residual and fused system callables."""
    import torch
    from femx.core.tensor_assembly import (
        assemble_nonlinear_system_tensor,
        build_nonlinear_tensor_cache,
    )
    from femx.core.tensor_geometry import evaluate_batched_geometry
    from femx.core.torch_kernels import element_dof_indices, neohookean_residual_global

    use_spmm = _spmm_supported(device)
    cache = None
    if use_spmm:
        try:
            cache = build_nonlinear_tensor_cache(
                dof_map, material, device=device, dtype=dtype,
            )
        except (NotImplementedError, RuntimeError) as exc:
            print(
                f"Note: Torch sparse routing unavailable on {device} ({type(exc).__name__}); "
                "using residual-only Torch + NumPy tangent."
            )
            use_spmm = False

    if cache is None:
        # Residual-only cache (no Stage-II routing)
        geom = evaluate_batched_geometry(mesh, device=device, dtype=dtype)
        elem_dofs = element_dof_indices(mesh, dof_map, "u", device=device)
        lam, mu = material.get_lame_parameters()

        class _ResidualOnly:
            pass

        cache = _ResidualOnly()
        cache.geom = geom
        cache.elem_dofs = elem_dofs
        cache.lambda_ = lam
        cache.mu = mu
        cache.F_e = None

    def sync():
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()

    def residual_fn(U_np: np.ndarray) -> np.ndarray:
        U = torch.tensor(U_np, dtype=dtype, device=device)
        R = neohookean_residual_global(
            cache.geom, U, cache.elem_dofs, cache.lambda_, cache.mu, F_e=cache.F_e,
        )
        return R.detach().cpu().numpy()

    def system_fn(state: State):
        if use_spmm:
            return assemble_nonlinear_system_tensor(
                dof_map,
                HyperelasticFormulation(material=material),
                state,
                cache=cache,
                device=device,
                dtype=dtype,
            )
        # Fallback: Torch residual + NumPy analytic tangent
        K, _ = assemble_nonlinear_system(
            dof_map, HyperelasticFormulation(material=material), state,
        )
        R = residual_fn(state.pack_vector(dof_map))
        return K, R

    return residual_fn, system_fn, sync, use_spmm


def _make_torch_residual(mesh: Mesh, dof_map: DofMap, material: NeoHookeanMaterial, device: str, dtype):
    """Backward-compatible residual-only helper used by micro-benchmarks."""
    residual_fn, _, sync, _ = _make_torch_assembly(mesh, dof_map, material, device, dtype)
    return residual_fn, sync


def run_neohookean_torch_hybrid(args, device: str, dtype) -> dict:
    geometry, kind = make_geometry(args)
    if not _torch_residual_supported(geometry, args.p, kind):
        raise RuntimeError(
            "Torch residual path requires --discretization lagrange --p 1 (Q1 hex)."
        )

    fields = [FieldSpec(name="u", components=3, location="nodes", unknown=True)]
    dof_map = DofMap(fields=fields, geometry=geometry)
    material = NeoHookeanMaterial(rho=1.0, E=1.0e5, nu=0.3)
    formulation = HyperelasticFormulation(material=material)

    n_ent = geometry.n_nodes
    state = State()
    state.initialize_field("u", n_ent, 3)
    residual_fn, system_fn, sync, use_spmm = _make_torch_assembly(
        geometry, dof_map, material, device, dtype,
    )
    if not use_spmm:
        print(
            f"Note: torch.sparse.mm unavailable on {device}; "
            "using Torch residual + NumPy tangent fallback."
        )
    newton = HybridTorchNeoNewton(rtol=1e-8, atol=1e-10, max_iter=25)

    n_steps = args.n_steps if args.n_steps is not None else default_n_steps(args.angle)
    history = [state.values["u"].copy()]
    angle_reached = 0.0
    failed = False
    fail_msg = None
    wall0 = time.perf_counter()

    for step in range(1, n_steps + 1):
        lam = step / float(n_steps)
        bcs = build_dirichlet_bcs(dof_map, geometry, args.angle, load_factor=lam)
        try:
            state, hist = newton.solve(
                dof_map,
                formulation,
                state,
                bcs,
                residual_fn=residual_fn,
                system_fn=system_fn,
                sync_fn=sync,
            )
            angle_reached = args.angle * lam
            history.append(state.values["u"].copy())
            print(
                f"[neohookean-torch] step {step}/{n_steps} "
                f"angle={angle_reached:.2f} deg residual={hist[-1]:.3e}"
            )
        except (RuntimeError, ValueError) as exc:
            failed = True
            fail_msg = str(exc)
            print(
                f"[neohookean-torch] FAILED at step {step}/{n_steps} "
                f"(requested {args.angle} deg, last accepted {angle_reached:.2f} deg): {exc}"
            )
            break

    sync()
    wall = time.perf_counter() - wall0
    return {
        "geometry": geometry,
        "dof_map": dof_map,
        "U": state.pack_vector(dof_map),
        "u_field": state.values["u"].copy(),
        "angle_requested": args.angle,
        "angle_reached": angle_reached,
        "formulation": "neohookean_torch_hybrid",
        "discretization": kind,
        "p": args.p,
        "failed": failed,
        "fail_msg": fail_msg,
        "history": history,
        "n_steps": n_steps,
        "timing": {
            "wall_s": wall,
            "residual_s": newton.time_residual,
            "tangent_s": newton.time_tangent,
            "assembly_s": newton.time_assembly,
            "n_residual_evals": newton.n_residual_evals,
            "n_tangent_evals": newton.n_tangent_evals,
            "n_assembly_evals": newton.n_assembly_evals,
            "device": device,
            "dtype": str(dtype).replace("torch.", ""),
            "spmm": use_spmm,
        },
    }


def _time_numpy_callable(fn) -> Tuple[object, float]:
    t0 = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps"),
        default="auto",
        help="Torch assembly device for hybrid Newton (auto/cpu=float64 CPU; mps=float32 GPU)",
    )
    parser.add_argument(
        "--skip-numpy-incompressible",
        action="store_true",
        default=True,
        help="Skip the NumPy mixed incompressible reference solve",
    )
    parser.add_argument(
        "--skip-numpy-neohookean",
        action="store_true",
        default=True,
        help="Skip the pure-NumPy compressible Neo-Hookean reference solve",
    )
    args = parser.parse_args()
    if args.n_steps is None:
        args.n_steps = default_n_steps(args.angle)

    device, dtype = _select_torch_device(args.device)
    print(f"Torch assembly device={device}, dtype={dtype}")

    timings = {}

    # 1) NumPy mixed incompressible (same as run_incompressible.py)
    if not args.skip_numpy_incompressible:
        print("\n=== NumPy mixed incompressible ===")
        result_inc, t_inc = _time_numpy_callable(lambda: run_incompressible(args))
        timings["numpy_incompressible_wall_s"] = t_inc
        print(f"wall time: {t_inc:.3f} s  reached={result_inc['angle_reached']:.1f} deg  failed={result_inc['failed']}")
    else:
        result_inc = None

    # 2) NumPy compressible Neo-Hookean
    if not args.skip_numpy_neohookean:
        print("\n=== NumPy compressible Neo-Hookean ===")
        result_neo, t_neo = _time_numpy_callable(lambda: run_neohookean(args))
        timings["numpy_neohookean_wall_s"] = t_neo
        print(f"wall time: {t_neo:.3f} s  reached={result_neo['angle_reached']:.1f} deg  failed={result_neo['failed']}")
    else:
        result_neo = None

    # 3) Torch-hybrid compressible Neo-Hookean (analytic residual+tangent on torch)
    result_torch = None
    geometry, kind = make_geometry(args)
    if _torch_residual_supported(geometry, args.p, kind):
        print(f"\n=== Torch-hybrid Neo-Hookean assembly ({device}) ===")
        try:
            result_torch = run_neohookean_torch_hybrid(args, device, dtype)
            tinfo = result_torch["timing"]
            timings["torch_neohookean_wall_s"] = tinfo["wall_s"]
            timings["torch_residual_s"] = tinfo["residual_s"]
            timings["torch_tangent_numpy_s"] = tinfo["tangent_s"]
            timings["torch_assembly_s"] = tinfo["assembly_s"]
            print(
                f"wall time: {tinfo['wall_s']:.3f} s  "
                f"assembly={tinfo['assembly_s']:.3f} s ({tinfo['n_assembly_evals']} evals)  "
                f"residual={tinfo['residual_s']:.3f} s ({tinfo['n_residual_evals']} evals)  "
                f"tangent_fallback={tinfo['tangent_s']:.3f} s"
            )
            print(
                f"reached={result_torch['angle_reached']:.1f} deg  failed={result_torch['failed']}  "
                f"spmm={tinfo['spmm']}"
            )
        except Exception as exc:
            print(f"Torch-hybrid solve failed: {exc}")
    else:
        print(
            "\nTorch residual path skipped "
            "(needs --discretization lagrange --p 1). "
            "Mixed incompressible has no torch kernel yet."
        )

    # Micro-benchmark: residual and fused system NumPy vs torch
    if _torch_residual_supported(geometry, args.p, kind):
        import torch

        fields = [FieldSpec(name="u", components=3, location="nodes", unknown=True)]
        dof_map = DofMap(fields=fields, geometry=geometry)
        material = NeoHookeanMaterial(rho=1.0, E=1.0e5, nu=0.3)
        formulation = HyperelasticFormulation(material=material)
        state = State()
        state.initialize_field("u", geometry.n_nodes, 3)
        for node in range(geometry.n_nodes):
            x, y, z = geometry.coords[node]
            state.values["u"][node] = [0.02 * y, 0.0, -0.01 * y]
        U0 = state.pack_vector(dof_map)

        def numpy_system():
            return assemble_nonlinear_system(dof_map, formulation, state)

        _ = numpy_system()
        reps = 5
        t_np = []
        Kn = Rn = None
        for _ in range(reps):
            t0 = time.perf_counter()
            Kn, Rn = numpy_system()
            t_np.append(time.perf_counter() - t0)
        timings["system_numpy_ms"] = 1e3 * float(np.median(t_np))
        print("\n=== Single system micro-benchmark (median of 5) ===")
        print(f"NumPy K+R:            {timings['system_numpy_ms']:.2f} ms")

        for bench_dev, bench_dtype, label in (
            ("cpu", torch.float64, "Torch CPU f64"),
            *(
                [("mps", torch.float32, "Torch MPS f32")]
                if _mps_available()
                else []
            ),
        ):
            residual_fn, system_fn, sync, use_spmm = _make_torch_assembly(
                geometry, dof_map, material, bench_dev, bench_dtype,
            )
            _ = residual_fn(U0)
            sync()
            t_th = []
            Rt = None
            for _ in range(reps):
                t0 = time.perf_counter()
                Rt = residual_fn(U0)
                sync()
                t_th.append(time.perf_counter() - t0)
            med_r = 1e3 * float(np.median(t_th))
            finite = Rt is not None and np.isfinite(Rt).all() and np.isfinite(Rn).all()
            err_r = float(np.max(np.abs(Rn - Rt))) if finite else float("nan")
            timings[f"residual_torch_{bench_dev}_ms"] = med_r
            timings[f"residual_torch_{bench_dev}_max_abs_err"] = err_r
            print(
                f"{label + ' residual:':24s} {med_r:.2f} ms  max|err|={err_r:.3e}"
            )

            if use_spmm:
                _ = system_fn(state)
                sync()
                t_sys = []
                Kt = None
                for _ in range(reps):
                    t0 = time.perf_counter()
                    Kt, Rt2 = system_fn(state)
                    sync()
                    t_sys.append(time.perf_counter() - t0)
                med_s = 1e3 * float(np.median(t_sys))
                err_k = float(np.max(np.abs(Kn.toarray() - Kt.toarray())))
                err_r2 = float(np.max(np.abs(Rn - Rt2)))
                timings[f"system_torch_{bench_dev}_ms"] = med_s
                timings[f"system_torch_{bench_dev}_max_abs_K_err"] = err_k
                timings[f"system_torch_{bench_dev}_max_abs_R_err"] = err_r2
                print(
                    f"{label + ' K+R:':24s} {med_s:.2f} ms  "
                    f"(speedup {timings['system_numpy_ms']/max(med_s,1e-12):.2f}x)  "
                    f"max|K err|={err_k:.3e}  max|R err|={err_r2:.3e}"
                )
            else:
                print(f"{label + ' K+R:':24s} skipped (no SpMM on device)")

    print("\n=== Timing summary ===")
    for key, val in timings.items():
        if key.endswith("_s"):
            print(f"  {key}: {val:.3f}")
        elif key.endswith("_ms"):
            print(f"  {key}: {val:.2f}")
        else:
            print(f"  {key}: {val}")

    # Prefer torch-hybrid result for visualization when available
    result = result_torch or result_inc or result_neo
    if result is not None:
        out = args.output_dir or os.path.join(
            os.path.dirname(__file__), "output", "incompressible_torch"
        )
        try:
            from examples.twisting_column.visualize import write_all
            write_all(result, out)
        except Exception as exc:
            print(f"Visualization skipped: {exc}")
    return result, timings


if __name__ == "__main__":
    main()
