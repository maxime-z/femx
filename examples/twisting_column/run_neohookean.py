"""Compressible Neo-Hookean twisting column with load stepping."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from femx.core.fields import FieldSpec
from femx.core.dofs import DofMap
from femx.core.state import State
from femx.materials.hyperelastic import NeoHookeanMaterial
from femx.formulations.hyperelasticity import HyperelasticFormulation
from femx.solvers.nonlinear import NewtonSolver
from examples.twisting_column.problem import (
    add_common_args,
    build_dirichlet_bcs,
    default_n_steps,
    make_geometry,
)


def run_neohookean(args):
    geometry, kind = make_geometry(args)
    fields = [FieldSpec(name="u", components=3, location="nodes", unknown=True)]
    dof_map = DofMap(fields=fields, geometry=geometry)
    material = NeoHookeanMaterial(rho=1.0, E=1.0e5, nu=0.3)
    formulation = HyperelasticFormulation(material=material)

    if isinstance(geometry, type(dof_map.geometry)) and hasattr(geometry, "n_nodes"):
        n_ent = geometry.n_nodes
    else:
        n_ent = geometry.n_control_points

    state = State()
    state.initialize_field("u", n_ent, 3)
    newton = NewtonSolver(rtol=1e-8, atol=1e-10, max_iter=25)

    n_steps = args.n_steps if args.n_steps is not None else default_n_steps(args.angle)
    history = [state.values["u"].copy()]
    angle_reached = 0.0
    failed = False
    fail_msg = None

    for step in range(1, n_steps + 1):
        lam = step / float(n_steps)
        bcs = build_dirichlet_bcs(dof_map, geometry, args.angle, load_factor=lam)
        try:
            state, hist = newton.solve(dof_map, formulation, state, bcs)
            angle_reached = args.angle * lam
            history.append(state.values["u"].copy())
            print(f"[neohookean] step {step}/{n_steps} angle={angle_reached:.2f} deg residual={hist[-1]:.3e}")
        except (RuntimeError, ValueError) as exc:
            failed = True
            fail_msg = str(exc)
            print(
                f"[neohookean] FAILED at step {step}/{n_steps} "
                f"(requested {args.angle} deg, last accepted {angle_reached:.2f} deg): {exc}"
            )
            print(
                f"  formulation=compressible_neohookean discretization={kind} "
                f"p={args.p} mesh=({args.nx},{args.ny},{args.nz})"
            )
            break

    U = state.pack_vector(dof_map)
    return {
        "geometry": geometry,
        "dof_map": dof_map,
        "U": U,
        "u_field": state.values["u"].copy(),
        "angle_requested": args.angle,
        "angle_reached": angle_reached,
        "formulation": "neohookean",
        "discretization": kind,
        "p": args.p,
        "failed": failed,
        "fail_msg": fail_msg,
        "history": history,
        "n_steps": n_steps,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()
    result = run_neohookean(args)
    out = args.output_dir or os.path.join(os.path.dirname(__file__), "output", "neohookean")
    try:
        from examples.twisting_column.visualize import write_all
        write_all(result, out)
    except Exception as exc:
        print(f"Visualization skipped: {exc}")
    return result


if __name__ == "__main__":
    main()
