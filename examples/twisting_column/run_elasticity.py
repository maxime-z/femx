"""Linear-elastic twisting column."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Allow running as a script from the repo root or this folder.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from femx.core.fields import FieldSpec
from femx.core.dofs import DofMap
from femx.core.assembly import assemble_system
from femx.materials.linear_elastic import LinearElasticMaterial
from femx.formulations.elasticity import LinearElasticityFormulation
from femx.solvers.linear import solve_system
from examples.twisting_column.problem import (
    add_common_args,
    build_dirichlet_bcs,
    default_n_steps,
    make_geometry,
)


def run_elasticity(args):
    geometry, kind = make_geometry(args)
    fields = [FieldSpec(name="u", components=3, location="nodes", unknown=True)]
    dof_map = DofMap(fields=fields, geometry=geometry)
    material = LinearElasticMaterial(rho=1.0, E=1.0e5, nu=0.3)
    formulation = LinearElasticityFormulation(material=material, mode="3d")
    K, _, f = assemble_system(dof_map, formulation, field_name="u")
    bcs = build_dirichlet_bcs(dof_map, geometry, args.angle, load_factor=1.0)
    U = solve_system(K, f, bcs)

    invalid = abs(args.angle) > 30.0
    if invalid:
        print(
            f"[elasticity] angle={args.angle} deg exceeds ~30 deg: "
            "solution is kinematically invalid for finite rotation (linear strain)."
        )
    else:
        print(f"[elasticity] solved at angle={args.angle} deg ({kind}, p={args.p})")

    u_field = U.reshape((-1, 3))
    return {
        "geometry": geometry,
        "dof_map": dof_map,
        "U": U,
        "u_field": u_field,
        "angle_requested": args.angle,
        "angle_reached": args.angle,
        "formulation": "elasticity",
        "discretization": kind,
        "p": args.p,
        "failed": False,
        "kinematically_invalid": invalid,
        "history": [u_field.copy()],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()
    if args.n_steps is None:
        args.n_steps = default_n_steps(args.angle)
    result = run_elasticity(args)
    out = args.output_dir or os.path.join(os.path.dirname(__file__), "output", "elasticity")
    try:
        from examples.twisting_column.visualize import write_all
        write_all(result, out)
    except Exception as exc:
        print(f"Visualization skipped: {exc}")
    return result


if __name__ == "__main__":
    main()
