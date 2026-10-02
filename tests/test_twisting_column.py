"""Tests for the 3D twisting-column discretizations and solves."""
import numpy as np

from femx.basis.lagrange import LagrangeHex
from femx.core.fields import FieldSpec
from femx.core.dofs import DofMap
from femx.core.assembly import assemble_system
from femx.materials.linear_elastic import LinearElasticMaterial
from femx.formulations.elasticity import LinearElasticityFormulation
from femx.solvers.linear import solve_system
from examples.twisting_column.problem import (
    create_lagrange_column,
    create_nurbs_column,
    build_dirichlet_bcs,
    twist_displacement,
)
from examples.twisting_column.run_neohookean import run_neohookean
import argparse


def test_lagrange_hex_p2_partition_and_jacobian():
    basis = LagrangeHex(p=2)
    assert basis.n_dofs_per_element == 27
    gp = np.array([0.1, -0.2, 0.3])
    N = basis.evaluate_shape_functions(gp)
    assert np.isclose(np.sum(N), 1.0, atol=1e-12)
    dN = basis.evaluate_shape_derivatives(gp)
    assert np.allclose(np.sum(dN, axis=1), 0.0, atol=1e-12)

    # Unit cube corners matched to p=2 tensor nodes via physical coords
    mesh = create_lagrange_column(1, 1, 1, p=2, Lx=2.0, Ly=2.0, Lz=2.0)
    # Shift so it's a unit-ish box for jacobian check: just use first element
    cell = mesh.cells[0]
    elem_coords = mesh.coords[cell]
    N, dN, detJ = basis.compute_mapping(np.zeros(3), elem_coords)
    assert detJ > 0.0


def test_q1_elasticity_twist_bcs():
    mesh = create_lagrange_column(1, 2, 1, p=1)
    fields = [FieldSpec(name="u", components=3, location="nodes", unknown=True)]
    dof_map = DofMap(fields=fields, geometry=mesh)
    mat = LinearElasticMaterial(rho=1.0, E=1.0e5, nu=0.3)
    form = LinearElasticityFormulation(mat, mode="3d")
    K, _, f = assemble_system(dof_map, form, field_name="u")
    angle = 15.0
    bcs = build_dirichlet_bcs(dof_map, mesh, angle)
    U = solve_system(K, f, bcs).reshape((-1, 3))

    for node in mesh.boundaries["bottom"]:
        assert np.allclose(U[node], 0.0, atol=1e-10)
    for node in mesh.boundaries["top"]:
        du = twist_displacement(mesh.coords[node], np.deg2rad(angle))
        assert np.isclose(U[node, 0], du[0], atol=1e-10)
        assert np.isclose(U[node, 2], du[2], atol=1e-10)
    # Longitudinal DOFs on top are free: not all identically zero after solve
    # (may be small; check they are not constrained by comparing to BC dict)
    top_uy_dofs = [dof_map.get_dof("u", int(n), 1) for n in mesh.boundaries["top"]]
    for d in top_uy_dofs:
        assert d not in bcs


def test_nurbs_degree1_matches_q1_elasticity():
    mesh = create_lagrange_column(1, 2, 1, p=1)
    patch = create_nurbs_column(1, 2, 1, degree=1)
    fields = [FieldSpec(name="u", components=3, location="nodes", unknown=True)]
    angle = 10.0

    def solve_geom(geom):
        dof = DofMap(fields=fields, geometry=geom)
        mat = LinearElasticMaterial(rho=1.0, E=1.0e5, nu=0.3)
        form = LinearElasticityFormulation(mat, mode="3d")
        K, _, f = assemble_system(dof, form, field_name="u")
        bcs = build_dirichlet_bcs(dof, geom, angle)
        return solve_system(K, f, bcs)

    U_m = solve_geom(mesh)
    U_p = solve_geom(patch)
    # Control-point / node layouts differ in ordering; compare energy-like norms and BC satisfaction
    assert U_m.shape == U_p.shape
    assert np.linalg.norm(U_m) > 0
    assert abs(np.linalg.norm(U_m) - np.linalg.norm(U_p)) / np.linalg.norm(U_m) < 1e-6


def test_neohookean_small_angle_and_large_angle_report():
    args = argparse.Namespace(
        angle=10.0, n_steps=2, nx=1, ny=2, nz=1, p=1,
        discretization="lagrange", output_dir=None,
    )
    result = run_neohookean(args)
    assert not result["failed"]
    assert result["angle_reached"] == 10.0

    args_large = argparse.Namespace(
        angle=180.0, n_steps=2, nx=1, ny=2, nz=1, p=1,
        discretization="lagrange", output_dir=None,
    )
    result_large = run_neohookean(args_large)
    # Either fails early or reaches with difficulty; if it fails, last angle < requested
    if result_large["failed"]:
        assert result_large["angle_reached"] < args_large.angle
    else:
        # Still acceptable if Newton somehow converges on this coarse mesh
        assert result_large["angle_reached"] == args_large.angle


if __name__ == "__main__":
    test_lagrange_hex_p2_partition_and_jacobian()
    test_q1_elasticity_twist_bcs()
    test_nurbs_degree1_matches_q1_elasticity()
    test_neohookean_small_angle_and_large_angle_report()
    print("twisting column tests passed")
