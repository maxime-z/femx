"""Constant-strain elasticity on a bilinear NURBS patch vs Q1 mesh."""
import numpy as np
from femx.backends.numpy_backend import array
from femx.core.mesh import Mesh
from femx.core.fields import FieldSpec
from femx.core.dofs import DofMap
from femx.core.assembly import assemble_system
from femx.materials.linear_elastic import LinearElasticMaterial
from femx.formulations.elasticity import LinearElasticityFormulation
from femx.solvers.linear import solve_system
from femx.geometry.nurbs import KnotVector, NurbsPatch


def _exact_disp(x, y, eps_xx0, eps_yy0, gam_xy0):
    ux = eps_xx0 * x + 0.5 * gam_xy0 * y
    uy = eps_yy0 * y + 0.5 * gam_xy0 * x
    return ux, uy


def test_elasticity_nurbs_matches_q1_patch():
    """
    Degree-1 NURBS unit square (one element) must match a single Q1 quad
    for a constant-strain Dirichlet problem.
    """
    eps_xx0 = 1.0e-3
    eps_yy0 = -0.5e-3
    gam_xy0 = 2.0e-3

    # Q1 mesh: one unit square
    coords = array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ])
    cells = array([[0, 1, 2, 3]])
    mesh = Mesh(coords=coords, cells=cells)

    # Bilinear NURBS patch of the unit square
    knots = KnotVector([0.0, 0.0, 1.0, 1.0])
    control_points = array([
        [[0.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.0], [1.0, 1.0]],
    ])
    weights = np.ones((2, 2))
    patch = NurbsPatch(
        p_u=1, p_v=1,
        knots_u=knots.knots, knots_v=knots.knots,
        control_points=control_points,
        weights=weights,
    )

    material = LinearElasticMaterial(rho=1.0, E=2.0e11, nu=0.3)
    formulation = LinearElasticityFormulation(material=material, mode="plane_strain")

    fields_q1 = [FieldSpec(name="u", components=2, location="nodes", unknown=True)]
    dof_q1 = DofMap(fields=fields_q1, geometry=mesh)
    K_q1, _, f_q1 = assemble_system(dof_q1, formulation, field_name="u")

    fields_iga = [FieldSpec(name="u", components=2, location="control_points", unknown=True)]
    dof_iga = DofMap(fields=fields_iga, geometry=patch)
    K_iga, _, f_iga = assemble_system(dof_iga, formulation, field_name="u")

    # Prescribe exact linear field on all DOFs (one-element, all-boundary)
    bcs_q1 = {}
    bcs_iga = {}
    # Q1 node order: 0,1,2,3. NURBS flat CP order (v-major after transpose flatten):
    # control_points[u,v] flattened as transpose(1,0).reshape -> v varies outer:
    # flat index = u + v * n_cp_u with n_cp_u=2
    # (u,v)=(0,0)->0, (1,0)->1, (0,1)->2, (1,1)->3
    # physical: (0,0), (1,0), (0,1), (1,1)
    # Q1 nodes: (0,0), (1,0), (1,1), (0,1) = indices 0,1,2,3
    # Mapping IGA flat -> Q1 node: 0->0, 1->1, 2->3, 3->2
    iga_to_q1 = [0, 1, 3, 2]

    for node in range(4):
        x, y = coords[node]
        ux, uy = _exact_disp(x, y, eps_xx0, eps_yy0, gam_xy0)
        bcs_q1[dof_q1.get_dof("u", node, 0)] = ux
        bcs_q1[dof_q1.get_dof("u", node, 1)] = uy

    for iga_idx, q1_idx in enumerate(iga_to_q1):
        x, y = coords[q1_idx]
        ux, uy = _exact_disp(x, y, eps_xx0, eps_yy0, gam_xy0)
        bcs_iga[dof_iga.get_dof("u", iga_idx, 0)] = ux
        bcs_iga[dof_iga.get_dof("u", iga_idx, 1)] = uy

    U_q1 = solve_system(K_q1, f_q1, bcs_q1)
    U_iga = solve_system(K_iga, f_iga, bcs_iga)

    # Reorder IGA solution to Q1 node order for comparison
    U_iga_as_q1 = np.zeros_like(U_q1)
    for iga_idx, q1_idx in enumerate(iga_to_q1):
        U_iga_as_q1[2 * q1_idx] = U_iga[2 * iga_idx]
        U_iga_as_q1[2 * q1_idx + 1] = U_iga[2 * iga_idx + 1]

    assert np.allclose(U_q1, U_iga_as_q1, atol=1e-10)

    # Stiffness matrices should match after the same DOF permutation
    # Build permutation P such that U_q1 = P @ U_iga
    P = np.zeros((8, 8))
    for iga_idx, q1_idx in enumerate(iga_to_q1):
        P[2 * q1_idx, 2 * iga_idx] = 1.0
        P[2 * q1_idx + 1, 2 * iga_idx + 1] = 1.0
    K_iga_q1 = P @ K_iga.toarray() @ P.T
    assert np.allclose(K_q1.toarray(), K_iga_q1, rtol=1e-10, atol=1e-6)
