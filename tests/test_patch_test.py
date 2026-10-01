import numpy as np
from femx.backends.numpy_backend import array
from femx.core.mesh import Mesh
from femx.core.fields import FieldSpec
from femx.core.dofs import DofMap
from femx.core.assembly import assemble_system
from femx.materials.linear_elastic import LinearElasticMaterial
from femx.formulations.elasticity import LinearElasticityFormulation
from femx.solvers.linear import solve_system
from femx.core.postprocessing import compute_element_stresses
from femx.basis.lagrange import LagrangeHex
from femx.core.quadrature import get_quadrature_3d


def test_constant_strain_patch_test():
    """
    Constant strain patch test for 2D plane strain linear elasticity.
    Applies exact linear displacement field to boundary nodes of a 4-element mesh.
    Verifies that interior node displacements match exact linear field and
    all element Gauss point strains and stresses match exact values.
    """
    coords = array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [0.0, 1.0],
        [0.9, 1.1],
        [2.0, 1.0],
        [0.0, 2.0],
        [1.0, 2.0],
        [2.0, 2.0]
    ])

    cells = array([
        [0, 1, 4, 3],
        [1, 2, 5, 4],
        [3, 4, 7, 6],
        [4, 5, 8, 7]
    ])

    mesh = Mesh(coords=coords, cells=cells)

    eps_xx0 = 1.0e-3
    eps_yy0 = -0.5e-3
    gam_xy0 = 2.0e-3

    def exact_disp(x, y):
        ux = eps_xx0 * x + 0.5 * gam_xy0 * y
        uy = eps_yy0 * y + 0.5 * gam_xy0 * x
        return ux, uy

    fields = [FieldSpec(name="u", components=2, location="nodes", unknown=True)]
    dof_map = DofMap(fields=fields, geometry=mesh)

    material = LinearElasticMaterial(rho=7800.0, E=2.0e11, nu=0.3)
    formulation = LinearElasticityFormulation(material=material, mode="plane_strain")

    K, M, f = assemble_system(dof_map, formulation, field_name="u")

    boundary_nodes = [0, 1, 2, 3, 5, 6, 7, 8]
    dirichlet_bcs = {}

    for node in boundary_nodes:
        x, y = coords[node]
        ux_exact, uy_exact = exact_disp(x, y)
        dirichlet_bcs[dof_map.get_dof("u", node, 0)] = ux_exact
        dirichlet_bcs[dof_map.get_dof("u", node, 1)] = uy_exact

    U = solve_system(K, f, dirichlet_bcs)

    x4, y4 = coords[4]
    ux4_exact, uy4_exact = exact_disp(x4, y4)
    assert np.isclose(U[dof_map.get_dof("u", 4, 0)], ux4_exact, atol=1e-12)
    assert np.isclose(U[dof_map.get_dof("u", 4, 1)], uy4_exact, atol=1e-12)

    result = compute_element_stresses(mesh, dof_map, formulation, U)

    expected_eps = array([eps_xx0, eps_yy0, gam_xy0])
    for elem_idx in range(mesh.n_elements):
        for q_idx in range(4):
            assert np.allclose(result.gauss_strains[elem_idx, q_idx], expected_eps, atol=1e-10)

    D = material.get_constitutive_matrix(mode="plane_strain")
    expected_sig = D @ expected_eps
    for elem_idx in range(mesh.n_elements):
        for q_idx in range(4):
            assert np.allclose(result.gauss_stresses[elem_idx, q_idx], expected_sig, atol=1e-8)


def test_constant_strain_hex_patch_test():
    """Constant strain patch test for one trilinear hex with u=(a x, b y, c z)."""
    a, b, c = 1.0e-3, -0.5e-3, 0.25e-3
    coords = array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ])
    cells = array([[0, 1, 2, 3, 4, 5, 6, 7]])
    mesh = Mesh(coords=coords, cells=cells)

    fields = [FieldSpec(name="u", components=3, location="nodes", unknown=True)]
    dof_map = DofMap(fields=fields, geometry=mesh)
    material = LinearElasticMaterial(rho=1.0, E=210.0e9, nu=0.3)
    formulation = LinearElasticityFormulation(material=material, mode="3d")

    K, M, f = assemble_system(dof_map, formulation, field_name="u")
    assert K.shape == (24, 24)

    def exact_disp(x, y, z):
        return a * x, b * y, c * z

    dirichlet_bcs = {}
    U_exact = np.zeros(dof_map.n_dofs)
    for node in range(mesh.n_nodes):
        x, y, z = coords[node]
        ux, uy, uz = exact_disp(x, y, z)
        for comp, val in enumerate((ux, uy, uz)):
            dof = dof_map.get_dof("u", node, comp)
            dirichlet_bcs[dof] = val
            U_exact[dof] = val

    U = solve_system(K, f, dirichlet_bcs)
    assert np.allclose(U, U_exact, atol=1e-12)

    basis = LagrangeHex(p=1)
    quad_pts, _ = get_quadrature_3d(2, 2, 2)
    D = material.get_constitutive_matrix(mode="3d")
    expected_eps = array([a, b, c, 0.0, 0.0, 0.0])
    expected_sig = D @ expected_eps

    elem_coords = coords[cells[0]]
    U_nodes = U_exact.reshape(8, 3)
    for gp in quad_pts:
        _, dN_dphys, detJ = basis.compute_mapping(gp, elem_coords)
        assert detJ > 0.0
        G = (dN_dphys @ U_nodes).T
        eps = array([
            G[0, 0],
            G[1, 1],
            G[2, 2],
            G[1, 2] + G[2, 1],
            G[0, 2] + G[2, 0],
            G[0, 1] + G[1, 0],
        ])
        assert np.allclose(eps, expected_eps, atol=1e-12)
        assert np.allclose(D @ eps, expected_sig, atol=1e-6)


if __name__ == "__main__":
    test_constant_strain_patch_test()
    test_constant_strain_hex_patch_test()
    print("Constant strain patch tests passed successfully!")
