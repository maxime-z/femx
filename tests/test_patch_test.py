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

def test_constant_strain_patch_test():
    """
    Constant strain patch test for 2D plane strain linear elasticity.
    Applies exact linear displacement field to boundary nodes of a 4-element mesh.
    Verifies that interior node displacements match exact linear field and
    all element Gauss point strains and stresses match exact values.
    """
    # 4-element mesh on [0, 2] x [0, 2] with an interior node perturbed to (0.9, 1.1)
    coords = array([
        [0.0, 0.0], # Node 0 (bot-left)
        [1.0, 0.0], # Node 1 (bot-mid)
        [2.0, 0.0], # Node 2 (bot-right)
        [0.0, 1.0], # Node 3 (mid-left)
        [0.9, 1.1], # Node 4 (perturbed interior node)
        [2.0, 1.0], # Node 5 (mid-right)
        [0.0, 2.0], # Node 6 (top-left)
        [1.0, 2.0], # Node 7 (top-mid)
        [2.0, 2.0]  # Node 8 (top-right)
    ])
    
    cells = array([
        [0, 1, 4, 3],
        [1, 2, 5, 4],
        [3, 4, 7, 6],
        [4, 5, 8, 7]
    ])
    
    mesh = Mesh(coords=coords, cells=cells)
    
    # Target constant strain state:
    eps_xx0 = 1.0e-3
    eps_yy0 = -0.5e-3
    gam_xy0 = 2.0e-3
    
    # Exact linear displacement field:
    # u_x(x, y) = eps_xx0 * x + 0.5 * gam_xy0 * y
    # u_y(x, y) = eps_yy0 * y + 0.5 * gam_xy0 * x
    def exact_disp(x, y):
        ux = eps_xx0 * x + 0.5 * gam_xy0 * y
        uy = eps_yy0 * y + 0.5 * gam_xy0 * x
        return ux, uy

    # Set up fields & DofMap
    fields = [FieldSpec(name="u", components=2, location="nodes", unknown=True)]
    dof_map = DofMap(fields=fields, geometry=mesh)
    
    # Setup material and formulation (E=2e11, nu=0.3)
    material = LinearElasticMaterial(rho=7800.0, E=2.0e11, nu=0.3)
    formulation = LinearElasticityFormulation(material=material, mode="plane_strain")
    
    # Assemble system
    K, M, f = assemble_system(dof_map, formulation, field_name="u")
    
    # Apply Dirichlet boundary conditions to all boundary nodes (nodes 0, 1, 2, 3, 5, 6, 7, 8)
    # Node 4 is the internal node to be solved
    boundary_nodes = [0, 1, 2, 3, 5, 6, 7, 8]
    dirichlet_bcs = {}
    
    for node in boundary_nodes:
        x, y = coords[node]
        ux_exact, uy_exact = exact_disp(x, y)
        dof_x = dof_map.get_dof("u", node, 0)
        dof_y = dof_map.get_dof("u", node, 1)
        dirichlet_bcs[dof_x] = ux_exact
        dirichlet_bcs[dof_y] = uy_exact
        
    # Solve system
    U = solve_system(K, f, dirichlet_bcs)
    
    # 1. Verify interior node 4 displacement matches exact linear solution to high precision
    x4, y4 = coords[4]
    ux4_exact, uy4_exact = exact_disp(x4, y4)
    ux4_fem = U[dof_map.get_dof("u", 4, 0)]
    uy4_fem = U[dof_map.get_dof("u", 4, 1)]
    
    assert np.isclose(ux4_fem, ux4_exact, atol=1e-12)
    assert np.isclose(uy4_fem, uy4_exact, atol=1e-12)
    
    # 2. Compute strains and stresses
    result = compute_element_stresses(mesh, dof_map, formulation, U)
    
    # Verify Gauss point strains match target [eps_xx0, eps_yy0, gam_xy0]
    expected_eps = array([eps_xx0, eps_yy0, gam_xy0])
    for elem_idx in range(mesh.n_elements):
        for q_idx in range(4):
            eps_q = result.gauss_strains[elem_idx, q_idx]
            assert np.allclose(eps_q, expected_eps, atol=1e-10)
            
    # Verify Gauss point stresses match exact D @ eps_xx0
    D = material.get_constitutive_matrix(mode="plane_strain")
    expected_sig = D @ expected_eps
    for elem_idx in range(mesh.n_elements):
        for q_idx in range(4):
            sig_q = result.gauss_stresses[elem_idx, q_idx]
            assert np.allclose(sig_q, expected_sig, atol=1e-8)

if __name__ == "__main__":
    test_constant_strain_patch_test()
    print("Constant strain patch test passed successfully!")
