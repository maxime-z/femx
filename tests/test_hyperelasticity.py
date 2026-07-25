import numpy as np
from femx.core.mesh import Mesh
from femx.core.fields import FieldSpec
from femx.core.dofs import DofMap
from femx.core.state import State
from femx.materials.hyperelastic import NeoHookeanMaterial
from femx.formulations.hyperelasticity import HyperelasticFormulation
from femx.solvers.nonlinear import NewtonSolver

def test_neohookean_constitutive():
    """Verify Neo-Hookean stress and tangent at zero deformation F = I."""
    material = NeoHookeanMaterial(rho=1.0, E=1.0e5, nu=0.3)
    lambda_, mu = material.get_lame_parameters()
    
    F = np.eye(2)
    P, C4 = material.update(F)
    
    # At F = I, P should be zero (undeformed state)
    np.testing.assert_allclose(P, np.zeros((2, 2)), atol=1e-12)
    
    # Check tangent tensor C4 at F = I
    # C4[i, j, k, l] = lambda * delta_ij * delta_kl + mu * (delta_ik * delta_jl + delta_il * delta_jk)
    delta = np.eye(2)
    C4_expected = mu * np.einsum('ik,jl->ijkl', delta, delta) + \
                  lambda_ * np.einsum('ij,kl->ijkl', delta, delta) + \
                  mu * np.einsum('kj,il->ijkl', delta, delta)
                  
    np.testing.assert_allclose(C4, C4_expected, atol=1e-12)

def test_single_element_hyperelastic_newton():
    """Verify Newton solver quadratic convergence on a 1-element hyperelastic block under shear."""
    # 1 element unit square mesh
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    cells = np.array([[0, 1, 2, 3]])
    mesh = Mesh(coords=coords, cells=cells)
    
    fields = [FieldSpec(name="u", components=2, location="nodes", unknown=True)]
    dof_map = DofMap(fields=fields, geometry=mesh)
    
    material = NeoHookeanMaterial(rho=1.0, E=1.0e5, nu=0.3)
    formulation = HyperelasticFormulation(material=material)
    
    state = State()
    state.initialize_field("u", 4, 2)
    
    # Dirichlet BCs: Fix bottom (nodes 0, 1) in y, node 0 in x. Prescribe node 2, 3 with u_x = 0.2
    dirichlet_bcs = {
        dof_map.get_dof("u", 0, 0): 0.0,
        dof_map.get_dof("u", 0, 1): 0.0,
        dof_map.get_dof("u", 1, 1): 0.0,
        dof_map.get_dof("u", 2, 0): 0.2,
        dof_map.get_dof("u", 3, 0): 0.2,
    }
    
    solver = NewtonSolver(rtol=1e-10, atol=1e-10, max_iter=10)
    final_state, history = solver.solve(dof_map, formulation, state, dirichlet_bcs)
    print("\nConvergence history:", history)
    assert len(history) <= 6
    assert history[-1] < 1e-8
