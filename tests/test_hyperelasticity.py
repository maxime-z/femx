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

    np.testing.assert_allclose(P, np.zeros((2, 2)), atol=1e-12)

    delta = np.eye(2)
    C4_expected = mu * np.einsum('ik,jl->ijkl', delta, delta) + \
                  lambda_ * np.einsum('ij,kl->ijkl', delta, delta) + \
                  mu * np.einsum('kj,il->ijkl', delta, delta)

    np.testing.assert_allclose(C4, C4_expected, atol=1e-12)

def test_single_element_hyperelastic_newton():
    """Verify Newton solver quadratic convergence on a 1-element hyperelastic block under shear."""
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    cells = np.array([[0, 1, 2, 3]])
    mesh = Mesh(coords=coords, cells=cells)

    fields = [FieldSpec(name="u", components=2, location="nodes", unknown=True)]
    dof_map = DofMap(fields=fields, geometry=mesh)

    material = NeoHookeanMaterial(rho=1.0, E=1.0e5, nu=0.3)
    formulation = HyperelasticFormulation(material=material)

    state = State()
    state.initialize_field("u", 4, 2)

    dirichlet_bcs = {
        dof_map.get_dof("u", 0, 0): 0.0,
        dof_map.get_dof("u", 0, 1): 0.0,
        dof_map.get_dof("u", 1, 1): 0.0,
        dof_map.get_dof("u", 2, 0): 0.2,
        dof_map.get_dof("u", 3, 0): 0.2,
    }

    solver = NewtonSolver(rtol=1e-10, atol=1e-10, max_iter=10)
    final_state, history = solver.solve(dof_map, formulation, state, dirichlet_bcs)
    assert len(history) <= 6
    assert history[-1] < 1e-8


def _make_stateful_unit_square():
    from femx.materials.stateful import AccumulatingDeformationMaterial
    from femx.formulations.hyperelasticity import StatefulHyperelasticFormulation

    coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    cells = np.array([[0, 1, 2, 3]])
    mesh = Mesh(coords=coords, cells=cells)
    fields = [FieldSpec(name="u", components=2, location="nodes", unknown=True)]
    dof_map = DofMap(fields=fields, geometry=mesh)
    material = AccumulatingDeformationMaterial(rho=1.0, E=1.0e5, nu=0.3)
    formulation = StatefulHyperelasticFormulation(material=material)
    state = State()
    state.initialize_field("u", 4, 2)
    state.initialize_gauss_variable("alpha", n_elements=1, n_gps=4, components=1)
    return mesh, dof_map, formulation, state


def test_rejected_trial_does_not_commit_alpha():
    """A residual evaluation that writes trial alpha must leave committed alpha unchanged."""
    from femx.core.assembly import assemble_nonlinear_system

    mesh, dof_map, formulation, state = _make_stateful_unit_square()
    state.values["u"][:] = 0.0
    state.values["u"][2, 0] = 0.1
    state.values["u"][3, 0] = 0.1
    state.copy_committed_to_trial()

    assert np.allclose(state.gauss_variables["alpha"], 0.0)
    assemble_nonlinear_system(dof_map, formulation, state)
    assert np.any(state.trial_gauss_variables["alpha"] > 0.0)
    assert np.allclose(state.gauss_variables["alpha"], 0.0)


def test_load_stepper_commits_alpha_across_increments():
    """Successful LoadStepper steps commit alpha; the next increment starts from that value."""
    from femx.solvers.nonlinear import LoadStepper

    mesh, dof_map, formulation, state = _make_stateful_unit_square()
    full_bcs = {
        dof_map.get_dof("u", 0, 0): 0.0,
        dof_map.get_dof("u", 0, 1): 0.0,
        dof_map.get_dof("u", 1, 1): 0.0,
        dof_map.get_dof("u", 2, 0): 0.2,
        dof_map.get_dof("u", 3, 0): 0.2,
    }
    stepper = LoadStepper(n_steps=2, newton_solver=NewtonSolver(rtol=1e-8, atol=1e-10, max_iter=15))
    final_state, history = stepper.solve(dof_map, formulation, state, full_bcs)

    alpha_after = final_state.gauss_variables["alpha"].copy()
    assert np.any(alpha_after > 0.0)

    full_bcs2 = {
        dof_map.get_dof("u", 0, 0): 0.0,
        dof_map.get_dof("u", 0, 1): 0.0,
        dof_map.get_dof("u", 1, 1): 0.0,
        dof_map.get_dof("u", 2, 0): 0.3,
        dof_map.get_dof("u", 3, 0): 0.3,
    }
    alpha_before_second = final_state.gauss_variables["alpha"].copy()
    stepper2 = LoadStepper(n_steps=1, newton_solver=NewtonSolver(rtol=1e-8, atol=1e-10, max_iter=15))
    final_state2, _ = stepper2.solve(dof_map, formulation, final_state, full_bcs2)
    assert np.all(final_state2.gauss_variables["alpha"] >= alpha_before_second - 1e-15)
    assert np.any(final_state2.gauss_variables["alpha"] > alpha_before_second)
