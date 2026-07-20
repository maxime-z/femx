import numpy as np
from femx.backends.numpy_backend import array
from femx.core.mesh import Mesh
from femx.core.fields import FieldSpec
from femx.core.dofs import DofMap
from femx.core.assembly import assemble_system
from femx.materials.linear_heat import LinearHeatMaterial
from femx.formulations.heat import HeatConductionFormulation
from femx.solvers.linear import apply_dirichlet_bcs, solve_system

def test_single_element_heat_solve():
    # Define a single Q1 square element [0, 1]x[0, 1]
    coords = array([
        [0.0, 0.0], # Node 0
        [1.0, 0.0], # Node 1
        [1.0, 1.0], # Node 2
        [0.0, 1.0]  # Node 3
    ])
    cells = array([[0, 1, 2, 3]])
    
    mesh = Mesh(coords=coords, cells=cells)
    
    # 1 unknown scalar temperature field
    fields = [FieldSpec(name="T", components=1, location="nodes", unknown=True)]
    dof_map = DofMap(fields=fields, geometry=mesh)
    
    assert dof_map.n_dofs == 4
    
    material = LinearHeatMaterial(rho=1.0, C=1.0, K=10.0)
    formulation = HeatConductionFormulation(material=material)
    
    # Assemble K, M, f
    K, M, f = assemble_system(dof_map, formulation, field_name="T", body_load=0.0)
    
    assert K.shape == (4, 4)
    assert M.shape == (4, 4)
    
    # Verify that K is symmetric and positive semi-definite (eigenvalues >= 0)
    K_dense = K.toarray()
    assert np.allclose(K_dense, K_dense.T)
    evals = np.linalg.eigvalsh(K_dense)
    assert np.all(evals >= -1e-12)
    
    # Set Dirichlet BCs: Left side (nodes 0, 3) = 10.0, Right side (nodes 1, 2) = 20.0
    # Global DOFs match node IDs because field has 1 component
    dirichlet_bcs = {
        0: 10.0,
        3: 10.0,
        1: 20.0,
        2: 20.0
    }
    
    # Solve system
    T_solution = solve_system(K, f, dirichlet_bcs)
    
    # Check boundary values
    assert np.isclose(T_solution[0], 10.0)
    assert np.isclose(T_solution[3], 10.0)
    assert np.isclose(T_solution[1], 20.0)
    assert np.isclose(T_solution[2], 20.0)
    
    # Analytical solution for this linear gradient along X is T(x) = 10 + 10 * x.
    # At the center (x = 0.5), T should be 15.0
    # Let's check intermediate points if we had multiple elements, but for a single element:
    # Since Q1 shape functions are bilinear, they interpolate linearly on edges.


def test_dirichlet_variants_agree_and_symmetry():
    """Symmetry-preserving and row-only BCs must give the same solution;
    only the former keeps K symmetric."""
    coords = array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 0.5],
    ])
    cells = array([[0, 1, 2, 3]])
    mesh = Mesh(coords=coords, cells=cells)
    fields = [FieldSpec(name="T", components=1, location="nodes", unknown=True)]
    dof_map = DofMap(fields=fields, geometry=mesh)

    material = LinearHeatMaterial(rho=1.0, C=1.0, K=1.0)
    formulation = HeatConductionFormulation(material=material)
    K, _, f = assemble_system(dof_map, formulation, field_name="T", body_load=0.0)

    # Non-homogeneous Dirichlet on two nodes; leave two free DOFs.
    dirichlet_bcs = {0: 10.0, 1: 20.0}

    K_sym, f_sym = apply_dirichlet_bcs(K, f, dirichlet_bcs, preserve_symmetry=True)
    K_row, f_row = apply_dirichlet_bcs(K, f, dirichlet_bcs, preserve_symmetry=False)

    assert np.allclose(K_sym.toarray(), K_sym.toarray().T)
    assert not np.allclose(K_row.toarray(), K_row.toarray().T)

    u_sym = solve_system(K, f, dirichlet_bcs, preserve_symmetry=True)
    u_row = solve_system(K, f, dirichlet_bcs, preserve_symmetry=False)
    assert np.allclose(u_sym, u_row)
    assert np.isclose(u_sym[0], 10.0)
    assert np.isclose(u_sym[1], 20.0)
