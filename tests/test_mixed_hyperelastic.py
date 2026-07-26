import numpy as np
from femx.core.fields import FieldSpec
from femx.core.dofs import DofMap
from femx.core.mesh import Mesh, NurbsPatch
from femx.core.state import State
from femx.core.assembly import assemble_nonlinear_system
from femx.materials.mixed_hyperelastic import MixedNeoHookeanMaterial, QuadraticPenalty, SimoMiehePenalty
from femx.formulations.hyperelasticity import MixedHyperelasticFormulation, FBarHyperelasticFormulation
from femx.materials.hyperelastic import NeoHookeanMaterial

def create_unit_square_mesh(nx: int = 1, ny: int = 1) -> Mesh:
    x = np.linspace(0, 1.0, nx + 1)
    y = np.linspace(0, 1.0, ny + 1)
    X, Y = np.meshgrid(x, y)
    coords = np.vstack([X.ravel(), Y.ravel()]).T
    cells = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = (j + 1) * (nx + 1) + i + 1
            n3 = n2 - 1
            cells.append([n0, n1, n2, n3])
    return Mesh(coords=coords, cells=np.array(cells, dtype=int))


def test_mixed_hyperelastic_residual_and_tangent():
    # 2D Q1 mesh (4 nodes, 1 element)
    mesh = create_unit_square_mesh(nx=1, ny=1)
    
    u_field = FieldSpec(name="u", components=2, location="nodes", unknown=True)
    p_field = FieldSpec(name="p", components=1, location="nodes", unknown=True)
    dof_map = DofMap([u_field, p_field], mesh)
    
    mat = MixedNeoHookeanMaterial(rho=1.0, mu=10.0, kappa=1000.0, penalty=QuadraticPenalty())
    form = MixedHyperelasticFormulation(mat)
    
    state = State(values={
        "u": np.zeros((mesh.n_nodes, 2)),
        "p": np.zeros(mesh.n_nodes)
    })
    
    K, R = assemble_nonlinear_system(dof_map, form, state)
    
    assert K.shape == (dof_map.n_dofs, dof_map.n_dofs)
    assert R.shape == (dof_map.n_dofs,)
    # At u=0, p=0, residual should be zero
    np.testing.assert_allclose(R, 0.0, atol=1e-12)


def test_fbar_hyperelastic():
    mesh = create_unit_square_mesh(nx=2, ny=2)
    u_field = FieldSpec(name="u", components=2, location="nodes", unknown=True)
    dof_map = DofMap([u_field], mesh)
    
    mat = NeoHookeanMaterial(rho=1.0, E=100.0, nu=0.499)
    form = FBarHyperelasticFormulation(mat)
    
    state = State(values={
        "u": np.zeros((mesh.n_nodes, 2))
    })
    
    K, R = assemble_nonlinear_system(dof_map, form, state)
    
    assert K.shape == (dof_map.n_dofs, dof_map.n_dofs)
    assert R.shape == (dof_map.n_dofs,)
    np.testing.assert_allclose(R, 0.0, atol=1e-12)


if __name__ == "__main__":
    test_mixed_hyperelastic_residual_and_tangent()
    test_fbar_hyperelastic()
    print("All tests passed successfully!")
