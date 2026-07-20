import numpy as np
import torch
from femx.backends.numpy_backend import array
from femx.core.mesh import Mesh
from femx.core.fields import FieldSpec
from femx.core.dofs import DofMap
from femx.core.assembly import assemble_system as assemble_system_traditional
from femx.core.tensor_assembly import assemble_system_tensor
from femx.materials.linear_heat import LinearHeatMaterial
from femx.materials.linear_elastic import LinearElasticMaterial
from femx.formulations.heat import HeatConductionFormulation
from femx.formulations.elasticity import LinearElasticityFormulation
from femx.solvers.linear import solve_system

def create_sample_mesh_quads() -> Mesh:
    """Create a 2x2 grid of bilinear Q1 quad elements."""
    coords = array([
        [0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
        [0.0, 0.5], [0.5, 0.5], [1.0, 0.5],
        [0.0, 1.0], [0.5, 1.0], [1.0, 1.0]
    ])
    cells = array([
        [0, 1, 4, 3],
        [1, 2, 5, 4],
        [3, 4, 7, 6],
        [4, 5, 8, 7]
    ])
    left_nodes = array([0, 3, 6])
    right_nodes = array([2, 5, 8])
    boundaries = {"left": left_nodes, "right": right_nodes}
    return Mesh(coords=coords, cells=cells, boundaries=boundaries)

def create_sample_mesh_triangles() -> Mesh:
    """Create a 2-element linear T1 triangle mesh."""
    coords = array([
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]
    ])
    cells = array([
        [0, 1, 2],
        [0, 2, 3]
    ])
    boundaries = {"left": array([0, 3]), "right": array([1, 2])}
    return Mesh(coords=coords, cells=cells, boundaries=boundaries)

def test_tensor_vs_traditional_heat_quads():
    """Verify local and global heat matrices between Traditional and TensorGalerkin engines."""
    mesh = create_sample_mesh_quads()
    fields = [FieldSpec(name="T", components=1, location="nodes", unknown=True)]
    dof_map = DofMap(fields=fields, geometry=mesh)
    
    material = LinearHeatMaterial(rho=1.0, C=1.0, K=10.0)
    formulation = HeatConductionFormulation(material=material)
    
    # Traditional COO Assembly
    K_trad, M_trad, f_trad = assemble_system_traditional(dof_map, formulation, field_name="T")
    
    # TensorGalerkin Assembly
    K_tens, M_tens, f_tens, K_local, F_local = assemble_system_tensor(dof_map, formulation, field_name="T")
    
    # 1. Compare local matrices K_local[e] vs traditional compute_element_matrices
    from femx.core.quadrature import get_quadrature_2d
    quad_pts, quad_wts = get_quadrature_2d(2, 2)
    
    K_local_np = K_local.cpu().numpy()
    for e in range(mesh.n_elements):
        elem_coords = mesh.coords[mesh.cells[e]]
        Ke_expected, _, _ = formulation.compute_element_matrices(elem_coords, quad_pts, quad_wts)
        assert np.allclose(K_local_np[e], Ke_expected, atol=1e-6)
        
    # 2. Compare global stiffness matrix K
    assert np.allclose(K_tens.toarray(), K_trad.toarray(), atol=1e-6)

def test_tensor_vs_traditional_elasticity():
    """Verify local and global elasticity matrices and solve solution equivalence."""
    mesh = create_sample_mesh_quads()
    fields = [FieldSpec(name="u", components=2, location="nodes", unknown=True)]
    dof_map = DofMap(fields=fields, geometry=mesh)
    
    material = LinearElasticMaterial(rho=1.0, E=2.0e7, nu=0.3)
    formulation = LinearElasticityFormulation(material=material, mode="plane_strain")
    
    # Traditional COO Assembly
    K_trad, M_trad, f_trad = assemble_system_traditional(dof_map, formulation, field_name="u")
    
    # TensorGalerkin Assembly
    K_tens, M_tens, f_tens, K_local, F_local = assemble_system_tensor(dof_map, formulation, field_name="u")
    
    # 1. Compare global stiffness matrix K
    assert np.allclose(K_tens.toarray(), K_trad.toarray(), atol=1e-4)
    
    # 2. Compare solution vectors under Dirichlet BCs
    dirichlet_bcs = {}
    for node in mesh.boundaries["left"]:
        dof_x = dof_map.get_dof("u", node, 0)
        dof_y = dof_map.get_dof("u", node, 1)
        dirichlet_bcs[dof_x] = 0.0
        dirichlet_bcs[dof_y] = 0.0
        
    # Apply tip force on right boundary
    for node in mesh.boundaries["right"]:
        dof_y = dof_map.get_dof("u", node, 1)
        f_trad[dof_y] += -500.0
        f_tens[dof_y] += -500.0
        
    U_trad = solve_system(K_trad, f_trad, dirichlet_bcs)
    U_tens = solve_system(K_tens, f_tens, dirichlet_bcs)
    
    assert np.allclose(U_tens, U_trad, atol=1e-6)

if __name__ == "__main__":
    test_tensor_vs_traditional_heat_quads()
    test_tensor_vs_traditional_elasticity()
    print("TensorGalerkin local and global validation tests passed successfully!")
