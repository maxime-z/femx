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
    from femx.basis.lagrange import LagrangeQuad
    quad_pts, quad_wts = get_quadrature_2d(2, 2)
    elem_basis = LagrangeQuad(p=1)

    K_local_np = K_local.cpu().numpy()
    for e in range(mesh.n_elements):
        elem_coords = mesh.coords[mesh.cells[e]]
        Ke_expected, _, _ = formulation.compute_element_matrices(
            elem_coords, quad_pts, quad_wts, elem_basis=elem_basis
        )
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

def test_tensor_body_load_matches_traditional():
    """Body loads sampled on the tensor path match the element-loop assembler."""
    mesh = create_sample_mesh_quads()

    def heat_source(xyz):
        return 4.0 * xyz[0] - xyz[1]

    heat_fields = [FieldSpec(name="T", components=1, location="nodes", unknown=True)]
    heat_dofs = DofMap(fields=heat_fields, geometry=mesh)
    heat = HeatConductionFormulation(material=LinearHeatMaterial(rho=1.0, C=1.0, K=10.0))
    _, _, f_heat_trad = assemble_system_traditional(heat_dofs, heat, field_name="T", body_load=heat_source)
    _, _, f_heat_tens, _, _ = assemble_system_tensor(heat_dofs, heat, field_name="T", body_load=heat_source)
    assert np.allclose(f_heat_tens, f_heat_trad, atol=1e-10)

    per_element = np.arange(mesh.n_elements, dtype=float)[:, None]
    _, _, f_elem_trad = assemble_system_traditional(heat_dofs, heat, field_name="T", body_load=per_element)
    _, _, f_elem_tens, _, _ = assemble_system_tensor(heat_dofs, heat, field_name="T", body_load=per_element)
    assert np.allclose(f_elem_tens, f_elem_trad, atol=1e-10)

    elastic_fields = [FieldSpec(name="u", components=2, location="nodes", unknown=True)]
    elastic_dofs = DofMap(fields=elastic_fields, geometry=mesh)
    elastic = LinearElasticityFormulation(
        material=LinearElasticMaterial(rho=1.0, E=2.0e7, nu=0.3), mode="plane_strain",
    )
    gravity = np.array([0.0, -9.81])
    _, _, f_elas_trad = assemble_system_traditional(elastic_dofs, elastic, field_name="u", body_load=gravity)
    _, _, f_elas_tens, _, _ = assemble_system_tensor(elastic_dofs, elastic, field_name="u", body_load=gravity)
    assert np.allclose(f_elas_tens, f_elas_trad, atol=1e-8)
    assert np.linalg.norm(f_elas_trad) > 0.0

    from femx.materials.thermoelastic import LinearThermoelasticMaterial
    from femx.formulations.thermoelasticity import LinearThermoelasticityFormulation

    coupled_fields = [
        FieldSpec(name="u", components=2, location="nodes", unknown=True),
        FieldSpec(name="T", components=1, location="nodes", unknown=True),
    ]
    coupled_dofs = DofMap(fields=coupled_fields, geometry=mesh)
    coupled = LinearThermoelasticityFormulation(
        material=LinearThermoelasticMaterial(
            rho=1.0, E=2.0e7, nu=0.3, K_th=10.0, alpha=1.0e-5, C_cap=1.0, T0=20.0,
        ),
        mode="plane_stress",
    )
    coupled_load = {"u": gravity, "T": heat_source}
    _, _, f_coupled_trad = assemble_system_traditional(coupled_dofs, coupled, body_load=coupled_load)
    _, _, f_coupled_tens, _, _ = assemble_system_tensor(coupled_dofs, coupled, body_load=coupled_load)
    assert np.allclose(f_coupled_tens, f_coupled_trad, atol=1e-6)

if __name__ == "__main__":
    test_tensor_vs_traditional_heat_quads()
    test_tensor_vs_traditional_elasticity()
    test_tensor_body_load_matches_traditional()
    print("TensorGalerkin local and global validation tests passed successfully!")
