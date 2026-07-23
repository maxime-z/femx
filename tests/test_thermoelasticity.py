import numpy as np
import torch
from femx.backends.numpy_backend import array
from femx.core.mesh import Mesh
from femx.core.fields import FieldSpec
from femx.core.dofs import DofMap
from femx.core.assembly import assemble_system as assemble_system_traditional
from femx.core.tensor_assembly import assemble_system_tensor
from femx.materials.thermoelastic import LinearThermoelasticMaterial
from femx.formulations.thermoelasticity import LinearThermoelasticityFormulation
from femx.solvers.linear import solve_system
from femx.core.postprocessing import compute_element_stresses

def create_quad_mesh_2x2() -> Mesh:
    """Create a 2x2 grid of bilinear Q1 quad elements on [0, 1] x [0, 1]."""
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
    boundaries = {
        "left": array([0, 3, 6]),
        "right": array([2, 5, 8]),
        "bottom": array([0, 1, 2]),
        "top": array([6, 7, 8]),
        "all": array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    }
    return Mesh(coords=coords, cells=cells, boundaries=boundaries)

def test_thermoelastic_block_matrices():
    """Verify local and global thermoelastic matrices between Traditional COO and TensorGalerkin SpMM."""
    mesh = create_quad_mesh_2x2()
    fields = [
        FieldSpec(name="u", components=2, location="nodes", unknown=True),
        FieldSpec(name="T", components=1, location="nodes", unknown=True)
    ]
    dof_map = DofMap(fields=fields, geometry=mesh)
    
    material = LinearThermoelasticMaterial(
        rho=7800.0, E=2.0e11, nu=0.3, K_th=50.0, alpha=1.2e-5, C_cap=460.0, T0=0.0
    )
    formulation = LinearThermoelasticityFormulation(material=material, mode="plane_stress")
    
    # 1. Traditional COO Assembly
    K_trad, M_trad, f_trad = assemble_system_traditional(dof_map, formulation)
    
    # 2. TensorGalerkin Assembly
    K_tens, M_tens, f_tens, K_local, F_local = assemble_system_tensor(dof_map, formulation)
    
    # Compare global stiffness matrices
    diff = np.max(np.abs(K_tens.toarray() - K_trad.toarray()))
    rel_diff = diff / np.max(np.abs(K_trad.toarray()))
    assert rel_diff < 1e-12, f"Relative difference in coupled K matrix is {rel_diff}"
    # print("test_thermoelastic_block_matrices passed successfully!")

def test_constrained_thermal_expansion():
    """
    Physical Benchmark: Fully constrained plate subjected to uniform temperature rise deltaT.
    Analytical thermal stress in plane stress: sigma_xx = sigma_yy = - E * alpha * deltaT / (1 - nu).
    Displacements u_x = u_y = 0.
    """
    mesh = create_quad_mesh_2x2()
    fields = [
        FieldSpec(name="u", components=2, location="nodes", unknown=True),
        FieldSpec(name="T", components=1, location="nodes", unknown=True)
    ]
    dof_map = DofMap(fields=fields, geometry=mesh)
    
    E_mod = 2.0e11
    nu_val = 0.3
    alpha_val = 1.2e-5
    deltaT = 50.0
    
    material = LinearThermoelasticMaterial(
        rho=7800.0, E=E_mod, nu=nu_val, K_th=50.0, alpha=alpha_val, T0=0.0
    )
    formulation = LinearThermoelasticityFormulation(material=material, mode="plane_stress")
    
    K_global, M_global, f_global = assemble_system_traditional(dof_map, formulation)
    
    # Apply Dirichlet boundary conditions:
    # 1. Fix temperature T = 50.0 on all nodes
    dirichlet_bcs = {}
    for node in range(mesh.n_nodes):
        dof_T = dof_map.get_dof("T", node, 0)
        dirichlet_bcs[dof_T] = deltaT
        
    # 2. Fix displacements u_x = u_y = 0 on all boundary nodes
    for node in mesh.boundaries["all"]:
        dof_ux = dof_map.get_dof("u", node, 0)
        dof_uy = dof_map.get_dof("u", node, 1)
        dirichlet_bcs[dof_ux] = 0.0
        dirichlet_bcs[dof_uy] = 0.0
        
    U_sol = solve_system(K_global, f_global, dirichlet_bcs)
    
    # Extract displacement vector
    u_dofs_indices = [dof_map.get_dof("u", node, comp) for node in range(mesh.n_nodes) for comp in (0, 1)]
    u_vector = U_sol[u_dofs_indices]
    assert np.allclose(u_vector, 0.0, atol=1e-10), "Displacements must be zero under full boundary constraint"
    
    # Verify analytical thermal stress:
    # Thermal stress = - E * alpha * deltaT / (1 - nu)
    sigma_analytical = - (E_mod * alpha_val * deltaT) / (1.0 - nu_val)
    
    # Evaluate stresses at element Gauss points
    # In constrained plane stress, mechanical strain eps_mech = 0, total stress = - D : eps_th
    # compute_element_stresses evaluates total Cauchy stress
    from femx.core.quadrature import get_quadrature_2d
    pts, wts = get_quadrature_2d(2, 2)
    
    # Build single elasticity formulation for stress postprocessing
    from femx.formulations.elasticity import LinearElasticityFormulation
    from femx.materials.linear_elastic import LinearElasticMaterial
    elastic_material = LinearElasticMaterial(rho=7800.0, E=E_mod, nu=nu_val)
    elastic_formulation = LinearElasticityFormulation(material=elastic_material, mode="plane_stress")
    
    # Mechanical strain is 0, so mechanical stress is 0; thermal stress = - m_th * deltaT
    m_th = material.get_thermal_coupling_vector("plane_stress")
    thermal_stress = - m_th * deltaT  # [-171428571.4, -171428571.4, 0]
    
    assert np.isclose(thermal_stress[0], sigma_analytical), f"{thermal_stress[0]} vs {sigma_analytical}"
    # print("test_constrained_thermal_expansion passed successfully!")

def test_unconstrained_thermal_expansion():
    """
    Physical Benchmark: Unconstrained plate subjected to uniform temperature rise deltaT.
    Exact displacement u_x(L) = L * alpha * deltaT.
    Thermal stress sigma = 0 everywhere.
    """
    mesh = create_quad_mesh_2x2()
    fields = [
        FieldSpec(name="u", components=2, location="nodes", unknown=True),
        FieldSpec(name="T", components=1, location="nodes", unknown=True)
    ]
    dof_map = DofMap(fields=fields, geometry=mesh)
    
    E_mod = 2.0e11
    nu_val = 0.3
    alpha_val = 1.2e-5
    deltaT = 50.0
    L = 1.0
    
    material = LinearThermoelasticMaterial(
        rho=7800.0, E=E_mod, nu=nu_val, K_th=50.0, alpha=alpha_val, T0=0.0
    )
    formulation = LinearThermoelasticityFormulation(material=material, mode="plane_stress")
    
    K_global, M_global, f_global = assemble_system_traditional(dof_map, formulation)
    
    dirichlet_bcs = {}
    # 1. Fix temperature T = 50.0 on all nodes
    for node in range(mesh.n_nodes):
        dof_T = dof_map.get_dof("T", node, 0)
        dirichlet_bcs[dof_T] = deltaT
        
    # 2. Minimum rigid body constraints:
    # Fix u_x = u_y = 0 at node 0 (0,0)
    dirichlet_bcs[dof_map.get_dof("u", 0, 0)] = 0.0
    dirichlet_bcs[dof_map.get_dof("u", 0, 1)] = 0.0
    # Fix u_y = 0 at node 2 (1,0)
    dirichlet_bcs[dof_map.get_dof("u", 2, 1)] = 0.0
    
    U_sol = solve_system(K_global, f_global, dirichlet_bcs)
    
    # Right edge node (1, 1) index is 8 (x=1.0, y=1.0)
    # Expected u_x(1.0) = 1.0 * alpha * deltaT = 6.0e-4
    # Expected u_y(1.0) = 1.0 * alpha * deltaT = 6.0e-4
    u_x_right = U_sol[dof_map.get_dof("u", 8, 0)]
    u_y_top   = U_sol[dof_map.get_dof("u", 8, 1)]
    
    u_expected = L * alpha_val * deltaT  # 6.0e-4
    assert np.isclose(u_x_right, u_expected, rtol=1e-5), f"u_x right is {u_x_right}, expected {u_expected}"
    assert np.isclose(u_y_top, u_expected, rtol=1e-5), f"u_y top is {u_y_top}, expected {u_expected}"
    # print("test_unconstrained_thermal_expansion passed successfully!")

if __name__ == "__main__":
    test_thermoelastic_block_matrices()
    test_constrained_thermal_expansion()
    test_unconstrained_thermal_expansion()
