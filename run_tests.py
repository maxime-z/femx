import sys
import traceback
from tests.test_basis import (
    test_q1_shape_functions,
    test_q1_derivatives_fd,
    test_nurbs_find_span,
    test_nurbs_basis_partition_of_unity,
    test_nurbs_2d_mapping
)
from tests.test_solvers import (
    test_single_element_heat_solve,
    test_dirichlet_variants_agree_and_symmetry,
)
from tests.test_patch_test import (
    test_constant_strain_patch_test
)
from tests.test_tensor_assembly import (
    test_tensor_vs_traditional_heat_quads,
    test_tensor_vs_traditional_elasticity
)
from tests.test_thermoelasticity import (
    test_thermoelastic_block_matrices,
    test_constrained_thermal_expansion,
    test_unconstrained_thermal_expansion
)
from tests.test_hyperelasticity import (
    test_neohookean_constitutive,
    test_single_element_hyperelastic_newton
)

def run_test(name, func):
    print(f"Running {name:50s}...", end="")
    try:
        func()
        print(" SUCCESS")
        return True
    except Exception as e:
        print(" FAILED")
        traceback.print_exc()
        return False

def main():
    print("=== Executing femx Unit Tests ===")
    tests = {
        "test_q1_shape_functions": test_q1_shape_functions,
        "test_q1_derivatives_fd": test_q1_derivatives_fd,
        "test_nurbs_find_span": test_nurbs_find_span,
        "test_nurbs_basis_partition_of_unity": test_nurbs_basis_partition_of_unity,
        "test_nurbs_2d_mapping": test_nurbs_2d_mapping,
        "test_single_element_heat_solve": test_single_element_heat_solve,
        "test_dirichlet_variants_agree_and_symmetry": test_dirichlet_variants_agree_and_symmetry,
        "test_constant_strain_patch_test": test_constant_strain_patch_test,
        "test_tensor_vs_traditional_heat_quads": test_tensor_vs_traditional_heat_quads,
        "test_tensor_vs_traditional_elasticity": test_tensor_vs_traditional_elasticity,
        "test_thermoelastic_block_matrices": test_thermoelastic_block_matrices,
        "test_constrained_thermal_expansion": test_constrained_thermal_expansion,
        "test_unconstrained_thermal_expansion": test_unconstrained_thermal_expansion,
        "test_neohookean_constitutive": test_neohookean_constitutive,
        "test_single_element_hyperelastic_newton": test_single_element_hyperelastic_newton,
    }

    
    success = True
    for name, func in tests.items():
        if not run_test(name, func):
            success = False
            
    if success:
        print("\nAll unit tests passed successfully!")
        sys.exit(0)
    else:
        print("\nSome unit tests failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
