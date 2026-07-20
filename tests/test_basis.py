import numpy as np
from femx.backends.numpy_backend import array
from femx.core.mesh import NurbsPatch
from femx.basis.lagrange import evaluate_q1_shape_functions, evaluate_q1_shape_derivatives_ref, compute_q1_mapping
from femx.basis.nurbs import find_span, ders_basis_functions, compute_nurbs_mapping

def test_q1_shape_functions():
    # At center (0, 0), all shape functions should be 0.25
    N = evaluate_q1_shape_functions(0.0, 0.0)
    assert np.allclose(N, 0.25)
    
    # Check partition of unity at random points
    for xi in np.linspace(-1, 1, 5):
        for eta in np.linspace(-1, 1, 5):
            N = evaluate_q1_shape_functions(xi, eta)
            assert np.isclose(np.sum(N), 1.0)

def test_q1_derivatives_fd():
    # Compare analytical reference derivatives to central finite differences
    h = 1e-6
    for xi in np.linspace(-0.8, 0.8, 4):
        for eta in np.linspace(-0.8, 0.8, 4):
            # Analytical
            dN_dref = evaluate_q1_shape_derivatives_ref(xi, eta)
            
            # FD for xi
            N_plus = evaluate_q1_shape_functions(xi + h, eta)
            N_minus = evaluate_q1_shape_functions(xi - h, eta)
            dN_dxi_fd = (N_plus - N_minus) / (2.0 * h)
            
            # FD for eta
            N_plus = evaluate_q1_shape_functions(xi, eta + h)
            N_minus = evaluate_q1_shape_functions(xi, eta - h)
            dN_deta_fd = (N_plus - N_minus) / (2.0 * h)
            
            assert np.allclose(dN_dref[0], dN_dxi_fd, atol=1e-5)
            assert np.allclose(dN_dref[1], dN_deta_fd, atol=1e-5)

def test_nurbs_find_span():
    # Simple open knot vector of degree 2
    knots = array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0])
    n = 4 # 5 control points, n = 5 - 1 = 4
    p = 2
    
    # Test coordinates
    assert find_span(n, p, 0.5, knots) == 2  # [0.0, 1.0)
    assert find_span(n, p, 1.5, knots) == 3  # [1.0, 2.0)
    assert find_span(n, p, 2.5, knots) == 4  # [2.0, 3.0)
    assert find_span(n, p, 3.0, knots) == 4  # Right boundary

def test_nurbs_basis_partition_of_unity():
    # Simple open knot vector of degree 2
    knots = array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0])
    n = 4
    p = 2
    
    for u in np.linspace(0.1, 2.9, 10):
        span = find_span(n, p, u, knots)
        ders = ders_basis_functions(span, u, p, 1, knots)
        # Row 0 contains active basis function values
        assert np.isclose(np.sum(ders[0]), 1.0)
        # Row 1 contains derivatives, sum of derivatives should be 0.0
        assert np.isclose(np.sum(ders[1]), 0.0, atol=1e-10)

def test_nurbs_2d_mapping():
    # Define a simple quadratic unit square patch
    knots_u = array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    knots_v = array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    
    # Control points coordinate grid for unit square [0, 1]x[0, 1]
    cp_x = array([[0.0, 0.0, 0.0],
                  [0.5, 0.5, 0.5],
                  [1.0, 1.0, 1.0]])
    cp_y = array([[0.0, 0.5, 1.0],
                  [0.0, 0.5, 1.0],
                  [0.0, 0.5, 1.0]])
    control_points = np.stack([cp_x, cp_y], axis=-1)
    
    weights = np.ones((3, 3))
    
    patch = NurbsPatch(
        p_u=2, p_v=2,
        knots_u=knots_u, knots_v=knots_v,
        control_points=control_points,
        weights=weights
    )
    
    spans = patch.get_element_spans()
    assert len(spans) == 1
    span_u, span_v = spans[0]
    
    # Test reference points
    for xi in np.linspace(-0.8, 0.8, 5):
        for eta in np.linspace(-0.8, 0.8, 5):
            R, dR_dphys, detJ = compute_nurbs_mapping(array([xi, eta]), patch, span_u, span_v)
            
            # Partition of unity checks
            assert np.isclose(np.sum(R), 1.0)
            assert np.isclose(np.sum(dR_dphys[0]), 0.0, atol=1e-10)
            assert np.isclose(np.sum(dR_dphys[1]), 0.0, atol=1e-10)
            
            # Since control weights are all 1.0 and grid is uniform/flat,
            # determinant of physical mapping Jacobian from parametric to physical should be 1.0.
            # Reference to parametric scales u by 0.5, v by 0.5, so total detJ = 0.25
            assert np.isclose(detJ, 0.25)
