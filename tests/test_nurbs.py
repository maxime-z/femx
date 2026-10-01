import numpy as np
from femx.geometry.nurbs import KnotVector, NurbsPatch, insert_knot, degree_elevate, h_refine, decompose_to_beziers
from femx.basis.nurbs import (
    compute_nurbs_mapping, get_quadrature_spans, ders_basis_functions,
    NurbsQuadratureCache, NurbsBasis,
)
from femx.core.quadrature import get_quadrature_2d

def test_knot_vector():
    knots = [0., 0., 0., 1., 2., 2., 2.]
    U = KnotVector(knots)
    assert len(U) == 7
    assert U.find_span(2, 0.5) == 2
    assert U.find_span(2, 1.5) == 3
    assert U.find_multiplicity(0.) == 3
    assert U.find_multiplicity(1.) == 1
    assert U.find_multiplicity(2.) == 3

    uk, counts = U.unique_knots()
    assert np.allclose(uk, [0., 1., 2.])
    assert np.array_equal(counts, [3, 1, 3])

def get_simple_patch():
    p = 2
    knots = [0., 0., 0., 1., 2., 2., 2.]
    U = KnotVector(knots)

    Pw = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [2.0, 0.0, 1.0],
        [3.0, 1.0, 1.0],
    ])
    return NurbsPatch.from_weighted_control_points((p,), (U,), Pw)

def eval_curve(patch: NurbsPatch, u: float) -> np.ndarray:
    """Evaluate a 1D NURBS curve at parametric coordinate u."""
    p = patch.degrees[0]
    U = patch.knot_vectors[0]
    span = U.find_span(p, u)
    ders = ders_basis_functions(span, u, p, 0, U.knots)
    N = ders[0]
    w_local = patch.weights[span - p : span + 1]
    P_local = patch.control_points[span - p : span + 1]
    W = np.dot(w_local, N)
    R = (w_local * N) / W
    return R @ P_local

def eval_surface(patch: NurbsPatch, u: float, v: float) -> np.ndarray:
    """Evaluate a 2D NURBS surface at parametric (u, v)."""
    span_u = patch.knot_vectors[0].find_span(patch.degrees[0], u)
    span_v = patch.knot_vectors[1].find_span(patch.degrees[1], v)
    u1, u2 = patch.knots_u[span_u], patch.knots_u[span_u + 1]
    v1, v2 = patch.knots_v[span_v], patch.knots_v[span_v + 1]
    xi = -1.0 if u2 == u1 else 2.0 * (u - u1) / (u2 - u1) - 1.0
    eta = -1.0 if v2 == v1 else 2.0 * (v - v1) / (v2 - v1) - 1.0
    R, _, _ = compute_nurbs_mapping(np.array([xi, eta]), patch, span_u, span_v)
    cell = patch.get_element_control_points(span_u, span_v)
    flat = patch.control_points.transpose(1, 0, 2).reshape((-1, patch.physical_dim))
    return R @ flat[cell]

def test_knot_insertion():
    patch = get_simple_patch()
    us = np.linspace(0.0, 2.0, 21)
    pts_before = np.array([eval_curve(patch, u) for u in us])

    patch2 = insert_knot(patch, 0, 1.5, r=1)
    assert len(patch2.knot_vectors[0]) == len(patch.knot_vectors[0]) + 1
    pts_after = np.array([eval_curve(patch2, u) for u in us])
    assert np.max(np.linalg.norm(pts_after - pts_before, axis=1)) < 1e-12

    bez_patch = decompose_to_beziers(patch)
    assert len(bez_patch.knot_vectors[0]) == 8

def test_degree_elevation():
    # Multi-span: structural checks
    patch = get_simple_patch()
    elevated = degree_elevate(patch, 0, 1)
    assert elevated.degrees[0] == 3
    uk, counts = elevated.knot_vectors[0].unique_knots()
    assert np.allclose(uk, [0., 1., 2.])
    assert np.array_equal(counts, [4, 2, 4])
    assert len(elevated.control_points) == 6

    # Pointwise geometry preservation on a single Bezier segment
    U = KnotVector([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    Pw = np.array([[0.0, 0.0, 1.0], [0.5, 1.0, 1.0], [1.0, 0.0, 1.0]])
    bezier = NurbsPatch.from_weighted_control_points((2,), (U,), Pw)
    elev_bez = degree_elevate(bezier, 0, 1)
    us = np.linspace(0.0, 1.0, 21)
    pts_before = np.array([eval_curve(bezier, u) for u in us])
    pts_after = np.array([eval_curve(elev_bez, u) for u in us])
    assert np.max(np.linalg.norm(pts_after - pts_before, axis=1)) < 1e-12

def get_quarter_annulus():
    w = 1.0 / np.sqrt(2.0)
    Pw = np.array([
        [[1.0, 0.0, 1.0], [w, w, w], [0.0, 1.0, 1.0]],
        [[2.0, 0.0, 1.0], [2.0*w, 2.0*w, w], [0.0, 2.0, 1.0]]
    ])

    U = KnotVector([0.0, 0.0, 1.0, 1.0])
    V = KnotVector([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])

    return NurbsPatch.from_weighted_control_points((1, 2), (U, V), Pw)

def test_surface_geometry_preservation():
    patch = get_quarter_annulus()
    samples = [(u, v) for u in np.linspace(0.0, 1.0, 5) for v in np.linspace(0.0, 1.0, 5)]
    pts_before = np.array([eval_surface(patch, u, v) for u, v in samples])

    refined = insert_knot(patch, 0, 0.5, r=1)
    pts_after = np.array([eval_surface(refined, u, v) for u, v in samples])
    assert np.max(np.linalg.norm(pts_after - pts_before, axis=1)) < 1e-12

    elevated = degree_elevate(patch, 1, 1)
    pts_elev = np.array([eval_surface(elevated, u, v) for u, v in samples])
    assert np.max(np.linalg.norm(pts_elev - pts_before, axis=1)) < 1e-12

def test_positive_jacobian_quarter_annulus():
    patch = get_quarter_annulus()
    spans = patch.get_element_spans()
    gps, _ = get_quadrature_2d(patch.p_u + 1, patch.p_v + 1)
    for span_u, span_v in spans:
        for gp in gps:
            _, _, detJ = compute_nurbs_mapping(gp, patch, span_u, span_v)
            assert detJ > 0.0

def test_nurbs_quadrature_cache():
    patch = get_quarter_annulus()
    cache = NurbsQuadratureCache(patch)
    assert cache.N.ndim == 3
    assert cache.dN_dxi.shape[-1] == 2
    assert np.all(cache.detJ > 0.0)

    span_u, span_v = cache.spans[0]
    basis = NurbsBasis(patch, span_u, span_v, cache=cache)
    gp = cache.quad_pts[0]
    N_c, dN_c, detJ_c = basis.compute_mapping(gp)
    N, dN, detJ = compute_nurbs_mapping(gp, patch, span_u, span_v)
    assert np.allclose(N_c, N)
    assert np.allclose(dN_c, dN)
    assert np.isclose(detJ_c, detJ)

def get_quarter_hollow_sphere():
    """Quarter of a hollow sphere in the positive octant."""
    w2 = 1.0 / np.sqrt(2.0)
    w = np.array([1.0, w2, 1.0])
    W = np.einsum('j,k->jk', w, w)
    Pw = np.zeros((2, 3, 3, 4))

    for i, r in enumerate([1.0, 2.0]):
        for j in range(3):
            if j == 0:
                pt_xz = np.array([0.0, r])
            elif j == 1:
                pt_xz = np.array([r, r])
            else:
                pt_xz = np.array([r, 0.0])

            for k in range(3):
                if k == 0:
                    x = pt_xz[0]
                    y = 0.0
                elif k == 1:
                    x = pt_xz[0]
                    y = pt_xz[0]
                else:
                    x = 0.0
                    y = pt_xz[0]
                z = pt_xz[1]
                weight = W[j, k]
                Pw[i, j, k] = np.array([x * weight, y * weight, z * weight, weight])

    U = KnotVector([0.0, 0.0, 1.0, 1.0])
    V = KnotVector([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    W_kv = KnotVector([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    return NurbsPatch.from_weighted_control_points((1, 2, 2), (U, V, W_kv), Pw)

def test_quadrature_precision():
    """
    Test Gaussian quadrature to compute the area of a NURBS quarter annulus
    and construct a 3D NURBS quarter hollow sphere (volume integral deferred).
    """
    patch2d = get_quarter_annulus()
    spans2d = get_quadrature_spans(patch2d)

    exact_area = 0.75 * np.pi

    def compute_area(nu, nv):
        area = 0.0
        gps, weights = get_quadrature_2d(nu, nv)
        for (span_u, span_v), domain in spans2d:
            for i in range(len(weights)):
                _, _, detJ = compute_nurbs_mapping(gps[i], patch2d, span_u, span_v)
                area += detJ * weights[i]
        return area

    area_p1 = compute_area(patch2d.degrees[0] + 1, patch2d.degrees[1] + 1)
    err_p1 = abs(area_p1 - exact_area)

    area_high = compute_area(patch2d.degrees[0] + 4, patch2d.degrees[1] + 4)
    err_high = abs(area_high - exact_area)

    assert err_high < err_p1
    assert err_high < 1e-4

    patch3d = get_quarter_hollow_sphere()
    assert patch3d.parametric_dim == 3
    assert patch3d.physical_dim == 3

if __name__ == '__main__':
    test_knot_vector()
    test_knot_insertion()
    test_degree_elevation()
    test_surface_geometry_preservation()
    test_positive_jacobian_quarter_annulus()
    test_nurbs_quadrature_cache()
    test_quadrature_precision()
    print("All tests passed!")
