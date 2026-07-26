import numpy as np
import matplotlib.pyplot as plt
from femx.geometry.nurbs import KnotVector, NurbsPatch, insert_knot, degree_elevate, h_refine, decompose_to_beziers
from femx.basis.nurbs import compute_nurbs_mapping, get_quadrature_spans, ders_basis_functions
from femx.core.quadrature import get_quadrature_2d

def get_quarter_annulus():
    w = 1.0 / np.sqrt(2.0)
    Pw = np.array([
        [[1.0, 0.0, 1.0], [w, w, w], [0.0, 1.0, 1.0]],
        [[2.0, 0.0, 1.0], [2.0*w, 2.0*w, w], [0.0, 2.0, 1.0]]
    ])
    
    U = KnotVector([0.0, 0.0, 1.0, 1.0])
    V = KnotVector([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    return NurbsPatch.from_weighted_control_points((1, 2), (U, V), Pw)

def eval_physical_point(patch, span_u, span_v, xi, eta):
    p_u, p_v = patch.degrees[0], patch.degrees[1]
    knots_u = patch.knot_vectors[0].knots
    knots_v = patch.knot_vectors[1].knots
    
    u1, u2 = knots_u[span_u], knots_u[span_u + 1]
    v1, v2 = knots_v[span_v], knots_v[span_v + 1]
    
    u = 0.5 * (u2 - u1) * xi + 0.5 * (u2 + u1)
    v = 0.5 * (v2 - v1) * eta + 0.5 * (v2 + v1)
    
    ders_u = ders_basis_functions(span_u, u, p_u, 0, knots_u)
    ders_v = ders_basis_functions(span_v, v, p_v, 0, knots_v)
    
    w_sum = 0.0
    xy_sum = np.zeros(2)
    
    for j in range(p_v + 1):
        idx_v = span_v - p_v + j
        for i in range(p_u + 1):
            idx_u = span_u - p_u + i
            N = ders_u[0, i] * ders_v[0, j]
            w = patch.weights[idx_u, idx_v]
            w_sum += N * w
            xy_sum += N * w * patch.control_points[idx_u, idx_v]
            
    return xy_sum / w_sum

def plot_patch(ax, patch, title):
    # 1. Plot control net
    cp = patch.control_points
    nu, nv, _ = cp.shape
    
    for i in range(nu):
        ax.plot(cp[i, :, 0], cp[i, :, 1], 'k--o', alpha=0.4, markersize=4, label='Control Net' if i==0 else "")
    for j in range(nv):
        ax.plot(cp[:, j, 0], cp[:, j, 1], 'k--o', alpha=0.4, markersize=4)

    # 2. Evaluate and plot physical surface / element boundaries
    spans = get_quadrature_spans(patch)
    n_samples = 15
    xi_vec = np.linspace(-1, 1, n_samples)
    eta_vec = np.linspace(-1, 1, n_samples)
    
    first_elem = True
    for (span_u, span_v), _ in spans:
        X = np.zeros((n_samples, n_samples))
        Y = np.zeros((n_samples, n_samples))
        
        for i, xi in enumerate(xi_vec):
            for j, eta in enumerate(eta_vec):
                pt = eval_physical_point(patch, span_u, span_v, xi, eta)
                X[i, j] = pt[0]
                Y[i, j] = pt[1]
                
        # Fill surface patch
        ax.pcolormesh(X, Y, np.ones_like(X), cmap='Pastel1', alpha=0.3, shading='auto')
        
        # Internal isoparametric grid lines
        for i in range(n_samples):
            ax.plot(X[i, :], Y[i, :], color='teal', alpha=0.2, linewidth=0.8)
            ax.plot(X[:, i], Y[:, i], color='teal', alpha=0.2, linewidth=0.8)
            
        # Boundary of element (span)
        ax.plot(X[0, :], Y[0, :], 'b-', linewidth=1.5, label='Element Boundary' if first_elem else "")
        ax.plot(X[-1, :], Y[-1, :], 'b-', linewidth=1.5)
        ax.plot(X[:, 0], Y[:, 0], 'b-', linewidth=1.5)
        ax.plot(X[:, -1], Y[:, -1], 'b-', linewidth=1.5)
        first_elem = False

    # Format knot vectors string
    u_str = np.array2string(np.round(patch.knot_vectors[0].knots, 2), separator=', ')
    v_str = np.array2string(np.round(patch.knot_vectors[1].knots, 2), separator=', ')
    full_title = f"{title}\nU = {u_str}\nV = {v_str}"
    
    ax.set_title(full_title, fontsize=9)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.6)

def demonstrate_operations():
    patch = get_quarter_annulus()
    
    refined = h_refine(patch)
    elevated_u = degree_elevate(patch, 0, 1)
    elevated = degree_elevate(elevated_u, 1, 1)
    
    patch_with_knots = insert_knot(insert_knot(patch, 0, 0.5), 1, 0.5)
    bezier = decompose_to_beziers(patch_with_knots)
    
    fig, axs = plt.subplots(2, 2, figsize=(11, 11))
    plot_patch(axs[0, 0], patch, "Original Quarter Annulus (p=1, q=2)")
    plot_patch(axs[0, 1], refined, "h-Refined (Uniform Knot Insertion)")
    plot_patch(axs[1, 0], elevated, "p-Refined (Degree Elevated to p=2, q=3)")
    plot_patch(axs[1, 1], bezier, "Bezier Decomposition (C^-1 at Inner Knots)")
    
    import os
    os.makedirs('examples/output', exist_ok=True)
    plt.tight_layout()
    plt.savefig('examples/output/nurbs_operations.png', dpi=200)
    print("Saved updated operations showcase to examples/output/nurbs_operations.png")

def plot_gauss_points(ax, patch, nu, nv, title):
    # Plot geometry base
    spans = get_quadrature_spans(patch)
    n_samples = 15
    xi_vec = np.linspace(-1, 1, n_samples)
    eta_vec = np.linspace(-1, 1, n_samples)
    
    for (span_u, span_v), _ in spans:
        X = np.zeros((n_samples, n_samples))
        Y = np.zeros((n_samples, n_samples))
        for i, xi in enumerate(xi_vec):
            for j, eta in enumerate(eta_vec):
                pt = eval_physical_point(patch, span_u, span_v, xi, eta)
                X[i, j] = pt[0]
                Y[i, j] = pt[1]
        ax.pcolormesh(X, Y, np.ones_like(X), cmap='Pastel1', alpha=0.3, shading='auto')
        ax.plot(X[0, :], Y[0, :], 'b-', linewidth=1.2)
        ax.plot(X[-1, :], Y[-1, :], 'b-', linewidth=1.2)
        ax.plot(X[:, 0], Y[:, 0], 'b-', linewidth=1.2)
        ax.plot(X[:, -1], Y[:, -1], 'b-', linewidth=1.2)

    # Compute Gauss points and plot them
    gps_ref, _ = get_quadrature_2d(nu, nv)
    all_gp_x = []
    all_gp_y = []
    
    for (span_u, span_v), _ in spans:
        for gp in gps_ref:
            pt = eval_physical_point(patch, span_u, span_v, gp[0], gp[1])
            all_gp_x.append(pt[0])
            all_gp_y.append(pt[1])
            
    ax.scatter(all_gp_x, all_gp_y, color='red', s=25, zorder=5, label=f'Gauss Points ({nu}x{nv})')
    ax.set_title(title, fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right')

def test_integration_convergence():
    patch = get_quarter_annulus()
    spans = get_quadrature_spans(patch)
    exact_area = 0.75 * np.pi
    
    def compute_area(nu, nv):
        area = 0.0
        gps, weights = get_quadrature_2d(nu, nv)
        for (span_u, span_v), _ in spans:
            for i in range(len(weights)):
                _, _, detJ = compute_nurbs_mapping(gps[i], patch, span_u, span_v)
                area += detJ * weights[i]
        return area

    orders = list(range(1, 11))
    errors = []
    
    for o in orders:
        area_calc = compute_area(o, o)
        err = abs(area_calc - exact_area)
        if err < 1e-16:
            err = 1e-16
        errors.append(err)
        print(f"Quadrature Order {o}x{o}: Area = {area_calc:.8f}, Error = {err:.2e}")
        
    fig = plt.figure(figsize=(15, 5))
    
    # Subplot 1: Convergence plot
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.semilogy(orders, errors, 'b-o', linewidth=2, markersize=6)
    ax1.axvline(x=max(patch.degrees)+1, color='r', linestyle='--', label=f'Classical p+1 = {max(patch.degrees)+1}')
    ax1.set_xlabel('Quadrature Order (n x n)')
    ax1.set_ylabel('Absolute Error in Area (log scale)')
    ax1.set_title('Integration Error vs Order')
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.legend()

    # Subplot 2: Gauss Points for 2x2
    ax2 = fig.add_subplot(1, 3, 2)
    plot_gauss_points(ax2, patch, 2, 2, "Gauss Points in Physical Space\n(Order 2x2)")

    # Subplot 3: Gauss Points for 4x4
    ax3 = fig.add_subplot(1, 3, 3)
    plot_gauss_points(ax3, patch, 4, 4, "Gauss Points in Physical Space\n(Order 4x4)")

    plt.tight_layout()
    plt.savefig('examples/output/nurbs_quadrature_convergence.png', dpi=200)
    print("Saved updated convergence plot to examples/output/nurbs_quadrature_convergence.png")

if __name__ == '__main__':
    print("Running NURBS Operations Demonstration...")
    demonstrate_operations()
    
    print("\nRunning Quadrature Convergence Test...")
    test_integration_convergence()
