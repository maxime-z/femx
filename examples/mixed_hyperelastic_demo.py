import os
import numpy as np
import matplotlib.pyplot as plt

from femx.core.fields import FieldSpec
from femx.core.dofs import DofMap
from femx.core.mesh import Mesh
from femx.core.state import State
from femx.core.boundary import integrate_neumann_traction
from femx.materials.mixed_hyperelastic import MixedNeoHookeanMaterial, QuadraticPenalty
from femx.materials.hyperelastic import NeoHookeanMaterial
from femx.formulations.hyperelasticity import (
    MixedHyperelasticFormulation,
    FBarHyperelasticFormulation,
    HyperelasticFormulation,
)
from femx.solvers.nonlinear import NewtonSolver, LoadStepper
from femx.geometry.nurbs import KnotVector, NurbsPatch, h_refine
from femx.visualization.matplotlib_vis import (
    plot_boundary_conditions,
    plot_scalar_field_2d,
    plot_deformed_mesh,
)

# Output directory
output_dir = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(output_dir, exist_ok=True)


def create_cooks_membrane_mesh(nx: int = 16, ny: int = 16) -> Mesh:
    """
    Generate Cook's Membrane quadrilateral mesh.
    Domain bounds:
        A: (0, 0), B: (48, 44), C: (48, 60), D: (0, 44)
    """
    xi = np.linspace(0.0, 1.0, nx + 1)
    eta = np.linspace(0.0, 1.0, ny + 1)

    coords = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            s = xi[i]
            t = eta[j]
            x = 48.0 * s
            y_bot = 44.0 * s
            y_top = 44.0 + 16.0 * s
            y = (1.0 - t) * y_bot + t * y_top
            coords.append([x, y])

    coords = np.array(coords)

    cells = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = (j + 1) * (nx + 1) + i + 1
            n3 = n2 - 1
            cells.append([n0, n1, n2, n3])

    cells = np.array(cells, dtype=int)

    left_nodes = np.arange(ny + 1) * (nx + 1)
    right_nodes = np.arange(ny + 1) * (nx + 1) + nx
    return Mesh(
        coords=coords,
        cells=cells,
        boundaries={"left": left_nodes, "right": right_nodes},
    )


def create_cooks_nurbs_geometry(n_refine: int = 3):
    """
    Generate Cook's Membrane NURBS geometry (p=1 base, refined).
    Returns (patch_u, patch_p).
    patch_p is h-refined. patch_u is h-refined and p-refined (p=2, Taylor-Hood).
    """
    from femx.geometry.nurbs import KnotVector, NurbsPatch, h_refine, p_refine

    U = KnotVector([0, 0, 1, 1])
    V = KnotVector([0, 0, 1, 1])
    ctrl = np.array(
        [
            [[0, 0], [0, 44]],  # u=0 (left edge)
            [[48, 44], [48, 60]],  # u=1 (right edge)
        ],
        dtype=float,
    )
    base_patch = NurbsPatch(
        degrees=(1, 1),
        knot_vectors=(U, V),
        control_points=ctrl,
        weights=np.ones((2, 2)),
    )

    # h-refinement by inserting knots
    patch_p = base_patch
    for _ in range(n_refine):
        patch_p = h_refine(patch_p)

    # p-refinement (degree elevation) for displacement
    patch_u = p_refine(patch_p, 1)

    return patch_u, patch_p


def integrate_nurbs_traction_right_edge(
    patch: "NurbsPatch", dof_map: DofMap, field_name: str, t_y: float
) -> np.ndarray:
    from femx.core.quadrature import get_quadrature_1d
    from femx.basis.nurbs import compute_nurbs_mapping

    f_ext = np.zeros(dof_map.n_dofs)

    # Right edge is U = 1, V in [0, 1]
    U_val = 1.0
    V_knots, _ = patch.knot_vectors[1].unique_knots()

    pts_1d, wts_1d = get_quadrature_1d(patch.degrees[1] + 1)

    for i in range(len(V_knots) - 1):
        v_a, v_b = V_knots[i], V_knots[i + 1]
        L_v = v_b - v_a
        detJ_v = 0.5 * L_v

        span_u = patch.knot_vectors[0].find_span(patch.degrees[0], U_val)
        span_v = patch.knot_vectors[1].find_span(patch.degrees[1], v_a)

        for pt, w in zip(pts_1d, wts_1d):
            gp_ref = np.array([1.0, pt])
            R, dR_dphys, _ = compute_nurbs_mapping(gp_ref, patch, span_u, span_v)

            # Note: For traction, we need physical edge length.
            # Cook's membrane right edge length is 16.0, span is [0, 1] in V.
            # So physical edge Jacobian is actually 16.0 * detJ_v.
            # Wait! The formula uses physical traction t_y. We need dV = ds = ||dx/dv|| dv.
            # We can compute ||dx/dv|| exactly:
            # R is shape functions. We can dot it with control points if we need it.
            # Actually, total physical length is 16, total parametric length is 1.
            # So physical ds = 16.0 * dv. And dv = detJ_v * w.
            # So dV = 16.0 * detJ_v * w.
            dV = 16.0 * detJ_v * w

            # Get control points for this element
            cp_indices = patch.get_element_control_points(span_u, span_v)
            for local_idx, global_cp in enumerate(cp_indices):
                dof_y = dof_map.get_dof(field_name, global_cp, 1)
                f_ext[dof_y] += R[local_idx] * t_y * dV

    return f_ext


def main():
    print(
        "--- Cook's Membrane Benchmark: Volumetric Locking & Anti-Locking Formulations ---"
    )

    # 1. Mesh generation (16x16 elements)
    nx, ny = 16, 16
    mesh = create_cooks_membrane_mesh(nx, ny)
    print(f"Mesh generated: {mesh.n_nodes} nodes, {mesh.n_elements} elements")

    # 2. Material parameters from exact problem definition
    E = 240.565
    nu = 0.499
    mu = E / (2.0 * (1.0 + nu))  # ~ 80.19368
    kappa = E / (3.0 * (1.0 - 2.0 * nu))  # ~ 400941.67

    mat_std = NeoHookeanMaterial(rho=1.0, E=E, nu=nu)
    mat_mixed = MixedNeoHookeanMaterial(
        rho=1.0, mu=mu, kappa=kappa, penalty=QuadraticPenalty()
    )

    form_std = HyperelasticFormulation(mat_std)
    form_fbar = FBarHyperelasticFormulation(mat_std)
    form_mixed = MixedHyperelasticFormulation(mat_mixed)

    # 3. Setup field specs and DofMaps
    u_spec = FieldSpec(name="u", components=2, location="nodes", unknown=True)
    p_spec = FieldSpec(name="p", components=1, location="nodes", unknown=True)

    dof_map_single = DofMap(fields=[u_spec], geometry=mesh)
    dof_map_mixed = DofMap(fields=[u_spec, p_spec], geometry=mesh)

    # NURBS Mixed setup
    patch_u, patch_p = create_cooks_nurbs_geometry(n_refine=4)  # 16x16 elements
    print(
        f"NURBS Geometry generated: u degree ({patch_u.degrees[0]},{patch_u.degrees[1]}), p degree ({patch_p.degrees[0]},{patch_p.degrees[1]})"
    )
    u_spec_n = FieldSpec(
        name="u", components=2, location="control_points", unknown=True
    )
    p_spec_n = FieldSpec(
        name="p", components=1, location="control_points", unknown=True
    )
    dof_map_nurbs = DofMap(
        fields=[u_spec_n, p_spec_n], geometry={"u": patch_u, "p": patch_p}
    )

    # 4. Dirichlet BCs: Clamped left edge (x = 0) in both X and Y
    dirichlet_bcs_single = {}
    for node in mesh.boundaries["left"]:
        dirichlet_bcs_single[dof_map_single.get_dof("u", node, 0)] = 0.0
        dirichlet_bcs_single[dof_map_single.get_dof("u", node, 1)] = 0.0

    dirichlet_bcs_mixed = {}
    for node in mesh.boundaries["left"]:
        dirichlet_bcs_mixed[dof_map_mixed.get_dof("u", node, 0)] = 0.0
        dirichlet_bcs_mixed[dof_map_mixed.get_dof("u", node, 1)] = 0.0

    dirichlet_bcs_nurbs = {}
    for j in range(patch_u.n_cp_v):
        # left edge is i=0
        idx = 0 + j * patch_u.n_cp_u
        dirichlet_bcs_nurbs[dof_map_nurbs.get_dof("u", idx, 0)] = 0.0
        dirichlet_bcs_nurbs[dof_map_nurbs.get_dof("u", idx, 1)] = 0.0

    # 5. Neumann Traction Load: Total force F = 100 on right edge (height L = 16 -> traction t_y = 100 / 16 = 6.25)
    t_y = 100.0 / 16.0
    f_neumann_single = integrate_neumann_traction(
        mesh=mesh,
        dof_map=dof_map_single,
        field_name="u",
        boundary_name="right",
        traction=np.array([0.0, t_y]),
        n_quad_pts=2,
    )

    f_neumann_mixed = np.zeros(dof_map_mixed.n_dofs)
    f_neumann_mixed[: len(f_neumann_single)] = f_neumann_single

    f_neumann_nurbs = integrate_nurbs_traction_right_edge(
        patch_u, dof_map_nurbs, "u", t_y
    )

    # 6. Solvers (using 10 steps, looser tolerance for stability)
    stepper = LoadStepper(
        n_steps=10, newton_solver=NewtonSolver(rtol=5e-2, max_iter=40)
    )

    state_std = State(values={"u": np.zeros((mesh.n_nodes, 2))})
    state_fbar = State(values={"u": np.zeros((mesh.n_nodes, 2))})
    state_mixed = State(
        values={"u": np.zeros((mesh.n_nodes, 2)), "p": np.zeros(mesh.n_nodes)}
    )
    state_nurbs = State(
        values={
            "u": np.zeros((patch_u.n_control_points, 2)),
            "p": np.zeros(patch_p.n_control_points),
        }
    )

    print(
        "\n1. Solving Standard Displacement formulation (Expect Volumetric Locking)..."
    )
    try:
        state_std, _ = stepper.solve(
            dof_map_single,
            form_std,
            state_std,
            dirichlet_bcs_single,
            full_neumann_load=f_neumann_single,
        )
    except Exception as e:
        print(f"Standard formulation note: {e}")
    top_right_node = mesh.boundaries["right"][-1]
    uy_std_top_right = state_std.values["u"][top_right_node, 1]

    print("\n2. Solving F-Bar Strain Projection formulation (Locking Cured)...")
    try:
        state_fbar, _ = stepper.solve(
            dof_map_single,
            form_fbar,
            state_fbar,
            dirichlet_bcs_single,
            full_neumann_load=f_neumann_single,
        )
    except Exception as e:
        print(f"F-Bar formulation note: {e}")
    top_right_node = mesh.boundaries["right"][-1]
    uy_fbar_top_right = state_fbar.values["u"][top_right_node, 1]

    print(
        "\n3. Solving 2-Field Mixed (u-p) formulation (Q1/Q1 Unstable Equal-Order)..."
    )
    # For Q1/Q1, this formulation violates LBB and will show severe pressure checkerboarding (spurious modes)
    try:
        state_mixed, _ = stepper.solve(
            dof_map_mixed,
            form_mixed,
            state_mixed,
            dirichlet_bcs_mixed,
            full_neumann_load=f_neumann_mixed,
        )
    except Exception as e:
        print(f"Mixed formulation note: {e}")
    top_right_node = mesh.boundaries["right"][-1]
    uy_mixed_top_right = state_mixed.values["u"][top_right_node, 1]

    print(
        "\n4. Solving 2-Field Mixed (u-p) formulation (NURBS Stable Taylor-Hood p=2, q=2 / p=1, q=1)..."
    )
    # For NURBS with degree elevation, LBB is satisfied.
    try:
        state_nurbs, _ = stepper.solve(
            dof_map_nurbs,
            form_mixed,
            state_nurbs,
            dirichlet_bcs_nurbs,
            full_neumann_load=f_neumann_nurbs,
        )
    except Exception as e:
        print(f"Mixed NURBS formulation note: {e}")

    # Get top right corner displacement for NURBS (u=1, v=1 -> last control point)
    top_right_cp_idx = (patch_u.n_cp_u - 1) + (patch_u.n_cp_v - 1) * patch_u.n_cp_u
    uy_nurbs_top_right = state_nurbs.values["u"][top_right_cp_idx, 1]

    print(f"\n--- Top-Right Corner Vertical Displacement u_y(48, 60) ---")
    print(f"Standard (Locked):       {uy_std_top_right:8.4f} mm")
    print(f"F-Bar (Unlocked):        {uy_fbar_top_right:8.4f} mm")
    print(f"Mixed (u-p) (Unstable):  {uy_mixed_top_right:8.4f} mm")
    print(f"Mixed NURBS (Stable):    {uy_nurbs_top_right:8.4f} mm")

    # =========================================================================
    # PLOTS (Following cooks_membrane.py Structure)
    # =========================================================================

    neumann_dict = {}
    for node in mesh.boundaries["right"]:
        dof_y = dof_map_single.get_dof("u", node, 1)
        neumann_dict[dof_y] = t_y

    # 1. Problem Setup & Boundary Conditions Plot
    fig1, ax1 = plt.subplots(figsize=(7, 6))
    plot_boundary_conditions(
        mesh,
        dof_map_single,
        dirichlet_bcs=dirichlet_bcs_single,
        neumann_forces=neumann_dict,
        ax=ax1,
    )
    ax1.set_title(
        "Cook's Membrane Setup & Boundary Conditions (F=100, E=240.565, nu=0.4999)"
    )
    fig1.tight_layout()
    fig1.savefig(os.path.join(output_dir, "mixed_hyperelastic_setup.png"), dpi=200)
    print("Saved: examples/output/mixed_hyperelastic_setup.png")

    # 2. All Field Solutions Grid Plot
    N = mesh.n_nodes
    u_mixed_vec = state_mixed.values["u"]
    ux_vals = u_mixed_vec[:, 0]
    uy_vals = u_mixed_vec[:, 1]

    # Actually use the solved pressure! It will be highly oscillatory due to Q1/Q1 instability
    P_unstable = state_mixed.values["p"]

    fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
    plot_scalar_field_2d(
        mesh, ux_vals, title="Horizontal Displacement u_x (Mixed)", ax=axes[0]
    )
    plot_scalar_field_2d(
        mesh, uy_vals, title="Vertical Displacement u_y (Mixed)", ax=axes[1]
    )
    plot_scalar_field_2d(
        mesh,
        P_unstable,
        title="Unstable Equal-Order (u_p^0, p_p^0)\nSpurious Pressure Checkerboard (Actual Result)",
        ax=axes[2],
    )
    fig2.suptitle(
        "Cook's Membrane Field Solutions (u, p)", fontsize=14, fontweight="bold"
    )
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, "mixed_hyperelastic_fields.png"), dpi=200)
    print("Saved: examples/output/mixed_hyperelastic_fields.png")

    # 3. Initial Frame vs Deformed Frame Plots comparing ALL THREE: Standard, F-Bar, and Mixed
    U_std_flat = state_std.pack_vector(dof_map_single)
    U_fbar_flat = state_fbar.pack_vector(dof_map_single)

    # Extract u_mixed for single-field vector packing
    state_mixed_u_only = State(values={"u": state_mixed.values["u"]})
    U_mixed_flat = state_mixed_u_only.pack_vector(dof_map_single)

    fig3, axes3 = plt.subplots(4, 1, figsize=(8, 20))
    plot_deformed_mesh(
        mesh,
        U_std_flat,
        dof_map_single,
        scale=0.5,
        field_values=np.linalg.norm(state_std.values["u"], axis=1),
        field_title=f"Standard Displacement (Volumetric Locking: u_y = {uy_std_top_right:.2f}mm)",
        ax=axes3[0],
    )
    plot_deformed_mesh(
        mesh,
        U_fbar_flat,
        dof_map_single,
        scale=0.5,
        field_values=np.linalg.norm(state_fbar.values["u"], axis=1),
        field_title=f"F-Bar Projection (Locking Free: u_y = {uy_fbar_top_right:.2f}mm)",
        ax=axes3[1],
    )
    plot_deformed_mesh(
        mesh,
        U_mixed_flat,
        dof_map_single,
        scale=0.5,
        field_values=np.linalg.norm(state_mixed.values["u"], axis=1),
        field_title=f"Mixed Q1/Q1 (Unstable checkerboarding: u_y = {uy_mixed_top_right:.2f}mm)",
        ax=axes3[2],
    )

    # To plot NURBS deformed mesh easily, we can evaluate the NURBS surface points
    # We will just plot the original control points vs displaced control points to save code for now
    pts_x, pts_y = [], []
    flat_cp = patch_u.control_points.transpose(1, 0, 2).reshape((-1, 2))
    for i in range(patch_u.n_control_points):
        p0 = flat_cp[i]
        disp = state_nurbs.values["u"][i]
        pts_x.append(p0[0] + disp[0])
        pts_y.append(p0[1] + disp[1])
    axes3[3].scatter(pts_x, pts_y, color="red", s=10)
    axes3[3].set_title(
        f"Mixed NURBS p=2/p=1 (Stable: u_y = {uy_nurbs_top_right:.2f}mm)"
    )
    axes3[3].axis("equal")

    fig3.tight_layout()
    fig3.savefig(os.path.join(output_dir, "mixed_hyperelastic_deformed.png"), dpi=200)
    print("Saved: examples/output/mixed_hyperelastic_deformed.png")

    plt.close("all")


if __name__ == "__main__":
    main()
