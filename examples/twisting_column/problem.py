"""Shared geometry and twist Dirichlet BCs for the 3D twisting column."""
from __future__ import annotations

import argparse
from typing import Dict, Tuple

import numpy as np

from femx.core.mesh import Mesh
from femx.core.dofs import DofMap
from femx.geometry.nurbs import KnotVector, NurbsPatch, degree_elevate, h_refine
from femx.basis.lagrange import LagrangeHex


def default_n_steps(angle_deg: float) -> int:
    """One load step per 15 degrees, at least two steps."""
    return max(2, int(np.ceil(abs(angle_deg) / 15.0)))


def create_lagrange_column(
    nx: int = 1,
    ny: int = 4,
    nz: int = 1,
    p: int = 1,
    Lx: float = 2.0,
    Ly: float = 12.0,
    Lz: float = 2.0,
) -> Mesh:
    """Structured hex column on x,z in [-Lx/2, Lx/2], y in [0, Ly]."""
    if p < 1:
        raise ValueError(p)
    basis = LagrangeHex(p=p)
    n_side = p + 1

    # Global tensor-product nodes for p>1 use the same local ordering as LagrangeHex.
    # For p=1 use VTK hex node ordering inside each cell.
    nx_nodes = p * nx + 1
    ny_nodes = p * ny + 1
    nz_nodes = p * nz + 1
    xs = np.linspace(-0.5 * Lx, 0.5 * Lx, nx_nodes)
    ys = np.linspace(0.0, Ly, ny_nodes)
    zs = np.linspace(-0.5 * Lz, 0.5 * Lz, nz_nodes)

    coords = []
    node_id = {}
    nid = 0
    for k, z in enumerate(zs):
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                node_id[(i, j, k)] = nid
                coords.append([x, y, z])
                nid += 1
    coords = np.asarray(coords, dtype=float)

    cells = []
    for ez in range(nz):
        for ey in range(ny):
            for ex in range(nx):
                if p == 1:
                    i0, j0, k0 = ex, ey, ez
                    nodes = [
                        node_id[(i0, j0, k0)],
                        node_id[(i0 + 1, j0, k0)],
                        node_id[(i0 + 1, j0 + 1, k0)],
                        node_id[(i0, j0 + 1, k0)],
                        node_id[(i0, j0, k0 + 1)],
                        node_id[(i0 + 1, j0, k0 + 1)],
                        node_id[(i0 + 1, j0 + 1, k0 + 1)],
                        node_id[(i0, j0 + 1, k0 + 1)],
                    ]
                else:
                    nodes = []
                    for kk in range(n_side):
                        for jj in range(n_side):
                            for ii in range(n_side):
                                nodes.append(
                                    node_id[(ex * p + ii, ey * p + jj, ez * p + kk)]
                                )
                cells.append(nodes)
    cells = np.asarray(cells, dtype=int)

    bottom = np.array([node_id[(i, 0, k)] for k in range(nz_nodes) for i in range(nx_nodes)], dtype=int)
    top = np.array(
        [node_id[(i, ny_nodes - 1, k)] for k in range(nz_nodes) for i in range(nx_nodes)],
        dtype=int,
    )
    return Mesh(
        coords=coords,
        cells=cells,
        boundaries={"bottom": bottom, "top": top},
    )


def create_nurbs_column(
    nx: int = 1,
    ny: int = 4,
    nz: int = 1,
    degree: int = 1,
    Lx: float = 2.0,
    Ly: float = 12.0,
    Lz: float = 2.0,
) -> NurbsPatch:
    """Degree-1 trivariate B-spline box, then h- and p-refined."""
    from femx.geometry.nurbs import insert_knot

    cps = np.zeros((2, 2, 2, 3), dtype=float)
    xs = [-0.5 * Lx, 0.5 * Lx]
    ys = [0.0, Ly]
    zs = [-0.5 * Lz, 0.5 * Lz]
    for iu, x in enumerate(xs):
        for iv, y in enumerate(ys):
            for iw, z in enumerate(zs):
                cps[iu, iv, iw] = [x, y, z]
    weights = np.ones((2, 2, 2), dtype=float)
    Pw = np.concatenate([cps * weights[..., None], weights[..., None]], axis=-1)
    U = KnotVector([0.0, 0.0, 1.0, 1.0])
    V = KnotVector([0.0, 0.0, 1.0, 1.0])
    W = KnotVector([0.0, 0.0, 1.0, 1.0])
    patch = NurbsPatch.from_weighted_control_points((1, 1, 1), (U, V, W), Pw)

    def refine_dir(patch, direction, n_spans):
        for _ in range(max(0, n_spans - 1)):
            uk, _ = patch.knot_vectors[direction].unique_knots()
            lengths = np.diff(uk)
            i = int(np.argmax(lengths))
            mid = 0.5 * (uk[i] + uk[i + 1])
            patch = insert_knot(patch, direction, mid, r=1)
        return patch

    patch = refine_dir(patch, 0, nx)
    patch = refine_dir(patch, 1, ny)
    patch = refine_dir(patch, 2, nz)

    elev = max(0, degree - 1)
    for _ in range(elev):
        for d in range(3):
            patch = degree_elevate(patch, d, 1)

    flat = patch.flatten_control_points()
    y = flat[:, 1]
    y0, y1 = float(y.min()), float(y.max())
    bottom = np.where(np.abs(y - y0) < 1e-10)[0]
    top = np.where(np.abs(y - y1) < 1e-10)[0]
    return NurbsPatch(
        degrees=patch.degrees,
        knot_vectors=patch.knot_vectors,
        control_points=patch.control_points,
        weights=patch.weights,
        boundaries={"bottom": bottom, "top": top},
    )


def twist_displacement(xyz: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rigid rotation about the y-axis through the origin in the x-z plane."""
    x, y, z = xyz
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    x_new = c * x - s * z
    z_new = s * x + c * z
    return np.array([x_new - x, 0.0, z_new - z])


def build_dirichlet_bcs(
    dof_map: DofMap,
    geometry,
    angle_deg: float,
    load_factor: float = 1.0,
) -> Dict[int, float]:
    """Clamp y=0; rotate u_x,u_z on y=Ly; leave u_y free on the top."""
    angle = np.deg2rad(angle_deg) * load_factor
    if isinstance(geometry, Mesh):
        coords = geometry.coords
        bottom = geometry.boundaries["bottom"]
        top = geometry.boundaries["top"]
    else:
        coords = geometry.flatten_control_points()
        bottom = geometry.boundaries["bottom"]
        top = geometry.boundaries["top"]

    bcs: Dict[int, float] = {}
    for node in bottom:
        for c in range(3):
            bcs[dof_map.get_dof("u", int(node), c)] = 0.0
    for node in top:
        du = twist_displacement(coords[int(node)], angle)
        bcs[dof_map.get_dof("u", int(node), 0)] = float(du[0])
        bcs[dof_map.get_dof("u", int(node), 2)] = float(du[2])
    return bcs


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--angle", type=float, default=15.0, help="Twist angle in degrees")
    parser.add_argument("--n-steps", type=int, default=None, help="Load steps (default: one per 15 deg)")
    parser.add_argument("--nx", type=int, default=2)
    parser.add_argument("--ny", type=int, default=16)
    parser.add_argument("--nz", type=int, default=2)
    parser.add_argument("--p", type=int, default=1, help="Lagrange/NURBS polynomial degree")
    parser.add_argument("--discretization", choices=("lagrange", "nurbs"), default="lagrange")
    parser.add_argument("--output-dir", type=str, default=None)
    return parser


def make_geometry(args) -> Tuple[object, str]:
    if args.discretization == "lagrange":
        return create_lagrange_column(args.nx, args.ny, args.nz, p=args.p), "lagrange"
    return create_nurbs_column(args.nx, args.ny, args.nz, degree=args.p), "nurbs"
