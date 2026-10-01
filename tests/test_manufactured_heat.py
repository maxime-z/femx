"""Manufactured solution for 2D heat conduction on Q1 meshes."""
import numpy as np
from femx.core.mesh import Mesh
from femx.core.fields import FieldSpec
from femx.core.dofs import DofMap
from femx.core.assembly import assemble_system
from femx.materials.linear_heat import LinearHeatMaterial
from femx.formulations.heat import HeatConductionFormulation
from femx.solvers.linear import solve_system


def create_q1_grid(Lx: float, Ly: float, nx: int, ny: int) -> Mesh:
    x = np.linspace(0, Lx, nx + 1)
    y = np.linspace(0, Ly, ny + 1)
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

    left = np.arange(ny + 1) * (nx + 1)
    right = np.arange(ny + 1) * (nx + 1) + nx
    bottom = np.arange(nx + 1)
    top = np.arange(nx + 1) + ny * (nx + 1)
    boundaries = {"left": left, "right": right, "bottom": bottom, "top": top}
    return Mesh(coords=coords, cells=np.array(cells, dtype=int), boundaries=boundaries)


def exact_T(x: float, y: float) -> float:
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def source_f(xyz) -> float:
    # -k * laplace(T) = f with k=1 and T = sin(pi x) sin(pi y)
    return 2.0 * np.pi**2 * exact_T(float(xyz[0]), float(xyz[1]))


def solve_manufactured(nx: int, ny: int):
    mesh = create_q1_grid(1.0, 1.0, nx, ny)
    fields = [FieldSpec(name="T", components=1, location="nodes", unknown=True)]
    dof_map = DofMap(fields=fields, geometry=mesh)
    material = LinearHeatMaterial(rho=1.0, C=1.0, K=1.0)
    formulation = HeatConductionFormulation(material=material)

    K, M, f = assemble_system(dof_map, formulation, field_name="T", body_load=source_f)

    # Homogeneous Dirichlet on the whole boundary (exact T = 0 there)
    dirichlet_bcs = {}
    for name in ("left", "right", "bottom", "top"):
        for node in mesh.boundaries[name]:
            dof = dof_map.get_dof("T", int(node), 0)
            x, y = mesh.coords[node]
            dirichlet_bcs[dof] = exact_T(x, y)

    T_sol = solve_system(K, f, dirichlet_bcs)

    T_exact = np.array([exact_T(x, y) for x, y in mesh.coords])
    # Nodal L2 error (Euclidean / sqrt(n_nodes))
    err = np.linalg.norm(T_sol - T_exact) / np.sqrt(mesh.n_nodes)
    return err


def test_manufactured_heat_refinement():
    err_coarse = solve_manufactured(4, 4)
    err_fine = solve_manufactured(8, 8)
    print(f"Coarse error: {err_coarse}, Fine error: {err_fine}")
    assert err_coarse > 0.0
    assert err_fine < err_coarse
    # Q1 should be roughly O(h^2) in L2; allow loose factor of 3x reduction on 2x refine
    assert err_fine < 0.5 * err_coarse


if __name__ == "__main__":
    test_manufactured_heat_refinement()
