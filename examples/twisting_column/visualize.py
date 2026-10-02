"""Off-screen PyVista figures for the twisting column."""
from __future__ import annotations

import os
from typing import Dict

import numpy as np


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_all(result: Dict, output_dir: str):
    try:
        import pyvista as pv
    except ImportError as exc:
        raise RuntimeError("pyvista is required for twisting-column visualization") from exc

    from femx.visualization.pyvista_vis import to_pyvista_grid
    from femx.core.mesh import Mesh

    _ensure_dir(output_dir)
    pv.OFF_SCREEN = True
    geometry = result["geometry"]
    u_field = np.asarray(result["u_field"])
    title = (
        f"{result['formulation']} {result['discretization']} p={result['p']} "
        f"angle {result['angle_reached']:.1f}/{result['angle_requested']:.1f} deg"
    )
    if result.get("failed"):
        title += " (FAILED)"
    if result.get("kinematically_invalid"):
        title += " (kinematically invalid)"

    # Setup: undeformed wireframe
    grid0 = to_pyvista_grid(geometry, values=None)
    pl = pv.Plotter(off_screen=True, window_size=(900, 700))
    pl.add_mesh(grid0, style="wireframe", color="black")
    pl.add_title(f"Setup: {title}")
    pl.show_axes()
    pl.screenshot(os.path.join(output_dir, "setup.png"))
    pl.close()

    # Deformed vs undeformed
    grid = to_pyvista_grid(geometry, values=u_field, field_name="u")
    pl = pv.Plotter(off_screen=True, window_size=(900, 700))
    pl.add_mesh(grid, style="wireframe", color="#adb5bd", opacity=0.4)
    warped = grid.warp_by_vector("u", factor=1.0)
    pl.add_mesh(warped, color="steelblue", show_edges=True, opacity=0.9)
    pl.add_title(f"Deformed: {title}")
    pl.show_axes()
    pl.screenshot(os.path.join(output_dir, "deformed.png"))
    pl.close()

    # Von Mises on deformed solid (Mesh only for now)
    vm = None
    if isinstance(geometry, Mesh) and u_field.shape[1] == 3:
        try:
            if result["formulation"] == "elasticity":
                from femx.materials.linear_elastic import LinearElasticMaterial
                from femx.formulations.elasticity import LinearElasticityFormulation
                from femx.core.postprocessing import element_center_von_mises_elasticity
                mat = LinearElasticMaterial(rho=1.0, E=1.0e5, nu=0.3)
                form = LinearElasticityFormulation(mat, mode="3d")
                vm = element_center_von_mises_elasticity(geometry, form, result["U"])
            else:
                from femx.materials.hyperelastic import NeoHookeanMaterial
                from femx.core.postprocessing import element_center_von_mises_hyperelastic
                mat = NeoHookeanMaterial(rho=1.0, E=1.0e5, nu=0.3)
                vm = element_center_von_mises_hyperelastic(geometry, mat, u_field)
        except Exception as exc:
            print(f"von Mises skipped: {exc}")

    if vm is not None:
        pl = pv.Plotter(off_screen=True, window_size=(900, 700))
        warped = grid.warp_by_vector("u", factor=1.0)
        warped.cell_data["von_Mises"] = vm
        pl.add_mesh(warped, scalars="von_Mises", show_edges=True, cmap="plasma")
        pl.add_title(f"von Mises: {title}")
        pl.show_axes()
        pl.screenshot(os.path.join(output_dir, "von_mises.png"))
        pl.close()

    # Animation over accepted history
    history = result.get("history") or [u_field]
    if len(history) >= 2:
        anim_path = os.path.join(output_dir, "twist.gif")
        pl = pv.Plotter(off_screen=True, window_size=(700, 600))
        pl.open_gif(anim_path)
        for frame, uf in enumerate(history):
            pl.clear()
            g = to_pyvista_grid(geometry, values=np.asarray(uf), field_name="u")
            pl.add_mesh(g, style="wireframe", color="#adb5bd", opacity=0.3)
            pl.add_mesh(g.warp_by_vector("u", factor=1.0), color="crimson", show_edges=True)
            pl.add_title(f"{title} frame {frame}/{len(history)-1}")
            pl.show_axes()
            pl.write_frame()
        pl.close()
        print(f"Wrote animation {anim_path}")

    print(f"Wrote figures under {output_dir}")
