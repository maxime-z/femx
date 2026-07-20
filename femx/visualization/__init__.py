# femx visualization package

from femx.visualization.matplotlib_vis import (
    plot_mesh,
    plot_boundary_conditions,
    plot_scalar_field_2d,
    plot_nurbs_geometry,
    plot_nurbs_scalar_field_2d
)

try:
    from femx.visualization.pyvista_vis import (
        to_pyvista_grid,
        plot_pyvista,
        export_to_vtk
    )
except ImportError:
    # Allow running without PyVista if only Matplotlib is available/wanted
    to_pyvista_grid = None
    plot_pyvista = None
    export_to_vtk = None
