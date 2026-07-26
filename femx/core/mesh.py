from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from femx.backends.numpy_backend import ndarray, array

@dataclass
class Mesh:
    """Standard FEM Mesh representation."""
    # Coordinates of nodes: array of shape (n_nodes, 2)
    coords: ndarray
    # Cell connectivity: array of shape (n_elements, 4) for Q1 elements
    cells: ndarray
    # Boundary definitions: dict mapping name to array of node indices
    boundaries: Dict[str, ndarray] = field(default_factory=dict)
    
    @property
    def n_nodes(self) -> int:
        return self.coords.shape[0]
        
    @property
    def n_elements(self) -> int:
        return self.cells.shape[0]

    def plot(self, values: ndarray = None, show_nodes: bool = True, boundary_colors: Dict[str, str] = None, title: str = "Mesh Plot", ax = None):
        """Plot the mesh layout or a scalar field defined on it."""
        from femx.visualization.matplotlib_vis import plot_mesh, plot_scalar_field_2d
        if values is not None:
            return plot_scalar_field_2d(self, values, title=title, ax=ax)
        return plot_mesh(self, show_nodes=show_nodes, boundary_colors=boundary_colors, ax=ax)

from femx.geometry.nurbs import NurbsPatch
