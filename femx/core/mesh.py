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

@dataclass
class NurbsPatch:
    """2D NURBS Patch representation for IGA."""
    p_u: int  # Degree in u (xi) direction
    p_v: int  # Degree in v (eta) direction
    knots_u: ndarray  # 1D array of knots in u
    knots_v: ndarray  # 1D array of knots in v
    control_points: ndarray  # Control points coordinate array of shape (n_cp_u, n_cp_v, 2)
    weights: ndarray  # Control points weights array of shape (n_cp_u, n_cp_v)
    boundaries: Dict[str, ndarray] = field(default_factory=dict) # e.g. control point indices or boundary edges

    @property
    def n_cp_u(self) -> int:
        return self.control_points.shape[0]

    @property
    def n_cp_v(self) -> int:
        return self.control_points.shape[1]

    @property
    def n_control_points(self) -> int:
        return self.n_cp_u * self.n_cp_v

    def get_element_spans(self) -> List[Tuple[int, int]]:
        """
        Identify active element spans (knot intervals with non-zero length).
        Returns a list of (span_u, span_v) indices.
        """
        spans = []
        # Active spans lie between degree p and m - p - 1
        for i_u in range(self.p_u, len(self.knots_u) - self.p_u - 1):
            for i_v in range(self.p_v, len(self.knots_v) - self.p_v - 1):
                # Non-zero knot span length check
                if (self.knots_u[i_u + 1] > self.knots_u[i_u] and
                    self.knots_v[i_v + 1] > self.knots_v[i_v]):
                    spans.append((i_u, i_v))
        return spans

    def get_element_control_points(self, span_u: int, span_v: int) -> ndarray:
        """
        Get the 1D indices of the control points that support the element at (span_u, span_v).
        The supporting control points are index_u in [span_u - p_u, span_u] and index_v in [span_v - p_v, span_v].
        Returns a flat array of shape ((p_u + 1) * (p_v + 1),) containing 1D control point indices.
        """
        indices = []
        # Loop through active basis functions supporting the span
        for j in range(self.p_v + 1):
            idx_v = span_v - self.p_v + j
            for i in range(self.p_u + 1):
                idx_u = span_u - self.p_u + i
                # Compute 1D index: idx_u + idx_v * n_cp_u
                idx_1d = idx_u + idx_v * self.n_cp_u
                indices.append(idx_1d)
        return array(indices, dtype=int)

    def plot(self, values: ndarray = None, show_control_grid: bool = True, title: str = "NURBS Plot", ax = None):
        """Plot the NURBS patch geometry, control grid, or a scalar field defined on it."""
        from femx.visualization.matplotlib_vis import plot_nurbs_geometry, plot_nurbs_scalar_field_2d
        if values is not None:
            return plot_nurbs_scalar_field_2d(self, values, title=title, ax=ax)
        return plot_nurbs_geometry(self, show_control_grid=show_control_grid, ax=ax)
