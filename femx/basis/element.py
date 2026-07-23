from abc import ABC, abstractmethod
from typing import Tuple
from femx.backends.numpy_backend import ndarray

class ElementBasis(ABC):
    """
    Abstract base class for Finite Element and Isogeometric basis functions.
    Handles reference shape functions, derivatives, geometric mappings, and default quadratures.
    """
    
    @property
    @abstractmethod
    def n_dofs_per_element(self) -> int:
        """Number of local shape functions / DoFs per element."""
        pass

    @property
    @abstractmethod
    def dim(self) -> int:
        """Parametric dimension of the element (1, 2, or 3)."""
        pass
        
    @property
    def function_space(self) -> str:
        """Type of function space: 'H1' (nodal), 'Hcurl' (edge), 'Hdiv' (face). Default is 'H1'."""
        return "H1"

    @abstractmethod
    def evaluate_shape_functions(self, ref_coords: ndarray) -> ndarray:
        """Evaluate shape functions at reference coordinates."""
        pass

    @abstractmethod
    def evaluate_shape_derivatives(self, ref_coords: ndarray) -> ndarray:
        """Evaluate derivatives of shape functions with respect to reference coordinates."""
        pass

    @abstractmethod
    def compute_mapping(self, ref_coords: ndarray, elem_coords: ndarray) -> Tuple[ndarray, ndarray, float]:
        """
        Compute reference-to-physical mapping.
        Returns:
            N: Shape function values (shape (n_dofs,))
            dN_dphys: Derivatives wrt physical coordinates (shape (dim, n_dofs))
            detJ: Jacobian determinant (float)
        """
        pass

    @abstractmethod
    def get_default_quadrature(self) -> Tuple[ndarray, ndarray]:
        """
        Return optimal default quadrature points and weights for this element.
        """
        pass
