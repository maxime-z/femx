from abc import ABC, abstractmethod
from typing import ClassVar, Tuple, Optional, Generic, TypeVar
from femx.backends.numpy_backend import ndarray
from femx.materials.base import Material
from femx.basis.element import ElementBasis

T_Material = TypeVar('T_Material', bound=Material)

class Formulation(ABC, Generic[T_Material]):
    """
    Abstract base class for physical problem formulations.
    Parametrized by the specific Material type it requires (Generic[T_Material]).
    """
    # Class-level static typed attribute for unknown field names
    field_names: ClassVar[Tuple[str, ...]]
    
    # Strongly-typed material model instance
    material: T_Material

    def __init__(self, material: T_Material):
        self.material = material

    @abstractmethod
    def compute_element_matrices(
        self,
        elem_coords: ndarray,
        quadrature_pts: ndarray,
        quadrature_wts: ndarray,
        elem_basis: Optional[ElementBasis] = None,
        **kwargs
    ) -> Tuple[ndarray, ndarray, ndarray]:
        """
        Compute element stiffness matrix Ke, mass matrix Me, and load vector fe.
        """
        pass

    @abstractmethod
    def get_physical_tensors(self, geom, device: str = "cpu", dtype = None) -> Tuple[ndarray, ndarray, Optional[ndarray]]:
        """
        Return true physical material tensors for the TensorGalerkin engine.
        """
        pass

    @abstractmethod
    def compute_batch_map(self, geom, tensors, device: str = "cpu", dtype = None):
        """
        Stage I Batch-Map TensorGalerkin contraction.
        Returns:
            K_local: element stiffness matrices (shape (E, n_dofs, n_dofs))
            F_local: element force vectors (shape (E, n_dofs))
        """
        pass
