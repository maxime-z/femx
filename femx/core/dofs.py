from typing import List, Union, Dict
import numpy as np
from femx.backends.numpy_backend import ndarray, array
from femx.core.fields import FieldSpec
from femx.core.mesh import Mesh, NurbsPatch

class DofMap:
    """Manages global equation numbering for multiple fields."""
    def __init__(self, fields: List[FieldSpec], geometry: Union[Mesh, NurbsPatch]):
        self.fields = fields
        self.geometry = geometry
        
        # Get entity count
        if isinstance(geometry, Mesh):
            self.n_entities = geometry.n_nodes
        elif isinstance(geometry, NurbsPatch):
            self.n_entities = geometry.n_control_points
        else:
            raise TypeError("Geometry must be Mesh or NurbsPatch")
            
        # Map field name to its properties and offsets
        self.field_offsets: Dict[str, int] = {}
        self.field_specs: Dict[str, FieldSpec] = {}
        
        current_offset = 0
        for field in fields:
            if field.unknown:
                self.field_specs[field.name] = field
                self.field_offsets[field.name] = current_offset
                current_offset += self.n_entities * field.components
                
        self.n_dofs = current_offset

    def get_dof(self, field_name: str, entity_idx: int, component_idx: int) -> int:
        """Get the global equation index for a specific degree of freedom."""
        if field_name not in self.field_specs:
            raise KeyError(f"Field {field_name} is not an unknown field in this DofMap")
            
        spec = self.field_specs[field_name]
        if component_idx >= spec.components:
            raise ValueError(f"Component index {component_idx} out of range for field {field_name}")
            
        offset = self.field_offsets[field_name]
        return offset + entity_idx * spec.components + component_idx

    def get_element_dofs(self, field_name: str, entity_indices: ndarray) -> ndarray:
        """
        Get global DOF indices for a list of element entity IDs.
        For a field with C components and element with E nodes, returns an array of shape (E * C,).
        Ordering: [node0_comp0, node0_comp1, ..., node1_comp0, ...]
        """
        spec = self.field_specs[field_name]
        offset = self.field_offsets[field_name]
        
        # Vectorized calculation
        # entity_indices is of shape (E,)
        # We need to construct a flat array of shape (E * C,)
        C = spec.components
        dofs = np.zeros(len(entity_indices) * C, dtype=int)
        for c in range(C):
            dofs[c::C] = offset + entity_indices * C + c
            
        return dofs

    def get_element_dofs_multi(self, field_names: List[str], entity_indices: ndarray) -> ndarray:
        """
        Get combined global DOF indices for a list of coupled fields in an element.
        Concatenates elemental DOF arrays for each field in order.
        """
        dof_arrays = [self.get_element_dofs(fn, entity_indices) for fn in field_names]
        return np.concatenate(dof_arrays)

    def get_field_dofs(self, field_name: str) -> ndarray:
        """Get all global DOF indices corresponding to a specific field."""
        if field_name not in self.field_specs:
            raise KeyError(f"Field {field_name} is not an unknown field in this DofMap")
        spec = self.field_specs[field_name]
        offset = self.field_offsets[field_name]
        count = self.n_entities * spec.components
        return np.arange(offset, offset + count, dtype=int)

