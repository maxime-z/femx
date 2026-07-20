from dataclasses import dataclass, field
from typing import Dict
import numpy as np
from femx.backends.numpy_backend import ndarray, zeros
from femx.core.dofs import DofMap

@dataclass
class State:
    """
    Stores field variables and Gauss-point internal states.
    """
    # Nodal/control point values: maps field_name to ndarray of shape (n_entities, n_components)
    values: Dict[str, ndarray] = field(default_factory=dict)
    
    # Gauss-point historical variables: maps variable_name to ndarray of shape (n_elements, n_gps, n_components)
    gauss_variables: Dict[str, ndarray] = field(default_factory=dict)

    def pack_vector(self, dof_map: DofMap) -> ndarray:
        """Pack all unknown field values into a single flat equation vector of size dof_map.n_dofs."""
        u_vec = zeros(dof_map.n_dofs)
        for field_name, offset in dof_map.field_offsets.items():
            spec = dof_map.field_specs[field_name]
            if field_name in self.values:
                val = self.values[field_name]  # shape (n_entities, n_components)
                # Flatten the field values to match dof numbering: offset + entity * components + comp
                u_vec[offset : offset + dof_map.n_entities * spec.components] = val.ravel()
        return u_vec

    def unpack_vector(self, u_vec: ndarray, dof_map: DofMap):
        """Unpack a flat equation vector into the values dictionary."""
        for field_name, offset in dof_map.field_offsets.items():
            spec = dof_map.field_specs[field_name]
            length = dof_map.n_entities * spec.components
            flat_val = u_vec[offset : offset + length]
            self.values[field_name] = flat_val.reshape(dof_map.n_entities, spec.components)
            
    def initialize_field(self, field_name: str, n_entities: int, components: int):
        """Initialize a field value dictionary to zero."""
        self.values[field_name] = zeros((n_entities, components))
        
    def initialize_gauss_variable(self, var_name: str, n_elements: int, n_gps: int, components: int):
        """Initialize a Gauss point variable dictionary to zero."""
        self.gauss_variables[var_name] = zeros((n_elements, n_gps, components))
