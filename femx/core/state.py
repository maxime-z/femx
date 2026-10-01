from dataclasses import dataclass, field
from typing import Dict
import numpy as np
from femx.backends.numpy_backend import ndarray, zeros
from femx.core.dofs import DofMap

@dataclass
class State:
    """
    Stores field variables and Gauss-point internal states.

    ``gauss_variables`` holds the last *committed* history.
    ``trial_gauss_variables`` holds values written during residual/tangent
    evaluation; they become committed only via ``commit_gauss``.
    """
    # Nodal/control point values: maps field_name to ndarray of shape (n_entities, n_components)
    values: Dict[str, ndarray] = field(default_factory=dict)

    # Committed Gauss-point historical variables: (n_elements, n_gps, n_components)
    gauss_variables: Dict[str, ndarray] = field(default_factory=dict)

    # Trial Gauss-point variables written during residual evaluation
    trial_gauss_variables: Dict[str, ndarray] = field(default_factory=dict)

    def pack_vector(self, dof_map: DofMap) -> ndarray:
        """Pack all unknown field values into a single flat equation vector of size dof_map.n_dofs."""
        u_vec = zeros(dof_map.n_dofs)
        for field_name, offset in dof_map.field_offsets.items():
            spec = dof_map.field_specs[field_name]
            n_ent = dof_map.field_entities[field_name]
            if field_name in self.values:
                val = self.values[field_name]
                u_vec[offset : offset + n_ent * spec.components] = val.ravel()
        return u_vec

    def unpack_vector(self, u_vec: ndarray, dof_map: DofMap):
        """Unpack a flat equation vector into the values dictionary."""
        for field_name, offset in dof_map.field_offsets.items():
            spec = dof_map.field_specs[field_name]
            n_ent = dof_map.field_entities[field_name]
            length = n_ent * spec.components
            flat_val = u_vec[offset : offset + length]
            self.values[field_name] = flat_val.reshape(n_ent, spec.components)

    def restore_nodal(self, u_vec: ndarray, dof_map: DofMap):
        """Restore nodal/control-point values from a previously packed vector."""
        self.unpack_vector(u_vec, dof_map)

    def initialize_field(self, field_name: str, n_entities: int, components: int):
        """Initialize a field value dictionary to zero."""
        self.values[field_name] = zeros((n_entities, components))

    def initialize_gauss_variable(self, var_name: str, n_elements: int, n_gps: int, components: int):
        """Initialize committed and trial Gauss-point variable buffers to zero."""
        self.gauss_variables[var_name] = zeros((n_elements, n_gps, components))
        self.trial_gauss_variables[var_name] = zeros((n_elements, n_gps, components))

    def copy_committed_to_trial(self):
        """Reset trial buffers from the last committed history."""
        self.trial_gauss_variables = {
            name: arr.copy() for name, arr in self.gauss_variables.items()
        }

    def commit_gauss(self):
        """Accept trial Gauss variables as the new committed history."""
        self.gauss_variables = {
            name: arr.copy() for name, arr in self.trial_gauss_variables.items()
        }
