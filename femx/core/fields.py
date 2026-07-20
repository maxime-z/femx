from dataclasses import dataclass

@dataclass(frozen=True)
class FieldSpec:
    """
    Specifies field properties.
    Args:
        name: Name of the field (e.g. 'u', 'T')
        components: Number of component dimensions (e.g. 1 for scalar, 2 for 2D vector)
        location: Entity where field is located ('nodes' or 'control_points')
        unknown: True if it is a solved unknown, False if it is auxiliary state
    """
    name: str
    components: int
    location: str
    unknown: bool = True
