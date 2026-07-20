class Material:
    """
    Base class representing material models and parameters.
    """
    def __init__(self, **properties):
        self.properties = properties

    def get_property(self, name: str, default: float = None) -> float:
        """Retrieve a material property."""
        return self.properties.get(name, default)
