# refrakt_viz/registry.py
from typing import Callable, Dict, Type, Any

_viz_registry: Dict[str, Type[Any]] = {}


def register_viz(name: str) -> Callable[[Type[Any]], Type[Any]]:
    """
    Decorator to register a visualization component class under a given name.
    """
    def decorator(cls: Type[Any]) -> Type[Any]:
        _viz_registry[name] = cls
        return cls
    
    return decorator


def get_viz(name: str) -> Type[Any]:
    """
    Retrieve a visualization component class by name.
    """
    if name not in _viz_registry:
        raise ValueError(f"Visualization component '{name}' not found in registry.")
    return _viz_registry[name] 