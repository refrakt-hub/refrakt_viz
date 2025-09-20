"""
Visualization registry for refrakt_viz.

This module provides a registry for all visualization components in the Refrakt framework.
It allows users to register, retrieve, and discover available visualizations by name.

Typical usage:
    from refrakt_viz.registry import register_viz, get_viz
    @register_viz("my_viz")
    class MyViz(...):
        ...
    VizClass = get_viz("my_viz")

Functions:
    - register_viz: Decorator to register a visualization component class.
    - get_viz: Retrieve a visualization component class by name.
"""

from typing import Any, Callable, Dict, Type

_viz_registry: Dict[str, Type[Any]] = {}


def register_viz(name: str) -> Callable[[Type[Any]], Type[Any]]:
    """
    Decorator to register a visualization component class under a given name.

    Args:
        name (str): Name to register the visualization under.
    Returns:
        Callable[[Type[Any]], Type[Any]]: Decorator that registers the class.
    """

    def decorator(cls: Type[Any]) -> Type[Any]:
        _viz_registry[name] = cls
        return cls

    return decorator


def get_viz(name: str) -> Type[Any]:
    """
    Retrieve a visualization component class by name.

    Args:
        name (str): Name of the visualization component.
    Returns:
        Type[Any]: The registered visualization component class.
    Raises:
        ValueError: If the visualization component is not found in the registry.
    """
    if name not in _viz_registry:
        raise ValueError(f"Visualization component '{name}' not found in registry.")
    return _viz_registry[name]
