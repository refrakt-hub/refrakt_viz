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

# Ensure new visualizations are imported so they are registered
try:
    from refrakt_viz.supervised.per_layer_metrics import PerLayerMetricsPlot
except ImportError:
    pass
try:
    from refrakt_viz.supervised.computation_graph import ComputationGraphPlot
except ImportError:
    pass
# Register contrastive visualizations
try:
    from refrakt_viz.contrastive.embedding_space import EmbeddingSpacePlot
except ImportError:
    pass
try:
    from refrakt_viz.contrastive.pair_similarity import PairSimilarityPlot
except ImportError:
    pass
try:
    from refrakt_viz.contrastive.loss_curve import ContrastiveLossCurvePlot
except ImportError:
    pass
try:
    from refrakt_viz.contrastive.nearest_neighbors import NearestNeighborsPlot
except ImportError:
    pass
try:
    from refrakt_viz.contrastive.cluster_assignments import ClusterAssignmentPlot
except ImportError:
    pass 