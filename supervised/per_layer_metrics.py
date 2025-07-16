from typing import Dict, List, Any
import matplotlib.pyplot as plt
from refrakt_viz.registry import register_viz
from .base import VisualizationComponent

@register_viz("per_layer_metrics")
class PerLayerMetricsPlot(VisualizationComponent):
    def __init__(self) -> None:
        self.metrics_history: List[Dict[str, Any]] = []

    def update(self, layer_metrics: Dict[str, Dict[str, float]]) -> None:
        # layer_metrics: {layer_name: {"activation_mean": float, "grad_mean": float, ...}}
        self.metrics_history.append(layer_metrics)

    def save(self, path: str) -> None:
        if not self.metrics_history:
            return
        # Plot mean activation and gradient for each layer over time
        layers = list(self.metrics_history[0].keys())
        num_layers = len(layers)
        fig, axes = plt.subplots(num_layers, 1, figsize=(8, 2 * num_layers))
        if num_layers == 1:
            axes = [axes]
        for i, layer in enumerate(layers):
            act_means = [h[layer]["activation_mean"] for h in self.metrics_history]
            grad_means = [h[layer]["grad_mean"] for h in self.metrics_history]
            axes[i].plot(act_means, label="Activation Mean")
            axes[i].plot(grad_means, label="Grad Mean")
            axes[i].set_title(layer)
            axes[i].legend()
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def show(self) -> None:
        self.save("/tmp/per_layer_metrics.png")
        img = plt.imread("/tmp/per_layer_metrics.png")
        plt.imshow(img)
        plt.axis('off')
        plt.show() 