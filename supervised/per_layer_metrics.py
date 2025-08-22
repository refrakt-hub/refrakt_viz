"""
Per-layer metrics visualization for supervised learning models.

This module provides a visualization component for displaying per-layer
activation and gradient metrics, allowing users to monitor internal
model statistics during training.

Typical usage:
    from refrakt_viz.supervised import PerLayerMetricsPlot
    viz = PerLayerMetricsPlot()
    viz.update(layer_metrics, split="train")
    viz.save("per_layer_metrics.png")

Classes:
    - PerLayerMetricsPlot: Visualize per-layer activation and gradient metrics.
"""

from typing import Any, Dict, List

import matplotlib.pyplot as plt

from refrakt_viz.base import VisualizationComponent
from refrakt_viz.registry import register_viz


@register_viz("per_layer_metrics")
class PerLayerMetricsPlot(VisualizationComponent):
    """
    Visualization component for displaying per-layer activation and gradient metrics.

    This class accumulates per-layer metrics and provides methods to save and display plots
    for monitoring internal model statistics during training.

    Attributes:
        metrics_history (List[Dict[str, Any]]): List of per-layer metrics dictionaries.
        split_history (List[str]): List of split names (e.g., "train", "val").
    """

    def __init__(self) -> None:
        """
        Initialize the PerLayerMetricsPlot visualization.
        """
        self.metrics_history: List[Dict[str, Any]] = []
        self.split_history: List[str] = []

    def update(self, *args, **kwargs) -> None:
        """
        Update the visualization with new data. Expects layer_metrics, split as arguments.
        """
        layer_metrics = kwargs.get("layer_metrics", args[0] if len(args) > 0 else None)
        split = kwargs.get("split", args[1] if len(args) > 1 else "train")
        if layer_metrics is None:
            raise ValueError("layer_metrics must be provided to update()")
        self.metrics_history.append(layer_metrics)
        self.split_history.append(split)

    def update_from_batch(self, model, batch, loss, epoch):
        """
        Update from a model and batch. (Stub implementation)
        """
        pass

    def save(self, path: str, mode: str = "train") -> None:
        """
        Save a plot of per-layer metrics for a given split to disk.

        Args:
            path (str): Output file path for the visualization.
            mode (str): Split name to plot (default: "train").
        """
        if not self.metrics_history:
            return
        import os
        from collections import defaultdict

        os.makedirs("visualizations", exist_ok=True)
        split_to_metrics = defaultdict(list)
        for metrics, split in zip(self.metrics_history, self.split_history):
            split_to_metrics[split].append(metrics)
        # Only plot and save for the current mode
        if mode in split_to_metrics:
            metrics_list = split_to_metrics[mode]
            layers = list(metrics_list[0].keys())
            n_layers = len(layers)
            fig, axes = plt.subplots(n_layers, 1, figsize=(8, 2 * n_layers), squeeze=False)
            for idx, layer in enumerate(layers):
                act_means = [h[layer]["activation_mean"] for h in metrics_list]
                grad_means = [h[layer]["grad_mean"] for h in metrics_list]
                ax = axes[idx, 0]
                ax.plot(act_means, label="Activation Mean")
                ax.plot(grad_means, label="Grad Mean")
                ax.set_title(f"{layer} ({mode})")
                ax.set_xlabel("Step")
                ax.set_ylabel("Mean Value")
                ax.legend()
            plt.tight_layout()
            plt.savefig(path)
            print(f"[PerLayerMetricsPlot] Saved per-layer metrics plot: {path}")
            plt.close()

    def save_with_name(self, model_name: str, mode: str = "train") -> None:
        """
        Save the per-layer metrics plot to a model-specific directory.

        Args:
            model_name (str): Name of the model for directory naming.
            mode (str): Split name to plot (default: "train").
        """
        import os

        out_dir = f"visualizations/{model_name}"
        os.makedirs(out_dir, exist_ok=True)
        self.save(f"{out_dir}/per_layer_metric_{mode}.png", mode=mode)

    def show(self, model_name: str, mode: str = "train") -> None:
        """
        Save and display the per-layer metrics plot for a model and split.

        Args:
            model_name (str): Name of the model for directory naming.
            mode (str): Split name to plot (default: "train").
        """
        self.save_with_name(model_name, mode)
        img_path = f"visualizations/{model_name}/per_layer_metric_{mode}.png"
        from refrakt_viz.utils.display_image import display_image

        display_image(img_path)
