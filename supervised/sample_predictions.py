"""
Sample predictions visualization for supervised learning models.

This module provides a visualization component for displaying sample predictions and true labels,
allowing users to inspect model predictions visually.

Typical usage:
    from refrakt_viz.supervised import SamplePredictionsPlot
    viz = SamplePredictionsPlot(class_names)
    viz.update(images, y_true, y_pred)
    viz.save("sample_predictions.png")

Classes:
    - SamplePredictionsPlot: Visualize sample predictions and true labels.
"""

from __future__ import annotations

import os
from typing import Any, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np

from refrakt_viz.base import VisualizationComponent

from ..registry import register_viz


@register_viz("sample_predictions")
class SamplePredictionsPlot(VisualizationComponent):
    """
    Visualization component for displaying sample predictions and true labels.

    This class accumulates image/label/prediction tuples and provides a method to save a grid
    of sample predictions to disk. Useful for visually inspecting model predictions.

    Attributes:
        samples (List[Tuple[np.ndarray[Any, Any], int, int]]):
            List of (image, true label, predicted label) tuples.
        class_names (List[str]): List of class names for labeling.
    """

    def __init__(self, class_names: List[str]) -> None:
        """
        Initialize the SamplePredictionsPlot visualization.

        Args:
            class_names (List[str]): List of class names for labeling.
        """
        self.samples: List[Tuple[np.ndarray[Any, Any], int, int]] = []
        self.class_names: List[str] = class_names

    def update(self, *args, **kwargs) -> None:
        """
        Update the visualization with new data. Expects images, y_true, y_pred as arguments.
        """
        images = kwargs.get("images", args[0] if len(args) > 0 else None)
        y_true = kwargs.get("y_true", args[1] if len(args) > 1 else None)
        y_pred = kwargs.get("y_pred", args[2] if len(args) > 2 else None)
        if images is None or y_true is None or y_pred is None:
            raise ValueError("images, y_true, and y_pred must be provided to update()")
        for img, t, p in zip(images, y_true, y_pred):
            self.samples.append((img, t, p))

    def update_from_batch(self, model, batch, loss, epoch):
        """
        Update from a model and batch. (Stub implementation)
        """
        pass

    def save_with_name(self, model_name: str, mode: str = "test") -> None:
        """
        Save the sample predictions plot to a model-specific directory.

        Args:
            model_name (str): Name of the model for directory naming.
            mode (str): Mode for the plot (default: "test").
        """
        out_dir = f"visualizations/{model_name}"
        os.makedirs(out_dir, exist_ok=True)
        self.save(f"{out_dir}/sample_predictions_{mode}.png")

    def save(self, path: str, n: int = 8) -> None:
        """
        Save a grid of sample predictions to disk.

        Args:
            path (str): Output file path for the visualization.
            n (int): Number of samples to display (default: 8).
        """
        n = min(n, len(self.samples))
        _, axes = plt.subplots(1, n, figsize=(n * 2, 2))
        for i, (img, t, p) in enumerate(self.samples[:n]):
            axes[i].imshow(img)
            axes[i].text(
                0.5,
                -0.15,
                f"Pred: {self.class_names[p]}",
                fontsize=10,
                color="red",
                ha="center",
                va="top",
                transform=axes[i].transAxes,
            )
            axes[i].set_title(f"T:{self.class_names[t]}")
            axes[i].axis("off")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def show(self, model_name: str) -> None:
        """
        Save and display the sample predictions plot for a model.

        Args:
            model_name (str): Name of the model for directory naming.
        """
        self.save_with_name(model_name)
        img_path = f"visualizations/{model_name}/sample_predictions_test.png"
        from refrakt_viz.utils.display_image import display_image

        display_image(img_path)
