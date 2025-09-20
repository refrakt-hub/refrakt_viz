"""
Confusion matrix visualization for supervised learning models.

This module provides a visualization component for displaying confusion matrices for classification
models, including support for multi-class and multi-label problems.

Typical usage:
    from refrakt_viz.supervised import ConfusionMatrixPlot
    viz = ConfusionMatrixPlot(class_names)
    viz.update(y_true, y_pred)
    viz.save("confusion_matrix.png")

Classes:
    - ConfusionMatrixPlot: Visualize confusion matrices for classification.
"""

import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from refrakt_viz.base import VisualizationComponent
from refrakt_viz.registry import register_viz


@register_viz("confusion_matrix")
class ConfusionMatrixPlot(VisualizationComponent):
    """
    Visualization component for displaying confusion matrices for classification models.

    This class accumulates true and predicted labels and provides a method to save a confusion
    matrix for model evaluation.

    Attributes:
        class_names (List[str]): List of class names for labeling axes.
        y_true (List[int]): List of true labels.
        y_pred (List[int]): List of predicted labels.
    """

    def __init__(self, class_names: List[str]) -> None:
        """
        Initialize the ConfusionMatrixPlot visualization.

        Args:
            class_names (List[str]): List of class names for labeling axes.
        """
        self.class_names = class_names
        self.y_true: List[int] = []
        self.y_pred: List[int] = []

    def update(self, *args, **kwargs) -> None:
        """
        Update the visualization with new data. Expects y_true, y_pred as arguments.
        """
        y_true = kwargs.get("y_true", args[0] if len(args) > 0 else None)
        y_pred = kwargs.get("y_pred", args[1] if len(args) > 1 else None)
        if y_true is None or y_pred is None:
            raise ValueError("y_true and y_pred must be provided to update()")
        self.y_true.extend(y_true)
        self.y_pred.extend(y_pred)

    def update_from_batch(self, model, batch, loss, epoch):
        """
        Update the confusion matrix plot from a model and batch.

        Args:
            model: The model used for predictions.
            batch: Input batch for the update.
            loss: Loss value (optional).
            epoch: Current epoch (optional).
        """
        pass

    def save(self, path: str) -> None:
        """
        Save a confusion matrix plot to disk.

        Args:
            path (str): Output file path for the visualization.
        """
        cm = confusion_matrix(self.y_true, self.y_pred)
        _, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(np.arange(len(self.class_names)))
        ax.set_yticks(np.arange(len(self.class_names)))
        ax.set_xticklabels(self.class_names)
        ax.set_yticklabels(self.class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.colorbar(im)
        # Add counts in each cell
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=12,
                )
        plt.tight_layout()
        plt.savefig(path)
        print(f"[ConfusionMatrixPlot] Saved confusion matrix plot: {path}")
        plt.close()

    def save_with_name(self, model_name: str, mode: str = "train") -> None:
        """
        Save the confusion matrix plot to a model-specific directory.

        Args:
            model_name (str): Name of the model for directory naming.
            mode (str): Mode for the plot (default: "train").
        """
        out_dir = f"visualizations/{model_name}"
        os.makedirs(out_dir, exist_ok=True)
        self.save(f"{out_dir}/confusion_matrix_{mode}.png")

    def show(self, model_name: str, mode: str = "test") -> None:
        """
        Save and display the confusion matrix plot for a model and mode.

        Args:
            model_name (str): Name of the model for directory naming.
            mode (str): Mode for the plot (default: "test").
        """
        self.save_with_name(model_name, mode)
        img_path = f"visualizations/{model_name}/confusion_matrix_{mode}.png"
        from refrakt_viz.utils.display_image import display_image
        display_image(img_path)
