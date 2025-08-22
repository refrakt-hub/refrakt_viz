"""
Loss and accuracy curve visualization for supervised learning models.

This module provides a visualization component for displaying loss and accuracy curves during
training, allowing users to monitor model performance over time.

Typical usage:
    from refrakt_viz.supervised import LossAccuracyPlot
    viz = LossAccuracyPlot()
    viz.update(train_loss, val_loss, train_acc, val_acc)
    viz.save("loss_accuracy.png")

Classes:
    - LossAccuracyPlot: Visualize loss and accuracy curves during training.
"""

import os
from typing import Dict, List

import matplotlib.pyplot as plt

from refrakt_viz.base import VisualizationComponent
from refrakt_viz.registry import register_viz
from refrakt_viz.utils.display_image import display_image


@register_viz("loss_accuracy")
class LossAccuracyPlot(VisualizationComponent):
    """
    Visualization component for displaying loss and accuracy curves during training.

    This class accumulates loss and accuracy values for training and validation, and provides
    methods to save and display plots for monitoring model performance.

    Attributes:
        history (Dict[str, List[float]]): Dictionary of loss and accuracy histories.
    """

    def __init__(self) -> None:
        """
        Initialize the LossAccuracyPlot visualization.
        """
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

    def update(
        self, train_loss: float, val_loss: float, train_acc: float, val_acc: float
    ) -> None:
        """
        Add loss and accuracy values for training and validation.

        Args:
            train_loss (float): Training loss value.
            val_loss (float): Validation loss value.
            train_acc (float): Training accuracy value.
            val_acc (float): Validation accuracy value.
        """
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["train_acc"].append(train_acc)
        self.history["val_acc"].append(val_acc)

    def update_from_batch(self, model, batch, loss, epoch):
        """
        Update from a model and batch. (Stub implementation)
        """
        pass

    def save(self, path: str) -> None:
        """
        Save loss and accuracy curves to disk.

        Args:
            path (str): Output file path for the visualization.
        """
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.history["train_loss"], label="Train Loss")
        plt.plot(self.history["val_loss"], label="Val Loss")
        plt.legend()
        plt.title("Loss")
        plt.subplot(1, 2, 2)
        plt.plot(self.history["train_acc"], label="Train Acc")
        plt.plot(self.history["val_acc"], label="Val Acc")
        plt.legend()
        plt.title("Accuracy")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def save_with_name(self, model_name: str, mode: str = "train") -> None:
        """
        Save loss and accuracy curves to a model-specific directory.

        Args:
            model_name (str): Name of the model for directory naming.
            mode (str): Mode for the plot (default: "train").
        """
        out_dir = f"visualizations/{model_name}"
        os.makedirs(out_dir, exist_ok=True)
        self.save(f"{out_dir}/loss_accuracy_{mode}.png")

    def show(self, model_name: str, mode: str = "train") -> None:
        """
        Save and display loss and accuracy curves for a model.

        Args:
            model_name (str): Name of the model for directory naming.
            mode (str): Mode for the plot (default: "train").
        """
        self.save_with_name(model_name, mode)
        img_path = f"visualizations/{model_name}/loss_accuracy_{mode}.png"
        display_image(img_path)
