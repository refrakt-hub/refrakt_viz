"""
Contrastive loss curve visualization for contrastive learning models.

This module provides a visualization component for displaying loss curves during
contrastive learning, allowing users to monitor training progress and convergence.

Typical usage:
    from refrakt_viz.contrastive import ContrastiveLossCurvePlot
    viz = ContrastiveLossCurvePlot()
    viz.update(loss)
    viz.save_with_name("model_name")

Classes:
    - ContrastiveLossCurvePlot: Visualize loss curves for contrastive learning.
"""

from __future__ import annotations

import os
from typing import Any, List

import matplotlib.pyplot as plt

from refrakt_viz.base import ContrastiveVisualizationComponent
from refrakt_viz.registry import register_viz


@register_viz("contrastive_loss_curve")
class ContrastiveLossCurvePlot(ContrastiveVisualizationComponent):
    """
    Visualization component for displaying loss curves during contrastive learning.

    This class accumulates loss values and provides methods to save and display loss curves
    for monitoring training progress and convergence.

    Attributes:
        loss_history (List[float]): List of loss values.
        title (str): Title for the visualization.
    """

    def __init__(self) -> None:
        """
        Initialize the ContrastiveLossCurvePlot visualization.
        """
        self.loss_history: List[float] = []
        self.title: str = "Contrastive Loss Curve"

    def update(self, *args, **kwargs) -> None:
        """
        Update the visualization with new data. Expects loss, title as arguments.
        """
        loss = kwargs.get("loss", args[0] if len(args) > 0 else None)
        title = kwargs.get("title", args[1] if len(args) > 1 else None)
        if loss is None:
            raise ValueError("loss must be provided to update()")
        self.loss_history.append(loss)
        if title is not None:
            self.title = title

    def update_from_batch(
        self, model: Any, batch: Any, loss: float, epoch: int
    ) -> None:
        """
        Add a loss value from a batch to the loss history.

        Args:
            model (Any): The contrastive model.
            batch (Any): Input batch (unused).
            loss (float): Loss value to add.
            epoch (int): Current epoch (unused).
        """
        self.update(loss)

    registry_name: str = "contrastive_loss_curve"

    def save_with_name(self, model_name: str = "model", mode: str = "batch") -> None:
        """
        Save the loss curve to a model-specific directory.

        Args:
            model_name (str): Name of the model for directory naming.
            mode (str): Mode for the x-axis ("batch" or "epoch").
        """
        out_dir = f"visualizations/{model_name}"
        os.makedirs(out_dir, exist_ok=True)
        self.save(f"{out_dir}/{self.registry_name}.png", mode=mode)

    def save(self, path: str, mode: str = "batch") -> None:
        """
        Save the loss curve to disk.

        Args:
            path (str): Output file path for the visualization.
            mode (str): Mode for the x-axis ("batch" or "epoch").
        Raises:
            ValueError: If no loss history is available.
        """
        if not self.loss_history:
            raise ValueError("No loss history to visualize. Call update() first.")
        plt.figure(figsize=(8, 5))
        plt.plot(self.loss_history, label="Contrastive Loss", color="blue")
        plt.xlabel("Batch" if mode == "batch" else "Epoch")
        plt.ylabel("Loss")
        plt.title(
            f"Contrastive Loss Curve ({'per batch' if mode == 'batch' else 'per epoch'})"
        )
        plt.legend()
        plt.tight_layout()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        plt.close()
