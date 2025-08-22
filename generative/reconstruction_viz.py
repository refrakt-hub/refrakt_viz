"""
Reconstruction visualization for generative models.

This module provides a visualization component for displaying input and reconstruction pairs
from generative models, allowing users to visually assess reconstruction quality.

Typical usage:
    from refrakt_viz.generative import ReconstructionViz
    viz = ReconstructionViz(n_samples=8, title="Reconstructions")
    viz.update(inputs, recons)
    viz.save("reconstructions.png")

Classes:
    - ReconstructionViz: Visualize input and reconstruction pairs.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from refrakt_viz.base import GenerativeVisualizationComponent
from refrakt_viz.registry import register_viz
from refrakt_viz.utils.reconstruction_viz_utils import (
    plot_reconstruction_grid,
    reshape_image,
)


@register_viz("reconstruction_viz")
class ReconstructionViz(GenerativeVisualizationComponent):
    """
    Visualization component for displaying input and reconstruction pairs from generative models.

    This class accumulates input and reconstruction batches and provides a method to save a grid
    of input/reconstruction pairs to disk. Useful for monitoring reconstruction quality during
    training or evaluation.

    Attributes:
        n_samples (int): Number of samples to display.
        title (str): Title for the visualization.
        inputs (List[np.ndarray[Any, Any]]): List of input batches.
        recons (List[np.ndarray[Any, Any]]): List of reconstruction batches.
    """

    registry_name: str = "reconstruction_viz"

    def __init__(
        self, n_samples: int = 8, title: str = "Reconstruction Visualization"
    ) -> None:
        """
        Initialize the ReconstructionViz visualization.

        Args:
            n_samples (int): Number of samples to display.
            title (str): Title for the visualization.
        """
        self.n_samples: int = n_samples
        self.title: str = title
        self.inputs: List[np.ndarray[Any, Any]] = []
        self.recons: List[np.ndarray[Any, Any]] = []

    def update(self, *args, **kwargs) -> None:
        """
        Update the visualization with new data. Expects inputs, recons, title as arguments.
        """
        inputs = kwargs.get("inputs", args[0] if len(args) > 0 else None)
        recons = kwargs.get("recons", args[1] if len(args) > 1 else None)
        title = kwargs.get("title", args[2] if len(args) > 2 else None)
        if inputs is None or recons is None:
            raise ValueError("inputs and recons must be provided to update()")
        self.inputs.append(np.array(inputs))
        self.recons.append(np.array(recons))
        if title is not None:
            self.title = title

    def update_from_batch(
        self, model: Any, batch: Any, loss: float, epoch: int
    ) -> None:
        """
        Add a batch of inputs and reconstructions from a model and batch.

        Args:
            model (Any): The generative model.
            batch (Any): Input batch for reconstruction.
            loss (float): Loss value (unused).
            epoch (int): Current epoch (unused).
        """
        inputs, recons = model.get_inputs_and_recons(batch)
        self.update(inputs, recons)

    def _reshape_image(self, img: Any) -> np.ndarray[Any, Any]:
        """
        Reshape an image for visualization (delegates to utils).

        Args:
            img (Any): The image to reshape.
        Returns:
            np.ndarray[Any, Any]: Reshaped image as np.ndarray.
        """
        return reshape_image(img)

    def save(self, path: str) -> None:
        """
        Save a grid of input and reconstruction pairs to disk.

        Args:
            path (str): Output file path for the visualization.
        """
        if len(self.inputs) == 0 or len(self.recons) == 0:
            print("[ReconstructionViz] No inputs or reconstructions to save.")
            return
        inputs = np.concatenate(self.inputs, axis=0)
        recons = np.concatenate(self.recons, axis=0)
        n_samples = min(self.n_samples, len(inputs))
        plot_reconstruction_grid(inputs, recons, n_samples, self.title, path)
        self.inputs.clear()
        self.recons.clear()
