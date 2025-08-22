"""
Disentanglement analysis visualization for generative models.

This module provides a visualization component for analyzing disentanglement in the latent space
of generative models, allowing users to visualize the effect of varying individual latent dimensions.

Typical usage:
    from refrakt_viz.generative import DisentanglementAnalysis
    viz = DisentanglementAnalysis(dim_range=3.0, steps=7, title="Disentanglement")
    viz.update(model, z_base)
    viz.save("disentanglement.png")

Classes:
    - DisentanglementAnalysis: Visualize disentanglement in latent space.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np

from refrakt_viz.base import GenerativeVisualizationComponent
from refrakt_viz.registry import register_viz
from refrakt_viz.utils.disentanglement_analysis_utils import (
    infer_expected_shape,
    plot_disentanglement_grid,
    reshape_img,
    traverse_latent_dim,
)


@register_viz("disentanglement_analysis")
class DisentanglementAnalysis(GenerativeVisualizationComponent):
    """
    Visualization component for disentanglement analysis in generative models.

    This class accumulates (model, z_base) pairs and provides a method to visualize the effect
    of varying each latent dimension independently, helping to assess disentanglement in the model's
    latent space.

    Attributes:
        dim_range (float): Range of values to traverse for each latent dimension.
        steps (int): Number of steps to traverse per dimension.
        title (str): Title for the visualization.
        pairs (List[Tuple[Any, np.ndarray[Any, Any]]]): List of (model, z_base) pairs for analysis.
    """

    registry_name: str = "disentanglement_analysis"

    def __init__(
        self,
        dim_range: float = 3.0,
        steps: int = 7,
        title: str = "Disentanglement Analysis",
    ) -> None:
        """
        Initialize the DisentanglementAnalysis visualization.

        Args:
            dim_range (float): Range of values to traverse for each latent dimension.
            steps (int): Number of steps to traverse per dimension.
            title (str): Title for the visualization.
        """
        self.dim_range: float = dim_range
        self.steps: int = steps
        self.title: str = title
        self.pairs: List[Tuple[Any, np.ndarray[Any, Any]]] = []

    def update(self, model: Any, z_base: Any) -> None:
        """
        Add a (model, z_base) pair for disentanglement analysis.

        Args:
            model (Any): The generative model.
            z_base (Any): Base latent vector for analysis.
        """
        self.pairs.append((model, np.array(z_base)))

    def update_from_batch(
        self, model: Any, batch: Any, loss: float, epoch: int
    ) -> None:
        """
        Add a (model, z_base) pair from a batch for disentanglement analysis.

        Args:
            model (Any): The generative model.
            batch (Any): Input batch for analysis.
            loss (float): Loss value (unused).
            epoch (int): Current epoch (unused).
        """
        z_base = model.get_disentanglement_latent(batch)
        self.update(model, z_base)

    def _reshape_img(
        self, img: np.ndarray[Any, Any], expected_shape: Any
    ) -> np.ndarray[Any, Any]:
        """
        Reshape an image for visualization (delegates to utils).

        Args:
            img (np.ndarray[Any, Any]): The image to reshape.
            expected_shape (Any): The expected shape for the image.
        Returns:
            np.ndarray[Any, Any]: Reshaped image as np.ndarray.
        """
        return reshape_img(img, expected_shape)

    def save(self, path: str) -> None:
        """
        Visualize the effect of varying each latent dimension and save the result to disk.

        Args:
            path (str): Output file path for the visualization.
        Raises:
            AttributeError: If the model does not have a hidden_dim attribute.
        """
        if not self.pairs:
            print(
                f"[DisentanglementAnalysis] Warning: No pairs to visualize, nothing saved to {path}"
            )
            return
        model, z_base = self.pairs[0]
        latent_dim = z_base.shape[0]
        device = next(model.parameters()).device
        hidden_dim = getattr(model, "hidden_dim", None)
        if hidden_dim is None and hasattr(model, "backbone"):
            hidden_dim = getattr(model.backbone, "hidden_dim", None)
        if hidden_dim is None:
            raise AttributeError(
                "Neither model nor model.backbone has attribute 'hidden_dim'"
            )
        expected_shape = infer_expected_shape(model, hidden_dim, device)
        imgs: List[List[np.ndarray[Any, Any]]] = []
        for d in range(latent_dim):
            row = traverse_latent_dim(
                z_base, d, self.dim_range, self.steps, model, device, expected_shape
            )
            imgs.append(row)
        imgs_np = np.array(imgs)
        plot_disentanglement_grid(
            imgs_np, latent_dim, self.steps, self.title, path, expected_shape
        )
