"""
Nearest neighbor visualization for contrastive learning models.

This module provides a visualization component for displaying nearest neighbor retrievals
in embedding space, allowing users to assess the quality of learned representations.

Typical usage:
    from refrakt_viz.contrastive import NearestNeighborsPlot
    viz = NearestNeighborsPlot()
    viz.update(anchors, anchor_imgs, candidates, candidate_imgs, k=5)
    viz.save("nearest_neighbors.png")

Classes:
    - NearestNeighborsPlot: Visualize nearest neighbor retrievals.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional

import numpy as np
import torch

from refrakt_viz.base import ContrastiveVisualizationComponent
from refrakt_viz.registry import register_viz
from refrakt_viz.utils.nearest_neighbors_utils import (
    plot_nearest_neighbors_grid,
)


@register_viz("nearest_neighbors")
class NearestNeighborsPlot(ContrastiveVisualizationComponent):
    """
    Visualization component for displaying nearest neighbor retrievals in embedding space.

    This class accumulates anchor and candidate embeddings and images, and provides a method to
    visualize the nearest neighbors for each anchor. Useful for evaluating representation learning
    in contrastive models.

    Attributes:
        anchors (Optional[np.ndarray[Any, Any]]): Anchor embeddings.
        anchor_imgs (Optional[List[np.ndarray[Any, Any]]]): Anchor images.
        candidates (Optional[np.ndarray[Any, Any]]): Candidate embeddings.
        candidate_imgs (Optional[List[np.ndarray[Any, Any]]]): Candidate images.
        k (int): Number of nearest neighbors to display.
        title (str): Title for the visualization.
    """

    def __init__(self) -> None:
        """
        Initialize the NearestNeighborsPlot visualization.
        """
        self.anchors: Optional[np.ndarray[Any, Any]] = None
        self.anchor_imgs: Optional[List[np.ndarray[Any, Any]]] = None
        self.candidates: Optional[np.ndarray[Any, Any]] = None
        self.candidate_imgs: Optional[List[np.ndarray[Any, Any]]] = None
        self.k: int = 5
        self.title: str = "Nearest Neighbor Retrieval"

    def update(
        self,
        anchors: np.ndarray[Any, Any],
        anchor_imgs: List[np.ndarray[Any, Any]],
        candidates: np.ndarray[Any, Any],
        candidate_imgs: List[np.ndarray[Any, Any]],
        k: int = 5,
        title: Optional[str] = None,
    ) -> (
        None
    ):  # pylint: disable=arguments-differ,too-many-arguments,too-many-positional-arguments
        """
        Add anchor and candidate embeddings/images for nearest neighbor visualization.

        Args:
            anchors (np.ndarray[Any, Any]): Anchor embeddings.
            anchor_imgs (List[np.ndarray[Any, Any]]): Anchor images.
            candidates (np.ndarray[Any, Any]): Candidate embeddings.
            candidate_imgs (List[np.ndarray[Any, Any]]): Candidate images.
            k (int): Number of nearest neighbors to display.
            title (Optional[str]): Title for the visualization (optional).
        """
        self.anchors = anchors
        self.anchor_imgs = anchor_imgs
        self.candidates = candidates
        self.candidate_imgs = candidate_imgs
        self.k = k
        if title is not None:
            self.title = title

    def update_from_batch(
        self, model: Any, batch: Any, loss: float, epoch: int
    ) -> None:
        """
        Add anchor and candidate embeddings/images from a model and batch.

        Args:
            model (Any): The contrastive model.
            batch (Any): Input batch for embedding extraction.
            loss (float): Loss value (unused).
            epoch (int): Current epoch (unused).
        """
        with torch.no_grad():
            view1 = batch[0]
            embeddings = (
                model.encode(view1.to(next(model.parameters()).device)).cpu().numpy()
            )
            anchor_imgs = view1[:8].cpu().numpy()
            anchors = embeddings[:8]
            candidate_imgs = view1[8:24].cpu().numpy()
            candidates = embeddings[8:24]
        self.update(anchors, list(anchor_imgs), candidates, list(candidate_imgs))

    def save(self, path: str) -> None:
        """
        Save a grid of nearest neighbor retrievals to disk.

        Args:
            path (str): Output file path for the visualization.
        Raises:
            ValueError: If required data is missing.
        """
        if (
            self.anchors is None
            or self.anchor_imgs is None
            or self.candidates is None
            or self.candidate_imgs is None
        ):
            raise ValueError(
                "Missing data for nearest neighbor visualization. Call update() first."
            )
        plot_nearest_neighbors_grid(
            self.anchors,
            self.anchor_imgs,
            self.candidates,
            self.candidate_imgs,
            self.k,
            self.title,
            path,
        )

    registry_name: str = "nearest_neighbors"

    def save_with_name(self, model_name: str = "model") -> None:
        """
        Save a grid of nearest neighbor retrievals to a model-specific directory.

        Args:
            model_name (str): Name of the model for directory naming.
        """
        out_dir = f"visualizations/{model_name}"
        os.makedirs(out_dir, exist_ok=True)
        self.save(f"{out_dir}/{self.registry_name}.png")
