"""
Latent space projection visualization for generative models.

This module provides a visualization component for projecting and visualizing latent spaces
of generative models using dimensionality reduction techniques such as PCA and t-SNE.

Typical usage:
    from refrakt_viz.generative import LatentSpaceProjection
    viz = LatentSpaceProjection(method="pca", title="Latent Space")
    viz.update(latents, labels)
    viz.save("latent_space.png")

Classes:
    - LatentSpaceProjection: Visualize projections of latent spaces (PCA, t-SNE).
"""

from __future__ import annotations

import os
from typing import Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from refrakt_viz.base import GenerativeVisualizationComponent
from refrakt_viz.registry import register_viz


@register_viz("latent_space_projection")
class LatentSpaceProjection(GenerativeVisualizationComponent):
    """
    Visualization component for projecting and visualizing latent spaces of generative models.

    This class accumulates batches of latent vectors and labels, and provides a method to project
    the latent space to 2D using PCA or t-SNE and save the result to disk.

    Attributes:
        method (str): Dimensionality reduction method ("pca" or "tsne").
        title (str): Title for the visualization.
        latents (List[np.ndarray[Any, Any]]): List of latent vector batches.
        labels (List[np.ndarray[Any, Any]]): List of label batches.
    """

    registry_name: str = "latent_space_projection"

    def __init__(
        self, method: str = "pca", title: str = "Latent Space Projection"
    ) -> None:
        """
        Initialize the LatentSpaceProjection visualization.

        Args:
            method (str): Dimensionality reduction method ("pca" or "tsne").
            title (str): Title for the visualization.
        """
        self.method: str = method
        self.title: str = title
        self.latents: List[np.ndarray[Any, Any]] = []
        self.labels: List[np.ndarray[Any, Any]] = []

    def update(self, latents: Any, labels: Optional[Any] = None) -> None:
        """
        Add a batch of latent vectors and labels to the visualization.

        Args:
            latents (Any): Batch of latent vectors.
            labels (Optional[Any]): Batch of labels (optional).
        """
        self.latents.append(np.array(latents))
        if labels is not None:
            self.labels.append(np.array(labels))

    def update_from_batch(
        self, model: Any, batch: Any, loss: float, epoch: int
    ) -> None:
        """
        Add a batch of latent vectors and labels from a model and batch.

        Args:
            model (Any): The generative model.
            batch (Any): Input batch for latent extraction.
            loss (float): Loss value (unused).
            epoch (int): Current epoch (unused).
        """
        latents, labels = model.get_latents_and_labels(batch)
        self.update(latents, labels)

    def save(self, path: str, epoch: Optional[int] = None) -> None:
        """
        Project the latent space to 2D and save the result to disk.

        Args:
            path (str): Output file path for the visualization.
            epoch (Optional[int]): Epoch number for the title (optional).
        Raises:
            ValueError: If the method is not "pca" or "tsne".
        """
        latents = np.concatenate(self.latents, axis=0)
        labels = np.concatenate(self.labels, axis=0) if self.labels else None
        if self.method == "pca":
            projector = PCA(n_components=2)
        elif self.method == "tsne":
            projector = TSNE(n_components=2, random_state=42)
        else:
            raise ValueError("method must be 'pca' or 'tsne'")
        proj = projector.fit_transform(latents)
        plt.figure(figsize=(8, 6))
        if labels is not None:
            scatter = plt.scatter(
                proj[:, 0], proj[:, 1], c=labels, cmap="tab10", alpha=0.7
            )
            plt.legend(*scatter.legend_elements(), title="Labels")
        else:
            plt.scatter(proj[:, 0], proj[:, 1], alpha=0.7)
        title = self.title
        if epoch is not None:
            title = f"{self.title} (Epoch {epoch})"
        plt.title(title)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.tight_layout()
        out_dir = os.path.dirname(path)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "latent_space_projection.png")
        plt.savefig(out_path)
        plt.close()
        print(f"[LatentSpaceProjection] Saved latent space projection to {out_path}")
