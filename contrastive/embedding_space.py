"""
Embedding space visualization for contrastive learning models.

This module provides a visualization component for projecting and visualizing embedding spaces
of contrastive models using dimensionality reduction techniques such as t-SNE and UMAP.

Typical usage:
    from refrakt_viz.contrastive import EmbeddingSpacePlot
    viz = EmbeddingSpacePlot()
    viz.update(embeddings, labels, method="tsne")
    viz.save("embedding_space.png")

Classes:
    - EmbeddingSpacePlot: Visualize embedding spaces using t-SNE or UMAP.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from refrakt_viz.base import ContrastiveVisualizationComponent
from refrakt_viz.registry import register_viz


@register_viz("embedding_space")
class EmbeddingSpacePlot(ContrastiveVisualizationComponent):
    """
    Visualization component for projecting and visualizing embedding spaces of contrastive models.

    This class accumulates batches of embeddings and labels, and provides a method to project
    the embedding space to 2D using t-SNE or UMAP and save the result to disk.

    Attributes:
        embeddings (Optional[np.ndarray[Any, Any]]): Embedding vectors.
        labels (Optional[np.ndarray[Any, Any]]): Labels for each embedding.
        method (str): Dimensionality reduction method ("tsne" or "umap").
        title (str): Title for the visualization.
    """

    def __init__(self) -> None:
        """
        Initialize the EmbeddingSpacePlot visualization.
        """
        self.embeddings: Optional[np.ndarray[Any, Any]] = None
        self.labels: Optional[np.ndarray[Any, Any]] = None
        self.method: str = "tsne"  # or "umap"
        self.title: str = "Embedding Space"

    def update(self, *args, **kwargs) -> None:
        """
        Update the visualization with new data. Expects embeddings, labels, method, title as arguments.
        """
        embeddings = kwargs.get("embeddings", args[0] if len(args) > 0 else None)
        labels = kwargs.get("labels", args[1] if len(args) > 1 else None)
        method = kwargs.get("method", args[2] if len(args) > 2 else "tsne")
        title = kwargs.get("title", args[3] if len(args) > 3 else None)
        self.embeddings = embeddings
        self.labels = labels
        self.method = method
        if title is not None:
            self.title = title

    registry_name: str = "embedding_space"

    def save_with_name(self, model_name: str = "model", mode: str = "test") -> None:
        """
        Save the embedding space visualization to a model-specific directory.

        Args:
            model_name (str): Name of the model for directory naming.
            mode (str): Mode for the plot (default: "test").
        """
        out_dir = f"visualizations/{model_name}"
        os.makedirs(out_dir, exist_ok=True)
        self.save(f"{out_dir}/{self.registry_name}_{mode}.png")

    def update_from_batch(
        self, model: Any, batch: Any, loss: float, epoch: int
    ) -> None:
        """
        Add a batch of embeddings and labels from a model and batch.

        Args:
            model (Any): The contrastive model.
            batch (Any): Input batch for embedding extraction.
            loss (float): Loss value (unused).
            epoch (int): Current epoch (unused).
        """
        with torch.no_grad():
            embeddings = (
                model.encode(batch[0].to(next(model.parameters()).device)).cpu().numpy()
            )
        labels = None
        if isinstance(batch, (tuple, list)) and len(batch) > 1:
            labels = batch[1]
            if hasattr(labels, "cpu"):
                labels = labels.cpu().numpy()
            elif hasattr(labels, "numpy"):
                labels = labels.numpy()
        if labels is None:
            labels = [0] * len(embeddings)
        self.update(embeddings, labels)

    def save(self, path: str) -> None:
        """
        Project the embedding space to 2D and save the result to disk.

        Args:
            path (str): Output file path for the visualization.
        Raises:
            ValueError: If embeddings are missing.
            ImportError: If UMAP is selected but not installed.
        """
        if self.embeddings is None:
            raise ValueError("No embeddings to visualize. Call update() first.")
        if self.method == "umap":
            try:
                import umap
            except ImportError:
                raise ImportError(
                    "UMAP is not installed. Install umap-learn or use method='tsne'."
                )
            reducer = umap.UMAP(n_components=2, random_state=42)
        elif self.method == "tsne":
            reducer = TSNE(n_components=2, random_state=42)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        reduced = reducer.fit_transform(self.embeddings)
        if not isinstance(reduced, np.ndarray):
            reduced = reduced.toarray()  # type: ignore[attr-defined]
        plt.figure(figsize=(8, 8))
        if self.labels is not None:
            scatter = plt.scatter(
                reduced[:, 0], reduced[:, 1], c=self.labels, cmap="tab10", alpha=0.7
            )
            plt.legend(*scatter.legend_elements(), title="Label")
        else:
            plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7)
        plt.title(self.title)
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.tight_layout()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        plt.close()
