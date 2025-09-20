"""
Pair similarity visualization for contrastive learning models.

This module provides a visualization component for displaying similarity distributions
for positive and negative pairs in contrastive learning, allowing users to assess separation.

Typical usage:
    from refrakt_viz.contrastive import PairSimilarityPlot
    viz = PairSimilarityPlot()
    viz.update(pos_sims, neg_sims)
    viz.save("pair_similarity.png")

Classes:
    - PairSimilarityPlot: Visualize similarity distributions for positive and negative pairs.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from refrakt_viz.base import ContrastiveVisualizationComponent
from refrakt_viz.registry import register_viz


@register_viz("pair_similarity")
class PairSimilarityPlot(ContrastiveVisualizationComponent):
    """
    Visualization component for displaying similarity distributions for positive and negative pairs.

    This class accumulates similarity scores for positive and negative pairs, and provides a method
    to visualize their distributions. Useful for assessing separation in contrastive learning.

    Attributes:
        pos_sims (Optional[np.ndarray[Any, Any]]): Similarity scores for positive pairs.
        neg_sims (Optional[np.ndarray[Any, Any]]): Similarity scores for negative pairs.
        title (str): Title for the visualization.
    """

    def __init__(self) -> None:
        """
        Initialize the PairSimilarityPlot visualization.
        """
        self.pos_sims: Optional[np.ndarray[Any, Any]] = None
        self.neg_sims: Optional[np.ndarray[Any, Any]] = None
        self.title: str = "Pair Similarity Distribution"

    def update(self, *args, **kwargs) -> None:
        """
        Update the visualization with new data. Expects pos_sims, neg_sims, title as arguments.
        """
        pos_sims = kwargs.get("pos_sims", args[0] if len(args) > 0 else None)
        neg_sims = kwargs.get("neg_sims", args[1] if len(args) > 1 else None)
        title = kwargs.get("title", args[2] if len(args) > 2 else None)
        self.pos_sims = pos_sims
        self.neg_sims = neg_sims
        if title is not None:
            self.title = title

    def update_from_batch(
        self, model: Any, batch: Any, loss: float, epoch: int
    ) -> None:
        """
        Add similarity scores from a model and batch.

        Args:
            model (Any): The contrastive model.
            batch (Any): Input batch for similarity computation.
            loss (float): Loss value (unused).
            epoch (int): Current epoch (unused).
        """
        with torch.no_grad():
            view1, view2 = batch[0], batch[1]
            z1 = model.encode(view1.to(next(model.parameters()).device))
            z2 = model.encode(view2.to(next(model.parameters()).device))
            z1 = torch.nn.functional.normalize(z1, dim=1)
            z2 = torch.nn.functional.normalize(z2, dim=1)
            pos_sims = (z1 * z2).sum(dim=1).cpu().numpy()
            z2_neg = z2[torch.randperm(z2.size(0))]
            neg_sims = (z1 * z2_neg).sum(dim=1).cpu().numpy()
        self.update(pos_sims.tolist(), neg_sims.tolist())

    def save(self, path: str) -> None:
        """
        Save a plot of similarity distributions to disk.

        Args:
            path (str): Output file path for the visualization.
        Raises:
            ValueError: If similarity data is missing.
        """
        if self.pos_sims is None or self.neg_sims is None:
            raise ValueError("No similarity data to visualize. Call update() first.")
        plt.figure(figsize=(8, 6))
        sns.histplot(
            self.pos_sims,
            color="green",
            label="Positive Pairs",
            kde=True,
            stat="density",
            bins=30,
            alpha=0.6,
        )
        sns.histplot(
            self.neg_sims,
            color="red",
            label="Negative Pairs",
            kde=True,
            stat="density",
            bins=30,
            alpha=0.6,
        )
        plt.legend()
        plt.title(self.title)
        plt.xlabel("Cosine Similarity")
        plt.ylabel("Density")
        plt.tight_layout()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        plt.close()

    registry_name: str = "pair_similarity"

    def save_with_name(self, model_name: str = "model") -> None:
        """
        Save the similarity distribution plot to a model-specific directory.

        Args:
            model_name (str): Name of the model for directory naming.
        """
        out_dir = f"visualizations/{model_name}"
        os.makedirs(out_dir, exist_ok=True)
        self.save(f"{out_dir}/{self.registry_name}.png")
