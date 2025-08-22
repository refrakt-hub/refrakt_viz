"""
Cluster assignment visualization for contrastive learning models.

This module provides a visualization component for displaying cluster assignments and confusion matrices
in embedding space, allowing users to assess clustering quality and label correspondence.

Typical usage:
    from refrakt_viz.contrastive import ClusterAssignmentPlot
    viz = ClusterAssignmentPlot()
    viz.update(assignments, labels)
    viz.save("cluster_assignments.png")

Classes:
    - ClusterAssignmentPlot: Visualize cluster assignments and confusion matrices.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

from refrakt_viz.base import ContrastiveVisualizationComponent
from refrakt_viz.registry import register_viz


@register_viz("cluster_assignments")
class ClusterAssignmentPlot(ContrastiveVisualizationComponent):
    """
    Visualization component for displaying cluster assignments and confusion matrices in embedding space.

    This class accumulates cluster assignments and labels, and provides a method to visualize the distribution
    of assignments and the correspondence between clusters and true labels.

    Attributes:
        assignments (Optional[np.ndarray[Any, Any]]): Cluster assignments.
        labels (Optional[np.ndarray[Any, Any]]): True labels for each sample.
        title (str): Title for the visualization.
    """

    def __init__(self) -> None:
        """
        Initialize the ClusterAssignmentPlot visualization.
        """
        self.assignments: Optional[np.ndarray[Any, Any]] = None
        self.labels: Optional[np.ndarray[Any, Any]] = None
        self.title: str = "Cluster Assignments"

    def update(self, *args, **kwargs) -> None:
        """
        Update the visualization with new data. Expects assignments, labels, title as arguments.
        """
        assignments = kwargs.get("assignments", args[0] if len(args) > 0 else None)
        labels = kwargs.get("labels", args[1] if len(args) > 1 else None)
        title = kwargs.get("title", args[2] if len(args) > 2 else None)
        self.assignments = np.array(assignments)
        self.labels = np.array(labels) if labels is not None else None
        if title is not None:
            self.title = title

    def update_from_batch(
        self, model: Any, batch: Any, loss: float, epoch: int
    ) -> None:
        """
        Add cluster assignments and labels from a model and batch.

        Args:
            model (Any): The contrastive model.
            batch (Any): Input batch for clustering.
            loss (float): Loss value (unused).
            epoch (int): Current epoch (unused).
        """
        with torch.no_grad():
            view1 = batch[0]
            embeddings = (
                model.encode(view1.to(next(model.parameters()).device)).cpu().numpy()
            )
        n_clusters = 10
        kmeans = KMeans(n_clusters=n_clusters, n_init=1, random_state=0)
        assignments = kmeans.fit_predict(embeddings)
        labels = getattr(batch, "labels", None)
        if labels is None:
            labels = [0] * len(assignments)
        self.update(assignments, labels)

    def save(self, path: str) -> None:
        """
        Save a plot of cluster assignments and (optionally) a confusion matrix to disk.

        Args:
            path (str): Output file path for the visualization.
        Raises:
            ValueError: If assignments are missing.
        """
        if self.assignments is None:
            raise ValueError(
                "No cluster assignments to visualize. Call update() first."
            )
        plt.figure(figsize=(8, 5))
        sns.countplot(x=self.assignments)
        plt.title(self.title)
        plt.xlabel("Cluster ID")
        plt.ylabel("Count")
        plt.tight_layout()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        plt.close()
        if self.labels is not None:
            cm = confusion_matrix(self.labels, self.assignments)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title("Confusion Matrix: True vs Cluster")
            plt.xlabel("Cluster ID")
            plt.ylabel("True Label")
            plt.tight_layout()
            cm_path = os.path.splitext(path)[0] + "_confusion.png"
            plt.savefig(cm_path)
            plt.close()

    def save_with_name(self, model_name: str = "model") -> None:
        """
        Save cluster assignments and confusion matrix to a model-specific directory.

        Args:
            model_name (str): Name of the model for directory naming.
        """
        out_dir = f"visualizations/{model_name}"
        os.makedirs(out_dir, exist_ok=True)
        self.save(f"{out_dir}/cluster_assignment.png")
