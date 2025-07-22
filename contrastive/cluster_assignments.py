from typing import Any, List, Optional, Sequence
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from refrakt_viz.registry import register_viz
from .base import ContrastiveVisualizationComponent

@register_viz("cluster_assignments")
class ClusterAssignmentPlot(ContrastiveVisualizationComponent):
    def __init__(self) -> None:
        self.assignments: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.title: str = "Cluster Assignments"

    def update(
        self,
        assignments: Sequence[int],
        labels: Optional[Sequence[Any]] = None,
        title: Optional[str] = None,
    ) -> None:
        self.assignments = np.array(assignments)
        self.labels = np.array(labels) if labels is not None else None
        if title is not None:
            self.title = title

    def update_from_batch(self, model: Any, batch: Any, loss: float, epoch: int) -> None:
        import torch
        from sklearn.cluster import KMeans
        with torch.no_grad():
            view1 = batch[0]
            embeddings = model.encode(view1.to(next(model.parameters()).device)).cpu().numpy()
        n_clusters = 10
        kmeans = KMeans(n_clusters=n_clusters, n_init=1, random_state=0)
        assignments = kmeans.fit_predict(embeddings)
        labels = getattr(batch, 'labels', None)
        if labels is None:
            labels = [0] * len(assignments)
        self.update(assignments, labels)

    def save(self, path: str) -> None:
        if self.assignments is None:
            raise ValueError("No cluster assignments to visualize. Call update() first.")
        plt.figure(figsize=(8, 5))
        sns.countplot(x=self.assignments)
        plt.title(self.title)
        plt.xlabel("Cluster ID")
        plt.ylabel("Count")
        plt.tight_layout()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        plt.close()
        # Optionally, plot confusion matrix if labels are provided
        if self.labels is not None:
            cm = confusion_matrix(self.labels, self.assignments)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title(f"Confusion Matrix: True vs Cluster")
            plt.xlabel("Cluster ID")
            plt.ylabel("True Label")
            plt.tight_layout()
            cm_path = os.path.splitext(path)[0] + "_confusion.png"
            plt.savefig(cm_path)
            plt.close()

    def save_with_name(self, model_name: str = "model") -> None:
        out_dir = f"visualizations/{model_name}"
        os.makedirs(out_dir, exist_ok=True)
        self.save(f"{out_dir}/cluster_assignment.png")
        # Confusion matrix will be saved as cluster_assignment_confusion.png by self.save() 