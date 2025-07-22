from typing import Any, Optional, Sequence
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
from refrakt_viz.registry import register_viz
from .base import ContrastiveVisualizationComponent

@register_viz("embedding_space")
class EmbeddingSpacePlot(ContrastiveVisualizationComponent):
    def __init__(self) -> None:
        self.embeddings: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.method: str = "tsne"  # or "umap"
        self.title: str = "Embedding Space"

    def update(
        self,
        embeddings: np.ndarray,
        labels: Optional[Sequence[Any]] = None,
        method: str = "tsne",
        title: Optional[str] = None,
    ) -> None:
        self.embeddings = embeddings
        self.labels = np.array(labels) if labels is not None else None
        self.method = method
        if title is not None:
            self.title = title

    registry_name = "embedding_space"
    def save_with_name(self, model_name: str = "model") -> None:
        out_dir = f"visualizations/{model_name}"
        os.makedirs(out_dir, exist_ok=True)
        self.save(f"{out_dir}/{self.registry_name}.png")

    def update_from_batch(self, model: Any, batch: Any, loss: float, epoch: int) -> None:
        import torch
        with torch.no_grad():
            embeddings = model.encode(batch[0].to(next(model.parameters()).device)).cpu().numpy()
        # Try to extract labels from batch if present
        labels = None
        if isinstance(batch, (tuple, list)) and len(batch) > 1:
            labels = batch[1]
            if hasattr(labels, 'cpu'):
                labels = labels.cpu().numpy()
            elif hasattr(labels, 'numpy'):
                labels = labels.numpy()
        if labels is None:
            labels = [0] * len(embeddings)
        self.update(embeddings, labels)

    def save(self, path: str) -> None:
        if self.embeddings is None:
            raise ValueError("No embeddings to visualize. Call update() first.")
        if self.method == "umap" and not HAS_UMAP:
            raise ImportError("UMAP is not installed. Install umap-learn or use method='tsne'.")
        reducer = TSNE(n_components=2, random_state=42) if self.method == "tsne" else umap.UMAP(n_components=2, random_state=42)
        reduced = reducer.fit_transform(self.embeddings)
        plt.figure(figsize=(8, 8))
        if self.labels is not None:
            scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=self.labels, cmap="tab10", alpha=0.7)
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