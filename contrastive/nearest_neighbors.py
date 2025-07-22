from typing import Any, List, Optional, Sequence
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances
from refrakt_viz.registry import register_viz
from .base import ContrastiveVisualizationComponent

@register_viz("nearest_neighbors")
class NearestNeighborsPlot(ContrastiveVisualizationComponent):
    def __init__(self) -> None:
        self.anchors: Optional[np.ndarray] = None
        self.anchor_imgs: Optional[List[np.ndarray]] = None
        self.candidates: Optional[np.ndarray] = None
        self.candidate_imgs: Optional[List[np.ndarray]] = None
        self.k: int = 5
        self.title: str = "Nearest Neighbor Retrieval"

    def update(
        self,
        anchors: np.ndarray,
        anchor_imgs: List[np.ndarray],
        candidates: np.ndarray,
        candidate_imgs: List[np.ndarray],
        k: int = 5,
        title: Optional[str] = None,
    ) -> None:
        self.anchors = anchors
        self.anchor_imgs = anchor_imgs
        self.candidates = candidates
        self.candidate_imgs = candidate_imgs
        self.k = k
        if title is not None:
            self.title = title

    def update_from_batch(self, model: Any, batch: Any, loss: float, epoch: int) -> None:
        import torch
        with torch.no_grad():
            view1 = batch[0]
            embeddings = model.encode(view1.to(next(model.parameters()).device)).cpu().numpy()
            anchor_imgs = view1[:8].cpu().numpy()
            anchors = embeddings[:8]
            candidate_imgs = view1[8:24].cpu().numpy()
            candidates = embeddings[8:24]
        self.update(anchors, list(anchor_imgs), candidates, list(candidate_imgs))

    def save(self, path: str) -> None:
        if (
            self.anchors is None or self.anchor_imgs is None or
            self.candidates is None or self.candidate_imgs is None
        ):
            raise ValueError("Missing data for nearest neighbor visualization. Call update() first.")
        n_anchors = len(self.anchor_imgs)
        dists = cosine_distances(self.anchors, self.candidates)
        nn_indices = np.argsort(dists, axis=1)[:, :self.k]
        fig, axes = plt.subplots(n_anchors, self.k + 1, figsize=(2 * (self.k + 1), 2 * n_anchors))
        if n_anchors == 1:
            axes = np.expand_dims(axes, 0)
        for i in range(n_anchors):
            img = self.anchor_imgs[i]
            if img.ndim == 3 and img.shape[0] in (1, 3):
                img = np.transpose(img, (1, 2, 0))
            axes[i, 0].imshow(img)
            axes[i, 0].set_title("Anchor")
            axes[i, 0].axis('off')
            for j in range(self.k):
                idx = nn_indices[i, j]
                nn_img = self.candidate_imgs[idx]
                if nn_img.ndim == 3 and nn_img.shape[0] in (1, 3):
                    nn_img = np.transpose(nn_img, (1, 2, 0))
                axes[i, j + 1].imshow(nn_img)
                axes[i, j + 1].set_title(f"NN {j+1}")
                axes[i, j + 1].axis('off')
        plt.suptitle(self.title)
        plt.tight_layout()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        plt.close()

    registry_name = "nearest_neighbors"
    def save_with_name(self, model_name: str = "model") -> None:
        out_dir = f"visualizations/{model_name}"
        os.makedirs(out_dir, exist_ok=True)
        self.save(f"{out_dir}/{self.registry_name}.png") 