from typing import Optional, Sequence
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from refrakt_viz.registry import register_viz
from .base import ContrastiveVisualizationComponent
from typing import Any

@register_viz("pair_similarity")
class PairSimilarityPlot(ContrastiveVisualizationComponent):
    def __init__(self) -> None:
        self.pos_sims: Optional[np.ndarray] = None
        self.neg_sims: Optional[np.ndarray] = None
        self.title: str = "Pair Similarity Distribution"

    def update(
        self,
        pos_sims: Sequence[float],
        neg_sims: Sequence[float],
        title: Optional[str] = None,
    ) -> None:
        self.pos_sims = np.array(pos_sims)
        self.neg_sims = np.array(neg_sims)
        if title is not None:
            self.title = title

    def update_from_batch(self, model: Any, batch: Any, loss: float, epoch: int) -> None:
        import torch
        with torch.no_grad():
            view1, view2 = batch[0], batch[1]
            z1 = model.encode(view1.to(next(model.parameters()).device))
            z2 = model.encode(view2.to(next(model.parameters()).device))
            z1 = torch.nn.functional.normalize(z1, dim=1)
            z2 = torch.nn.functional.normalize(z2, dim=1)
            pos_sims = (z1 * z2).sum(dim=1).cpu().numpy()
            z2_neg = z2[torch.randperm(z2.size(0))]
            neg_sims = (z1 * z2_neg).sum(dim=1).cpu().numpy()
        self.update(pos_sims, neg_sims)

    def save(self, path: str) -> None:
        if self.pos_sims is None or self.neg_sims is None:
            raise ValueError("No similarity data to visualize. Call update() first.")
        plt.figure(figsize=(8, 6))
        sns.histplot(self.pos_sims, color="green", label="Positive Pairs", kde=True, stat="density", bins=30, alpha=0.6)
        sns.histplot(self.neg_sims, color="red", label="Negative Pairs", kde=True, stat="density", bins=30, alpha=0.6)
        plt.legend()
        plt.title(self.title)
        plt.xlabel("Cosine Similarity")
        plt.ylabel("Density")
        plt.tight_layout()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        plt.close()

    registry_name = "pair_similarity"
    def save_with_name(self, model_name: str = "model") -> None:
        out_dir = f"visualizations/{model_name}"
        os.makedirs(out_dir, exist_ok=True)
        self.save(f"{out_dir}/{self.registry_name}.png") 