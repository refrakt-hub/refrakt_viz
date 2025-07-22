from typing import List, Optional
import os
import matplotlib.pyplot as plt
from refrakt_viz.registry import register_viz
from .base import ContrastiveVisualizationComponent
from typing import Any

@register_viz("contrastive_loss_curve")
class ContrastiveLossCurvePlot(ContrastiveVisualizationComponent):
    def __init__(self) -> None:
        self.loss_history: List[float] = []
        self.title: str = "Contrastive Loss Curve"

    def update(self, loss: float, title: Optional[str] = None) -> None:
        self.loss_history.append(loss)
        if title is not None:
            self.title = title

    def update_from_batch(self, model: Any, batch: Any, loss: float, epoch: int) -> None:
        self.update(loss)

    registry_name = "contrastive_loss_curve"
    def save_with_name(self, model_name: str = "model", mode: str = "batch") -> None:
        out_dir = f"visualizations/{model_name}"
        os.makedirs(out_dir, exist_ok=True)
        self.save(f"{out_dir}/{self.registry_name}.png", mode=mode)

    def save(self, path: str, mode: str = "batch") -> None:
        if not self.loss_history:
            raise ValueError("No loss history to visualize. Call update() first.")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.plot(self.loss_history, label="Contrastive Loss", color="blue")
        plt.xlabel("Batch" if mode == "batch" else "Epoch")
        plt.ylabel("Loss")
        plt.title(f"Contrastive Loss Curve ({'per batch' if mode == 'batch' else 'per epoch'})")
        plt.legend()
        plt.tight_layout()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        plt.close() 