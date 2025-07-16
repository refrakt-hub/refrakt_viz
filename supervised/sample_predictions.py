from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from ..registry import register_viz
from .base import VisualizationComponent

@register_viz("sample_predictions")
class SamplePredictionsPlot(VisualizationComponent):
    def __init__(self, class_names: List[str]) -> None:
        self.samples: List[Tuple[np.ndarray, int, int]] = []
        self.class_names = class_names

    def update(self, images: List[np.ndarray], y_true: List[int], y_pred: List[int]) -> None:
        for img, t, p in zip(images, y_true, y_pred):
            self.samples.append((img, t, p))

    def save_with_name(self, model_name: str = "model") -> None:
        import os
        out_dir = f"visualizations/{model_name}"
        os.makedirs(out_dir, exist_ok=True)
        self.save(f"{out_dir}/sample_predictions.png")

    def save(self, path: str, n: int = 8) -> None:
        n = min(n, len(self.samples))
        fig, axes = plt.subplots(1, n, figsize=(n * 2, 2))
        for i, (img, t, p) in enumerate(self.samples[:n]):
            axes[i].imshow(img)
            # Overlay predicted class label on top of the image
            axes[i].text(0.5, -0.15, f"Pred: {self.class_names[p]}", fontsize=10, color="red", ha="center", va="top", transform=axes[i].transAxes)
            axes[i].set_title(f"T:{self.class_names[t]}")
            axes[i].axis('off')
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def show(self, model_name: str = "model") -> None:
        self.save_with_name(model_name)
        img_path = f"visualizations/{model_name}/sample_predictions.png"
        import os
        if os.path.exists(img_path):
            img = plt.imread(img_path)
            plt.imshow(img)
            plt.axis('off')
            plt.show() 