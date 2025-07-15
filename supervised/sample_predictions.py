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

    def save(self, path: str, n: int = 8) -> None:
        n = min(n, len(self.samples))
        fig, axes = plt.subplots(1, n, figsize=(n * 2, 2))
        for i, (img, t, p) in enumerate(self.samples[:n]):
            axes[i].imshow(img)
            axes[i].set_title(f"T:{self.class_names[t]}\nP:{self.class_names[p]}")
            axes[i].axis('off')
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def show(self) -> None:
        self.save("/tmp/samples.png")
        img = plt.imread("/tmp/samples.png")
        plt.imshow(img)
        plt.axis('off')
        plt.show() 