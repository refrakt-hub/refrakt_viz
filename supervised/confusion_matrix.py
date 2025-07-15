from typing import List
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from ..registry import register_viz
from .base import VisualizationComponent

@register_viz("confusion_matrix")
class ConfusionMatrixPlot(VisualizationComponent):
    def __init__(self, class_names: List[str]) -> None:
        self.class_names = class_names
        self.y_true: List[int] = []
        self.y_pred: List[int] = []

    def update(self, y_true: List[int], y_pred: List[int]) -> None:
        self.y_true.extend(y_true)
        self.y_pred.extend(y_pred)

    def save(self, path: str) -> None:
        cm = confusion_matrix(self.y_true, self.y_pred)
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(np.arange(len(self.class_names)))
        ax.set_yticks(np.arange(len(self.class_names)))
        ax.set_xticklabels(self.class_names)
        ax.set_yticklabels(self.class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.colorbar(im)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def show(self) -> None:
        self.save("/tmp/confmat.png")
        img = plt.imread("/tmp/confmat.png")
        plt.imshow(img)
        plt.axis('off')
        plt.show() 