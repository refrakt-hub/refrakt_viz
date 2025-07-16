from typing import List
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from refrakt_viz.registry import register_viz
from refrakt_viz.supervised.base import VisualizationComponent

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
        # Add counts in each cell
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=12)
        plt.tight_layout()
        plt.savefig(path)
        print(f"[ConfusionMatrixPlot] Saved confusion matrix plot: {path}")
        plt.close()

    def save_with_name(self, model_name: str = "model", mode: str = "test") -> None:
        import os
        out_dir = f"visualizations/{model_name}"
        os.makedirs(out_dir, exist_ok=True)
        self.save(f"{out_dir}/confusion_matrix.png")

    def show(self, model_name: str = "model", mode: str = "test") -> None:
        self.save_with_name(model_name, mode)
        img_path = f"visualizations/{model_name}/confusion_matrix.png"
        import os
        if os.path.exists(img_path):
            img = plt.imread(img_path)
            plt.imshow(img)
            plt.axis('off')
            plt.show() 