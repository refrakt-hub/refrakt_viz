from typing import List, Dict
import matplotlib.pyplot as plt
from refrakt_viz.registry import register_viz
from .base import VisualizationComponent

@register_viz("loss_accuracy")
class LossAccuracyPlot(VisualizationComponent):
    def __init__(self) -> None:
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }

    def update(self, train_loss: float, val_loss: float, train_acc: float, val_acc: float) -> None:
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["train_acc"].append(train_acc)
        self.history["val_acc"].append(val_acc)

    def save(self, path: str) -> None:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.history["train_loss"], label="Train Loss")
        plt.plot(self.history["val_loss"], label="Val Loss")
        plt.legend()
        plt.title("Loss")
        plt.subplot(1, 2, 2)
        plt.plot(self.history["train_acc"], label="Train Acc")
        plt.plot(self.history["val_acc"], label="Val Acc")
        plt.legend()
        plt.title("Accuracy")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def show(self) -> None:
        self.save("/tmp/loss_acc.png")
        img = plt.imread("/tmp/loss_acc.png")
        plt.imshow(img)
        plt.axis('off')
        plt.show() 