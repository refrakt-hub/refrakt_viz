from typing import Dict, List, Any
import matplotlib.pyplot as plt
from refrakt_viz.registry import register_viz
from refrakt_viz.supervised.base import VisualizationComponent

@register_viz("per_layer_metrics")
class PerLayerMetricsPlot(VisualizationComponent):
    def __init__(self) -> None:
        self.metrics_history: List[Dict[str, Any]] = []
        self.split_history: List[str] = []

    def update(self, layer_metrics: Dict[str, Dict[str, float]], split: str = "train") -> None:
        # layer_metrics: {layer_name: {"activation_mean": float, "grad_mean": float, ...}}
        self.metrics_history.append(layer_metrics)
        self.split_history.append(split)

    def save(self, path: str, mode: str = "train") -> None:
        if not self.metrics_history:
            return
        from collections import defaultdict
        import os
        os.makedirs("visualizations", exist_ok=True)
        split_to_metrics = defaultdict(list)
        for metrics, split in zip(self.metrics_history, self.split_history):
            split_to_metrics[split].append(metrics)
        # Only plot and save for the current mode
        if mode in split_to_metrics:
            metrics_list = split_to_metrics[mode]
            layers = list(metrics_list[0].keys())
            num_layers = len(layers)
            fig, axes = plt.subplots(num_layers, 1, figsize=(8, 2 * num_layers))
            if num_layers == 1:
                axes = [axes]
            for i, layer in enumerate(layers):
                act_means = [h[layer]["activation_mean"] for h in metrics_list]
                grad_means = [h[layer]["grad_mean"] for h in metrics_list]
                axes[i].plot(act_means, label="Activation Mean")
                axes[i].plot(grad_means, label="Grad Mean")
                axes[i].set_title(f"{layer} ({mode})")
                axes[i].legend()
            plt.tight_layout()
            out_path = path
            plt.savefig(out_path)
            print(f"[PerLayerMetricsPlot] Saved per-layer metrics plot: {out_path}")
            plt.close()

    def save_with_name(self, model_name: str = "model", mode: str = "train") -> None:
        import os
        out_dir = f"visualizations/{model_name}"
        os.makedirs(out_dir, exist_ok=True)
        self.save(f"{out_dir}/per_layer_metric_{mode}.png", mode=mode)

    def show(self, model_name: str = "model", mode: str = "train") -> None:
        self.save_with_name(model_name, mode)
        # Optionally, show the last split
        if self.split_history:
            last_split = self.split_history[-1]
            img_path = f"visualizations/{model_name}/per_layer_metric_{mode}.png"
            import os
            if os.path.exists(img_path):
                img = plt.imread(img_path)
                plt.imshow(img)
                plt.axis('off')
                plt.show() 