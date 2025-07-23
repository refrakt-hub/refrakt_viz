from refrakt_viz.generative.base import GenerativeVisualizationComponent
from refrakt_viz.registry import register_viz
import numpy as np
import matplotlib.pyplot as plt
import os
import random

@register_viz("feature_attribution")
class FeatureAttribution(GenerativeVisualizationComponent):
    registry_name = "feature_attribution"
    def __init__(self, title="Feature Attribution"):
        self.title = title
        self.inputs = []
        self.models = []

    def update(self, input_img, model):
        self.inputs.append(np.array(input_img))
        self.models.append(model)

    def update_from_batch(self, model, batch, loss, epoch):
        # Assume batch provides input_img for saliency
        input_img = model.get_saliency_input(batch)
        self.update(input_img, model)

    def save(self, path: str):
        import torch
        if not self.inputs:
            print("[FeatureAttribution] No inputs to save.")
            return
        # Pick a random batch
        idx = random.randint(0, len(self.inputs) - 1)
        input_img = self.inputs[idx]
        model = self.models[idx]
        input_tensor = torch.tensor(input_img, dtype=torch.float32, requires_grad=True)
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor.unsqueeze(0)
        # Move model and input_tensor to CPU
        model_cpu = model.cpu() if hasattr(model, 'cpu') else model
        input_tensor = input_tensor.cpu()
        output = model_cpu(input_tensor)
        # Handle ModelOutput or tensor
        if hasattr(output, 'reconstruction') and output.reconstruction is not None:
            out_tensor = output.reconstruction
        elif hasattr(output, 'image') and output.image is not None:
            out_tensor = output.image
        elif hasattr(output, 'logits') and output.logits is not None:
            out_tensor = output.logits
        elif isinstance(output, torch.Tensor):
            out_tensor = output
        else:
            raise ValueError("Model output does not contain a tensor for saliency computation.")
        loss = out_tensor.norm()
        loss.backward()
        if input_tensor.grad is None:
            print(f"[FeatureAttribution] Warning: input_tensor.grad is None for sample {idx}, skipping.")
            return
        saliency = input_tensor.grad.abs().detach().cpu().numpy().squeeze()
        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(input_img.squeeze(), cmap="gray" if input_img.shape[-1] == 1 or len(input_img.shape) == 2 else None)
        plt.title(f"Input (batch {idx})")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(saliency, cmap="hot")
        plt.title("Saliency")
        plt.axis("off")
        plt.suptitle(self.title)
        plt.tight_layout()
        # Save as visualizations/autoencoder/feature_attribution.png
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        plt.close()
        print(f"[FeatureAttribution] Saved saliency map to {path} (batch {idx})")
        # Clear for next epoch
        self.inputs.clear()
        self.models.clear() 