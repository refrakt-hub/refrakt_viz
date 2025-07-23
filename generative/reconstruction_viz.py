from refrakt_viz.generative.base import GenerativeVisualizationComponent
from refrakt_viz.registry import register_viz
import numpy as np
import matplotlib.pyplot as plt
import os

@register_viz("reconstruction_viz")
class ReconstructionViz(GenerativeVisualizationComponent):
    registry_name = "reconstruction_viz"
    def __init__(self, n_samples=8, title="Reconstruction Visualization"):
        self.n_samples = n_samples
        self.title = title
        self.inputs = []
        self.recons = []

    def update(self, inputs, recons):
        self.inputs.append(np.array(inputs))
        self.recons.append(np.array(recons))

    def update_from_batch(self, model, batch, loss, epoch):
        # Assume model returns input and reconstruction for the batch
        inputs, recons = model.get_inputs_and_recons(batch)
        self.update(inputs, recons)

    def _reshape_image(self, img):
        # Robustly reshape flat images to 28x28 or infer square, else fallback
        if img is None:
            print(f"[ReconstructionViz] Warning: Received None image, using zeros fallback.")
            return np.zeros((28, 28), dtype=np.float32)
        img = np.array(img)
        if img.ndim == 1:
            side = int(np.sqrt(img.size))
            if side * side == img.size:
                return img.reshape(side, side)
            else:
                print(f"[ReconstructionViz] Warning: Cannot reshape flat image of size {img.size}, using zeros fallback.")
                return np.zeros((28, 28), dtype=np.float32)
        elif img.ndim == 2:
            return img
        elif img.ndim == 3 and img.shape[0] in [1, 3]:
            # (C, H, W) -> (H, W) or (H, W, C)
            if img.shape[0] == 1:
                return img[0]
            elif img.shape[0] == 3:
                return np.transpose(img, (1, 2, 0))
        elif img.ndim == 3 and img.shape[-1] in [1, 3]:
            return img
        else:
            print(f"[ReconstructionViz] Warning: Unexpected image shape {getattr(img, 'shape', None)}, using zeros fallback.")
            return np.zeros((28, 28), dtype=np.float32)

    def save(self, path: str):
        inputs = np.concatenate(self.inputs, axis=0)
        recons = np.concatenate(self.recons, axis=0)
        n_samples = min(self.n_samples, len(inputs))
        plt.figure(figsize=(n_samples * 2, 4))
        for i in range(n_samples):
            plt.subplot(2, n_samples, i + 1)
            img = self._reshape_image(inputs[i])
            try:
                img = np.array(img, dtype=np.float32)
            except Exception:
                img = np.zeros((28, 28), dtype=np.float32)
            plt.imshow(img, cmap="gray" if img.ndim == 2 or (img.ndim == 3 and img.shape[-1] == 1) else None)
            plt.axis("off")
            if i == 0:
                plt.ylabel("Input")
            plt.subplot(2, n_samples, n_samples + i + 1)
            recon_img = self._reshape_image(recons[i])
            try:
                recon_img = np.array(recon_img, dtype=np.float32)
            except Exception:
                recon_img = np.zeros((28, 28), dtype=np.float32)
            plt.imshow(recon_img, cmap="gray" if recon_img.ndim == 2 or (recon_img.ndim == 3 and recon_img.shape[-1] == 1) else None)
            plt.axis("off")
            if i == 0:
                plt.ylabel("Recon")
        plt.suptitle(self.title)
        plt.tight_layout()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        plt.close()
        print(f"[ReconstructionViz] Saved reconstruction visualization to {path}") 