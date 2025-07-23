from refrakt_viz.generative.base import GenerativeVisualizationComponent
from refrakt_viz.registry import register_viz
import numpy as np
import matplotlib.pyplot as plt
import os
import random

@register_viz("sample_generation")
class SampleGeneration(GenerativeVisualizationComponent):
    registry_name = "sample_generation"
    def __init__(self, nrow=8, title="Sample Generation"):
        self.nrow = nrow
        self.title = title
        self.samples = []

    def update(self, samples):
        self.samples.append(np.array(samples))

    def update_from_batch(self, model, batch, loss, epoch):
        # Assume model can generate samples for the batch
        samples = model.generate_samples(batch)
        self.update(samples)

    def _reshape_image(self, img):
        # Robustly reshape flat images to 28x28 or infer square, else fallback
        if img is None:
            print(f"[SampleGeneration] Warning: Received None image, using zeros fallback.")
            return np.zeros((28, 28), dtype=np.float32)
        img = np.array(img)
        if img.ndim == 1:
            side = int(np.sqrt(img.size))
            if side * side == img.size:
                return img.reshape(side, side)
            else:
                print(f"[SampleGeneration] Warning: Cannot reshape flat image of size {img.size}, using zeros fallback.")
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
            print(f"[SampleGeneration] Warning: Unexpected image shape {getattr(img, 'shape', None)}, using zeros fallback.")
            return np.zeros((28, 28), dtype=np.float32)

    def save(self, path: str):
        if not self.samples:
            print("[SampleGeneration] No samples to save.")
            return
        # Pick a random batch
        batch_idx = random.randint(0, len(self.samples) - 1)
        samples = self.samples[batch_idx]
        N = len(samples)
        nrow = self.nrow
        ncol = int(np.ceil(N / nrow))
        plt.figure(figsize=(nrow * 2, ncol * 2))
        for i in range(N):
            plt.subplot(ncol, nrow, i + 1)
            img = self._reshape_image(samples[i])
            try:
                img = np.array(img, dtype=np.float32)
            except Exception:
                img = np.zeros((28, 28), dtype=np.float32)
            plt.imshow(img, cmap="gray" if img.ndim == 2 or (img.ndim == 3 and img.shape[-1] == 1) else None)
            plt.axis("off")
        plt.suptitle(f"{self.title} (batch {batch_idx})")
        plt.tight_layout()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        plt.close()
        print(f"[SampleGeneration] Saved sample grid to {path} (batch {batch_idx})")
        # Clear for next epoch
        self.samples.clear()

@register_viz("latent_interpolation")
class LatentInterpolation(GenerativeVisualizationComponent):
    def __init__(self, steps=8, title="Latent Interpolation"):
        self.steps = steps
        self.title = title
        self.interpolations = []  # list of (model, z_start, z_end)

    def update(self, model, z_start, z_end):
        self.interpolations.append((model, np.array(z_start), np.array(z_end)))

    def update_from_batch(self, model, batch, loss, epoch):
        # Assume batch provides z_start and z_end for interpolation
        z_start, z_end = model.get_interpolation_latents(batch)
        self.update(model, z_start, z_end)

    def save(self, path: str):
        for idx, (model, z_start, z_end) in enumerate(self.interpolations):
            zs = np.linspace(0, 1, self.steps)[:, None] * z_end + (1 - np.linspace(0, 1, self.steps)[:, None]) * z_start
            imgs = np.array([model.decode(z[None]) if hasattr(model, 'decode') else model.generate(z[None]) for z in zs])
            imgs = imgs.squeeze()
            plt.figure(figsize=(self.steps * 2, 2))
            for i in range(self.steps):
                plt.subplot(1, self.steps, i + 1)
                plt.imshow(imgs[i], cmap="gray" if imgs.shape[-1] == 1 or len(imgs.shape) == 3 else None)
                plt.axis("off")
            plt.suptitle(self.title)
            plt.tight_layout()
            out_path = path.replace(".png", f"_{idx}.png")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            plt.savefig(out_path)
            plt.close()
            print(f"[LatentInterpolation] Saved latent interpolation to {out_path}") 