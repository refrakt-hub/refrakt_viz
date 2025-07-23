from refrakt_viz.generative.base import GenerativeVisualizationComponent
from refrakt_viz.registry import register_viz
import numpy as np
import matplotlib.pyplot as plt
import os
import torch

@register_viz("disentanglement_analysis")
class DisentanglementAnalysis(GenerativeVisualizationComponent):
    registry_name = "disentanglement_analysis"
    def __init__(self, dim_range=3.0, steps=7, title="Disentanglement Analysis"):
        self.dim_range = dim_range
        self.steps = steps
        self.title = title
        self.pairs = []  # list of (model, z_base)

    def update(self, model, z_base):
        self.pairs.append((model, np.array(z_base)))

    def update_from_batch(self, model, batch, loss, epoch):
        # Assume batch provides z_base for disentanglement
        z_base = model.get_disentanglement_latent(batch)
        self.update(model, z_base)

    def _reshape_img(self, img, expected_shape):
        # img: numpy array, expected_shape: tuple (C, H, W) or (H, W)
        if img.ndim == 1 and expected_shape is not None:
            img = img.reshape(expected_shape)
        elif img.ndim == 2 and expected_shape and img.shape != expected_shape:
            img = img.reshape(expected_shape)
        # If (C, H, W), convert to (H, W, C) for RGB plotting
        if img.ndim == 3 and img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))
        return img

    def save(self, path: str):
        import numpy as np
        import torch
        if not self.pairs:
            print(f"[DisentanglementAnalysis] Warning: No pairs to visualize, nothing saved to {path}")
            return
        # Only save the first pair for now (one image per epoch)
        model, z_base = self.pairs[0]
        latent_dim = z_base.shape[0]
        device = next(model.parameters()).device
        # Robustly get hidden_dim from model or backbone
        hidden_dim = getattr(model, "hidden_dim", None)
        if hidden_dim is None and hasattr(model, "backbone"):
            hidden_dim = getattr(model.backbone, "hidden_dim", None)
        if hidden_dim is None:
            raise AttributeError("Neither model nor model.backbone has attribute 'hidden_dim'")
        # Infer expected shape from a sample decode
        with torch.no_grad():
            sample = model.decode(torch.zeros(1, hidden_dim, device=device))
        if isinstance(sample, torch.Tensor):
            sample = sample.detach().cpu().numpy()
        if sample.ndim == 4:  # (B, C, H, W)
            expected_shape = sample.shape[1:]
        elif sample.ndim == 3:
            # (B, H, W) or (C, H, W)
            if sample.shape[0] in [1, 3]:
                expected_shape = sample.shape[1:]
            else:
                expected_shape = sample.shape
        elif sample.ndim == 2:
            expected_shape = sample.shape
        else:
            expected_shape = None
        imgs = []
        for d in range(latent_dim):
            row = []
            for v in np.linspace(-self.dim_range, self.dim_range, self.steps):
                z = z_base.copy()
                z_tensor = torch.as_tensor(z[None], dtype=torch.float32, device=device)
                if hasattr(model, 'decode'):
                    img = model.decode(z_tensor)
                elif hasattr(model, 'generate'):
                    img = model.generate(z_tensor)
                else:
                    raise AttributeError("Model must have either a 'decode' or 'generate' method for disentanglement analysis.")
                if isinstance(img, torch.Tensor):
                    img = img.detach().cpu().numpy()
                img = self._reshape_img(img, expected_shape)
                # Debug: print min/max/mean for each image
                print(f"[DisentanglementAnalysis] dim {d} value {v:.2f} img stats: min={img.min():.4f}, max={img.max():.4f}, mean={img.mean():.4f}")
                # Clip to [0, 1] for visualization
                img = np.clip(img, 0, 1)
                row.append(img.squeeze())
            imgs.append(row)
        imgs = np.array(imgs)
        import matplotlib.pyplot as plt
        plt.figure(figsize=(self.steps * 2, latent_dim * 2))
        for i in range(latent_dim):
            for j in range(self.steps):
                img = imgs[i, j]
                img = self._reshape_img(img, expected_shape)
                # Use gray colormap for single-channel, otherwise default
                cmap = "gray" if (img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1)) else None
                plt.subplot(latent_dim, self.steps, i * self.steps + j + 1)
                plt.imshow(img, cmap=cmap)
                plt.axis("off")
                if j == 0:
                    plt.ylabel(f"dim {i}")
        plt.suptitle(self.title)
        plt.tight_layout()
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        plt.close()
        print(f"[DisentanglementAnalysis] Saved disentanglement analysis to {path}") 