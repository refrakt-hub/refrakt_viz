from refrakt_viz.generative.base import GenerativeVisualizationComponent
from refrakt_viz.registry import register_viz
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os

@register_viz("latent_space_projection")
class LatentSpaceProjection(GenerativeVisualizationComponent):
    registry_name = "latent_space_projection"
    def __init__(self, method="pca", title="Latent Space Projection"):
        self.method = method
        self.title = title
        self.latents = []
        self.labels = []

    def update(self, latents, labels=None):
        self.latents.append(np.array(latents))
        if labels is not None:
            self.labels.append(np.array(labels))

    def update_from_batch(self, model, batch, loss, epoch):
        # Assume model returns latents and labels for the batch
        latents, labels = model.get_latents_and_labels(batch)
        self.update(latents, labels)

    def save(self, path: str, epoch: int = None):
        latents = np.concatenate(self.latents, axis=0)
        labels = np.concatenate(self.labels, axis=0) if self.labels else None
        if self.method == "pca":
            projector = PCA(n_components=2)
        elif self.method == "tsne":
            projector = TSNE(n_components=2, random_state=42)
        else:
            raise ValueError("method must be 'pca' or 'tsne'")
        proj = projector.fit_transform(latents)
        plt.figure(figsize=(8, 6))
        if labels is not None:
            scatter = plt.scatter(proj[:, 0], proj[:, 1], c=labels, cmap="tab10", alpha=0.7)
            plt.legend(*scatter.legend_elements(), title="Labels")
        else:
            plt.scatter(proj[:, 0], proj[:, 1], alpha=0.7)
        title = self.title
        if epoch is not None:
            title = f"{self.title} (Epoch {epoch})"
        plt.title(title)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.tight_layout()
        out_dir = os.path.dirname(path)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "latent_space_projection.png")
        plt.savefig(out_path)
        plt.close()
        print(f"[LatentSpaceProjection] Saved latent space projection to {out_path}") 