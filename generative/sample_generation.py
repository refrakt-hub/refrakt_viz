"""
Sample generation and latent interpolation visualizations for generative models.

This module provides visualization components for generating and displaying samples from generative models, as well as visualizing latent space interpolations.

Typical usage:
    from refrakt_viz.generative import SampleGeneration, LatentInterpolation
    viz = SampleGeneration(nrow=8, title="Samples")
    viz.update(samples)
    viz.save("samples.png")

Classes:
    - SampleGeneration: Visualize generated samples from a generative model.
    - LatentInterpolation: Visualize interpolations between latent vectors.
"""

from __future__ import annotations

import os
import random
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np

from refrakt_viz.base import GenerativeVisualizationComponent
from refrakt_viz.registry import register_viz
from refrakt_viz.utils.sample_generation_utils import plot_sample_grid, reshape_image


@register_viz("sample_generation")
class SampleGeneration(GenerativeVisualizationComponent):
    """
        Visualization component for displaying generated samples from a generative model.

        This class accumulates batches of generated samples and provides a method to save a grid
        of randomly selected samples to disk. Useful for monitoring generative model outputs
    during training or evaluation.

        Attributes:
            nrow (int): Number of images per row in the sample grid.
            title (str): Title for the visualization.
            samples (List[np.ndarray[Any, Any]]): List of sample batches to visualize.
    """

    registry_name: str = "sample_generation"

    def __init__(self, nrow: int = 8, title: str = "Sample Generation") -> None:
        """
        Initialize the SampleGeneration visualization.

        Args:
            nrow (int): Number of images per row in the grid.
            title (str): Title for the visualization.
        """
        self.nrow: int = nrow
        self.title: str = title
        self.samples: List[np.ndarray[Any, Any]] = []

    def update(self, *args, **kwargs) -> None:
        """
        Update the visualization with new data. Expects samples, title as arguments.
        """
        samples = kwargs.get("samples", args[0] if len(args) > 0 else None)
        title = kwargs.get("title", args[1] if len(args) > 1 else None)
        if samples is None:
            raise ValueError("samples must be provided to update()")
        self.samples.append(np.array(samples))
        if title is not None:
            self.title = title

    def update_from_batch(
        self, model: Any, batch: Any, loss: float, epoch: int
    ) -> None:
        """
        Generate and add samples from a model and batch.

        Args:
            model (Any): The generative model.
            batch (Any): Input batch for sample generation.
            loss (float): Loss value (unused).
            epoch (int): Current epoch (unused).
        """
        samples = model.generate_samples(batch)
        self.update(samples)

    def _reshape_image(self, img: Any) -> np.ndarray[Any, Any]:
        """
        Reshape an image for visualization (delegates to utils).

        Args:
            img (Any): The image to reshape.
        Returns:
            np.ndarray[Any, Any]: Reshaped image as np.ndarray.
        """
        return reshape_image(img)

    def save(self, path: str) -> None:
        """
        Save a grid of generated samples to disk.

        Args:
            path (str): Output file path for the visualization.
        Raises:
            ValueError: If no samples are available to save.
        """
        if len(self.samples) == 0:
            print("[SampleGeneration] No samples to save.")
            return
        batch_idx: int = random.randint(0, len(self.samples) - 1)
        samples: np.ndarray[Any, Any] = self.samples[batch_idx]
        plot_sample_grid(samples, self.nrow, f"{self.title} (batch {batch_idx})", path)
        self.samples.clear()


@register_viz("latent_interpolation")
class LatentInterpolation(GenerativeVisualizationComponent):
    """
    Visualization component for displaying latent space interpolations in generative models.

    This class accumulates interpolation tasks and provides a method to save the resulting
    interpolated images to disk. Useful for visualizing smooth transitions in the latent space
    of generative models.

    Attributes:
        steps (int): Number of interpolation steps.
        title (str): Title for the visualization.
        interpolations (List[tuple[Any, np.ndarray[Any, Any], np.ndarray[Any, Any]]]):
            List of (model, z_start, z_end) tuples for interpolation.
    """

    def __init__(self, steps: int = 8, title: str = "Latent Interpolation") -> None:
        """
        Initialize the LatentInterpolation visualization.

        Args:
            steps (int): Number of interpolation steps.
            title (str): Title for the visualization.
        """
        self.steps: int = steps
        self.title: str = title
        self.interpolations: List[
            tuple[Any, np.ndarray[Any, Any], np.ndarray[Any, Any]]
        ] = []

    def update(self, model: Any, z_start: Any, z_end: Any) -> None:
        """
        Add a latent interpolation task to the visualization.

        Args:
            model (Any): The generative model.
            z_start (Any): Starting latent vector.
            z_end (Any): Ending latent vector.
        """
        self.interpolations.append((model, np.array(z_start), np.array(z_end)))

    def update_from_batch(
        self, model: Any, batch: Any, loss: float, epoch: int
    ) -> None:
        """
        Add a latent interpolation from a model and batch.

        Args:
            model (Any): The generative model.
            batch (Any): Input batch for interpolation.
            loss (float): Loss value (unused).
            epoch (int): Current epoch (unused).
        """
        z_start, z_end = model.get_interpolation_latents(batch)
        self.update(model, z_start, z_end)

    def save(self, path: str) -> None:
        """
        Save latent space interpolations to disk.

        Args:
            path (str): Output file path for the visualization.
        """
        for idx, (model, z_start, z_end) in enumerate(self.interpolations):
            zs = (
                np.linspace(0, 1, self.steps)[:, None] * z_end
                + (1 - np.linspace(0, 1, self.steps)[:, None]) * z_start
            )
            imgs = np.array(
                [
                    (
                        model.decode(z[None])
                        if hasattr(model, "decode")
                        else model.generate(z[None])
                    )
                    for z in zs
                ]
            )
            imgs = imgs.squeeze()
            plt.figure(figsize=(self.steps * 2, 2))
            for i in range(self.steps):
                plt.subplot(1, self.steps, i + 1)
                plt.imshow(
                    imgs[i],
                    cmap=(
                        "gray" if imgs.shape[-1] == 1 or len(imgs.shape) == 3 else None
                    ),
                )
                plt.axis("off")
            plt.suptitle(self.title)
            plt.tight_layout()
            out_path = path.replace(".png", f"_{idx}.png")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            plt.savefig(out_path)
            plt.close()
            print(f"[LatentInterpolation] Saved latent interpolation to {out_path}")
