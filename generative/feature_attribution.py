"""
Feature attribution visualization for generative models.

This module provides a visualization component for feature attribution in generative models,
allowing users to visualize saliency maps or other attribution methods for generated samples.

Typical usage:
    from refrakt_viz.generative import FeatureAttribution
    viz = FeatureAttribution(title="Attribution")
    viz.update(input_img, model)
    viz.save("attribution.png")

Classes:
    - FeatureAttribution: Visualize feature attributions for generative models.
"""

from __future__ import annotations

import random
from typing import Any, List

import numpy as np

from refrakt_viz.base import GenerativeVisualizationComponent
from refrakt_viz.registry import register_viz
from refrakt_viz.utils.feature_attribution_utils import compute_saliency, plot_saliency


@register_viz("feature_attribution")
class FeatureAttribution(GenerativeVisualizationComponent):
    """
    Visualization component for feature attribution in generative models.

    This class accumulates input/model pairs and provides a method to compute and save
    saliency maps or other feature attributions for generated samples.

    Attributes:
        title (str): Title for the visualization.
        inputs (List[np.ndarray[Any, Any]]): List of input images.
        models (List[Any]): List of models corresponding to each input.
    """

    registry_name: str = "feature_attribution"

    def __init__(self, title: str = "Feature Attribution") -> None:
        """
        Initialize the FeatureAttribution visualization.

        Args:
            title (str): Title for the visualization.
        """
        self.title: str = title
        self.inputs: List[np.ndarray[Any, Any]] = []
        self.models: List[Any] = []

    def update(self, *args, **kwargs) -> None:
        """
        Update the visualization with new data. Expects input_img, model as arguments.
        """
        input_img = kwargs.get("input_img", args[0] if len(args) > 0 else None)
        model = kwargs.get("model", args[1] if len(args) > 1 else None)
        self.inputs.append(np.array(input_img))
        self.models.append(model)

    def update_from_batch(
        self, model: Any, batch: Any, loss: float, epoch: int
    ) -> None:
        """
        Add an input/model pair from a batch for attribution visualization.

        Args:
            model (Any): Model to use for attribution.
            batch (Any): Input batch for attribution.
            loss (float): Loss value (unused).
            epoch (int): Current epoch (unused).
        """
        input_img = model.get_saliency_input(batch)
        self.update(input_img, model)

    def save(self, path: str) -> None:
        """
        Compute and save a saliency map for a randomly selected input/model pair.

        Args:
            path (str): Output file path for the visualization.
        Raises:
            RuntimeError: If saliency computation fails due to missing gradients.
        """
        import torch

        if not self.inputs:
            print("[FeatureAttribution] No inputs to save.")
            return
        idx: int = random.randint(0, len(self.inputs) - 1)
        input_img = self.inputs[idx]
        model = self.models[idx]
        input_tensor = torch.tensor(input_img, dtype=torch.float32, requires_grad=True)
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor.unsqueeze(0)
        try:
            saliency = compute_saliency(input_tensor, model)
        except RuntimeError:
            print(
                f"[FeatureAttribution] Warning: input_tensor.grad is None for sample {idx}, skipping."
            )
            return
        plot_saliency(input_img, saliency, idx, self.title, path)
        self.inputs.clear()
        self.models.clear()
