"""
Feature attribution utility functions for generative visualizations.

This module provides helper functions for extracting model outputs, computing saliency maps,
and plotting feature attributions for generative models.

Typical usage:
    from refrakt_viz.utils.feature_attribution_utils import extract_output_tensor, compute_saliency, plot_saliency
    saliency = compute_saliency(input_tensor, model)
    plot_saliency(input_img, saliency, idx, title, path)

Functions:
    - extract_output_tensor: Extract the relevant output tensor from a model output.
    - compute_saliency: Compute a saliency map for a given input and model.
    - plot_saliency: Plot and save a saliency map for a given input.
"""

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


def extract_output_tensor(output: Any) -> torch.Tensor:
    """
    Extract the relevant output tensor from a model output for saliency computation.

    Args:
        output (Any): Model output object or tensor.
    Returns:
        torch.Tensor: The tensor to use for saliency computation.
    Raises:
        ValueError: If no suitable tensor is found in the output.
    """
    if hasattr(output, "reconstruction") and output.reconstruction is not None:
        return output.reconstruction  # type: ignore[no-any-return]
    elif hasattr(output, "image") and output.image is not None:
        return output.image  # type: ignore[no-any-return]
    elif hasattr(output, "logits") and output.logits is not None:
        return output.logits  # type: ignore[no-any-return]
    elif isinstance(output, torch.Tensor):
        return output
    else:
        raise ValueError(
            "Model output does not contain a tensor for saliency computation."
        )


def compute_saliency(input_tensor: torch.Tensor, model: Any) -> np.ndarray[Any, Any]:
    """
    Compute a saliency map for a given input tensor and model.

    Args:
        input_tensor (torch.Tensor): Input tensor for which to compute saliency.
        model (Any): Model to use for saliency computation.
    Returns:
        np.ndarray[Any, Any]: Computed saliency map.
    Raises:
        RuntimeError: If input_tensor.grad is None after backward pass.
    """
    model_cpu = model.cpu() if hasattr(model, "cpu") else model
    input_tensor = input_tensor.cpu()
    output = model_cpu(input_tensor)
    out_tensor = extract_output_tensor(output)
    loss = out_tensor.norm()
    loss.backward()
    if input_tensor.grad is None:
        raise RuntimeError("input_tensor.grad is None after backward pass.")
    saliency = input_tensor.grad.abs().detach().cpu().numpy().squeeze()
    return saliency


def plot_saliency(
    input_img: np.ndarray[Any, Any],
    saliency: np.ndarray[Any, Any],
    idx: int,
    title: str,
    path: str,
) -> None:
    """
    Plot and save a saliency map for a given input image.

    Args:
        input_img (np.ndarray[Any, Any]): Input image.
        saliency (np.ndarray[Any, Any]): Saliency map to plot.
        idx (int): Index of the sample (for labeling).
        title (str): Title for the plot.
        path (str): Output file path for the plot.
    """
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(
        input_img.squeeze(),
        cmap="gray" if input_img.shape[-1] == 1 or len(input_img.shape) == 2 else None,
    )
    plt.title(f"Input (batch {idx})")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(saliency, cmap="hot")
    plt.title("Saliency")
    plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()
    print(f"[FeatureAttribution] Saved saliency map to {path} (batch {idx})")
