"""
Disentanglement analysis utility functions for generative visualizations.

This module provides helper functions for inferring expected shapes, reshaping images, traversing latent dimensions,
and plotting disentanglement grids for generative models.

Typical usage:
    from refrakt_viz.utils.disentanglement_analysis_utils import (
        infer_expected_shape, reshape_img, traverse_latent_dim, plot_disentanglement_grid
    )
    expected_shape = infer_expected_shape(model, hidden_dim, device)
    imgs = traverse_latent_dim(z_base, dim, dim_range, steps, model, device, expected_shape)
    plot_disentanglement_grid(imgs_np, latent_dim, steps, title, path, expected_shape)

Functions:
    - infer_expected_shape: Infer the expected output shape from a generative model.
    - reshape_img: Reshape an image for visualization.
    - traverse_latent_dim: Traverse a latent dimension and collect images.
    - plot_disentanglement_grid: Plot and save a disentanglement grid.
"""

import os
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def infer_expected_shape(
    model: Any, hidden_dim: int, device: Any
) -> Optional[Tuple[int, ...]]:
    """
    Infer the expected output shape from a generative model.

    Args:
        model (Any): The generative model.
        hidden_dim (int): Hidden dimension size.
        device (Any): Device for computation.
    Returns:
        Optional[Tuple[int, ...]]: The expected output shape, or None if it cannot be inferred.
    """
    with torch.no_grad():
        sample = model.decode(torch.zeros(1, hidden_dim, device=device))
    if isinstance(sample, torch.Tensor):
        sample = sample.detach().cpu().numpy()
    if sample.ndim == 4:
        return sample.shape[1:]  # type: ignore[no-any-return]
    elif sample.ndim == 3:
        if sample.shape[0] in [1, 3]:
            return sample.shape[1:]  # type: ignore[no-any-return]
        else:
            return sample.shape  # type: ignore[no-any-return]
    elif sample.ndim == 2:
        return sample.shape  # type: ignore[no-any-return]
    return None


def reshape_img(
    img: np.ndarray[Any, Any], expected_shape: Optional[Tuple[int, ...]]
) -> np.ndarray[Any, Any]:
    """
    Reshape an image for visualization.

    Args:
        img (np.ndarray[Any, Any]): The image to reshape.
        expected_shape (Optional[Tuple[int, ...]]): The expected shape for the image.
    Returns:
        np.ndarray[Any, Any]: Reshaped image as np.ndarray.
    """
    if img.ndim == 1 and expected_shape is not None:
        img = img.reshape(expected_shape)
    elif img.ndim == 2 and expected_shape and img.shape != expected_shape:
        img = img.reshape(expected_shape)
    if img.ndim == 3 and img.shape[0] in [1, 3]:
        img = np.transpose(img, (1, 2, 0))
    return img


def traverse_latent_dim(
    z_base: np.ndarray[Any, Any],
    dim: int,
    dim_range: float,
    steps: int,
    model: Any,
    device: Any,
    expected_shape: Optional[Tuple[int, ...]],
) -> List[np.ndarray[Any, Any]]:
    """
    Traverse a latent dimension and collect images for disentanglement analysis.

    Args:
        z_base (np.ndarray[Any, Any]): Base latent vector.
        dim (int): Latent dimension to traverse.
        dim_range (float): Range of values to traverse.
        steps (int): Number of steps to traverse.
        model (Any): The generative model.
        device (Any): Device for computation.
        expected_shape (Optional[Tuple[int, ...]]): Expected output shape.
    Returns:
        List[np.ndarray[Any, Any]]: List of images for each step in the traversal.
    Raises:
        AttributeError: If the model does not have a decode or generate method.
    """
    row = []
    for v in np.linspace(-dim_range, dim_range, steps):
        z = z_base.copy()
        z[dim] = v
        z_tensor = torch.as_tensor(z[None], dtype=torch.float32, device=device)
        if hasattr(model, "decode"):
            img = model.decode(z_tensor)
        elif hasattr(model, "generate"):
            img = model.generate(z_tensor)
        else:
            raise AttributeError(
                "Model must have either a 'decode' or 'generate' method for disentanglement analysis."
            )
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        img = reshape_img(img, expected_shape)
        img = np.clip(img, 0, 1)
        row.append(img.squeeze())
    return row


def plot_disentanglement_grid(
    imgs: np.ndarray[Any, Any],
    latent_dim: int,
    steps: int,
    title: str,
    path: str,
    expected_shape: Optional[Tuple[int, ...]],
) -> None:
    """
    Plot and save a disentanglement grid for latent traversals.

    Args:
        imgs (np.ndarray[Any, Any]): Array of images for the grid.
        latent_dim (int): Number of latent dimensions.
        steps (int): Number of steps per dimension.
        title (str): Title for the plot.
        path (str): Output file path for the plot.
        expected_shape (Optional[Tuple[int, ...]]): Expected output shape.
    """
    plt.figure(figsize=(steps * 2, latent_dim * 2))
    for i in range(latent_dim):
        for j in range(steps):
            img = imgs[i, j]
            img = reshape_img(img, expected_shape)
            cmap = (
                "gray"
                if (img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1))
                else None
            )
            plt.subplot(latent_dim, steps, i * steps + j + 1)
            plt.imshow(img, cmap=cmap)
            plt.axis("off")
            if j == 0:
                plt.ylabel(f"dim {i}")
    plt.suptitle(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()
    print(f"[DisentanglementAnalysis] Saved disentanglement analysis to {path}")
