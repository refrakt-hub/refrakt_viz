"""
Reconstruction visualization utility functions for generative visualizations.

This module provides helper functions for reshaping images and plotting input/reconstruction grids
for generative models.

Typical usage:
    from refrakt_viz.utils.reconstruction_viz_utils import reshape_image, plot_reconstruction_grid
    img = reshape_image(raw_img)
    plot_reconstruction_grid(inputs, recons, n_samples, title, path)

Functions:
    - reshape_image: Reshape an image for visualization.
    - plot_reconstruction_grid: Plot and save a grid of input/reconstruction pairs.
"""

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _reshape_flat_image(img: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """
    Reshape a flat image to a square or fallback shape.

    Args:
        img (np.ndarray[Any, Any]): Flat image array.
    Returns:
        np.ndarray[Any, Any]: Reshaped image.
    """
    side = int(np.sqrt(img.size))
    if side * side == img.size:
        return img.reshape(side, side)
    print(
        f"[ReconstructionViz] Warning: Cannot reshape flat image of size {img.size}, using zeros fallback."
    )
    return np.zeros((28, 28), dtype=np.float32)


def _reshape_channel_first(img: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """
    Reshape a channel-first image to (H, W) or (H, W, C).

    Args:
        img (np.ndarray[Any, Any]): Channel-first image array.
    Returns:
        np.ndarray[Any, Any]: Reshaped image.
    """
    if img.shape[0] == 1:
        return np.asarray(img[0])
    elif img.shape[0] == 3:
        return np.transpose(img, (1, 2, 0))
    return img


def _reshape_channel_last(img: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """
    Return a channel-last image as is.

    Args:
        img (np.ndarray[Any, Any]): Channel-last image array.
    Returns:
        np.ndarray[Any, Any]: The input image.
    """
    return img


def _reshape_unexpected(img: Any) -> np.ndarray[Any, Any]:
    """
    Handle unexpected image shapes by returning a fallback image.

    Args:
        img (Any): The image to handle.
    Returns:
        np.ndarray[Any, Any]: Fallback image.
    """
    print(
        f"[ReconstructionViz] Warning: Unexpected image shape {getattr(img, 'shape', None)}, using zeros fallback."
    )
    return np.zeros((28, 28), dtype=np.float32)


def reshape_image(img: Any) -> np.ndarray[Any, Any]:
    """
    Reshape an image for visualization.

    Args:
        img (Any): The image to reshape.
    Returns:
        np.ndarray[Any, Any]: Reshaped image as np.ndarray.
    """
    if img is None:
        print(
            "[ReconstructionViz] Warning: Received None image, using zeros fallback."
        )
        return np.zeros((28, 28), dtype=np.float32)
    img = np.array(img)
    if img.ndim == 1:
        return _reshape_flat_image(img)
    elif img.ndim == 2:
        return img
    elif img.ndim == 3 and img.shape[0] in [1, 3]:
        return _reshape_channel_first(img)
    elif img.ndim == 3 and img.shape[-1] in [1, 3]:
        return _reshape_channel_last(img)
    return _reshape_unexpected(img)


def _plot_single_image(
    img: np.ndarray[Any, Any], idx: int, n_samples: int, is_recon: bool
) -> None:
    """
    Plot a single image in the reconstruction grid.

    Args:
        img (np.ndarray[Any, Any]): Image to plot.
        idx (int): Index of the image.
        n_samples (int): Number of samples in the grid.
        is_recon (bool): Whether the image is a reconstruction.
    """
    plt.subplot(2, n_samples, (n_samples if is_recon else 0) + idx + 1)
    try:
        img = np.array(img, dtype=np.float32)
    except Exception:
        img = np.zeros((28, 28), dtype=np.float32)
    plt.imshow(
        img,
        cmap=(
            "gray" if img.ndim == 2 or (img.ndim == 3 and img.shape[-1] == 1) else None
        ),
    )
    plt.axis("off")
    if idx == 0:
        plt.ylabel("Recon" if is_recon else "Input")


def plot_reconstruction_grid(
    inputs: np.ndarray[Any, Any],
    recons: np.ndarray[Any, Any],
    n_samples: int,
    title: str,
    path: str,
) -> None:
    """
    Plot and save a grid of input and reconstruction pairs.

    Args:
        inputs (np.ndarray[Any, Any]): Array of input images.
        recons (np.ndarray[Any, Any]): Array of reconstructed images.
        n_samples (int): Number of samples to display.
        title (str): Title for the plot.
        path (str): Output file path for the plot.
    """
    plt.figure(figsize=(n_samples * 2, 4))
    for i in range(n_samples):
        img = reshape_image(inputs[i])
        _plot_single_image(img, i, n_samples, is_recon=False)
        recon_img = reshape_image(recons[i])
        _plot_single_image(recon_img, i, n_samples, is_recon=True)
    plt.suptitle(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()
    print(f"[ReconstructionViz] Saved reconstruction visualization to {path}")
