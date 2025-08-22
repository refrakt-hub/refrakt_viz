"""
Sample generation utility functions for generative visualizations.

This module provides helper functions for reshaping images and plotting sample grids
for generative models.

Typical usage:
    from refrakt_viz.utils.sample_generation_utils import reshape_image, plot_sample_grid
    img = reshape_image(raw_img)
    plot_sample_grid(samples, nrow, title, path)

Functions:
    - reshape_image: Reshape an image for visualization.
    - plot_sample_grid: Plot and save a grid of generated samples.
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
        f"[SampleGeneration] Warning: Cannot reshape flat image of size {img.size}, using zeros fallback."
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
        f"[SampleGeneration] Warning: Unexpected image shape {getattr(img, 'shape', None)}, using zeros fallback."
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
        print("[SampleGeneration] Warning: Received None image, using zeros fallback.")
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
    img: np.ndarray[Any, Any], idx: int, nrow: int, ncol: int
) -> None:
    """
    Plot a single image in the sample grid.

    Args:
        img (np.ndarray[Any, Any]): Image to plot.
        idx (int): Index of the image.
        nrow (int): Number of rows in the grid.
        ncol (int): Number of columns in the grid.
    """
    plt.subplot(ncol, nrow, idx + 1)
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


def plot_sample_grid(
    samples: np.ndarray[Any, Any], nrow: int, title: str, path: str
) -> None:
    """
    Plot and save a grid of generated samples.

    Args:
        samples (np.ndarray[Any, Any]): Array of generated samples.
        nrow (int): Number of rows in the grid.
        title (str): Title for the plot.
        path (str): Output file path for the plot.
    """
    N = len(samples)
    ncol = int(np.ceil(N / nrow))
    plt.figure(figsize=(nrow * 2, ncol * 2))
    for i in range(N):
        img = reshape_image(samples[i])
        _plot_single_image(img, i, nrow, ncol)
    plt.suptitle(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()
    print(f"[SampleGeneration] Saved sample grid to {path}")
