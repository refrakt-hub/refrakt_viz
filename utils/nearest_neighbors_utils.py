"""
Nearest neighbor utility functions for contrastive visualizations.

This module provides helper functions for reshaping images and plotting nearest neighbor grids
for contrastive models.

Typical usage:
    from refrakt_viz.utils.nearest_neighbors_utils import reshape_image, plot_nearest_neighbors_grid
    img = reshape_image(raw_img)
    plot_nearest_neighbors_grid(anchors, anchor_imgs, candidates, candidate_imgs, k, title, path)

Functions:
    - reshape_image: Reshape an image for visualization.
    - plot_nearest_neighbors_grid: Plot and save a grid of nearest neighbor retrievals.
"""

import os
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np


def reshape_image(img: Any) -> np.ndarray[Any, Any]:
    """
    Reshape an image for visualization.

    Args:
        img (Any): The image to reshape.
    Returns:
        np.ndarray[Any, Any]: Reshaped image as np.ndarray.
    """
    if hasattr(img, "ndim") and img.ndim == 3 and img.shape[0] in (1, 3):
        if img.shape[0] == 1:
            return np.asarray(img[0])
        elif img.shape[0] == 3:
            return np.transpose(img, (1, 2, 0))
    return np.asarray(img)


def plot_nearest_neighbors_grid(
    anchors: np.ndarray[Any, Any],
    anchor_imgs: List[np.ndarray[Any, Any]],
    candidates: np.ndarray[Any, Any],
    candidate_imgs: List[np.ndarray[Any, Any]],
    k: int,
    title: str,
    path: str,
) -> None:
    """
    Plot and save a grid of nearest neighbor retrievals.

    Args:
        anchors (np.ndarray[Any, Any]): Anchor embeddings.
        anchor_imgs (List[np.ndarray[Any, Any]]): Anchor images.
        candidates (np.ndarray[Any, Any]): Candidate embeddings.
        candidate_imgs (List[np.ndarray[Any, Any]]): Candidate images.
        k (int): Number of nearest neighbors to display.
        title (str): Title for the plot.
        path (str): Output file path for the plot.
    """
    from sklearn.metrics.pairwise import cosine_distances

    n_anchors = len(anchor_imgs)
    dists = cosine_distances(anchors, candidates)
    nn_indices = np.argsort(dists, axis=1)[:, :k]
    fig, axes = plt.subplots(n_anchors, k + 1, figsize=(2 * (k + 1), 2 * n_anchors))
    if n_anchors == 1:
        axes = np.expand_dims(axes, 0)
    for i in range(n_anchors):
        img = reshape_image(anchor_imgs[i])
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Anchor")
        axes[i, 0].axis("off")
        for j in range(k):
            idx = nn_indices[i, j]
            nn_img = reshape_image(candidate_imgs[idx])
            axes[i, j + 1].imshow(nn_img)
            axes[i, j + 1].set_title(f"NN {j+1}")
            axes[i, j + 1].axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()
    print(f"[NearestNeighborsPlot] Saved nearest neighbors grid to {path}")
