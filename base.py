"""
Base classes for visualization components in refrakt_viz.

This module defines abstract base classes for all visualization components in the Refrakt framework.
It provides a unified interface for generative, contrastive, and supervised visualizations.

Typical usage:
    from refrakt_viz.base import (
        VisualizationComponent, GenerativeVisualizationComponent, ContrastiveVisualizationComponent
    )

Classes:
    - VisualizationComponent: Abstract base class for all visualizations.
    - GenerativeVisualizationComponent: Base class for generative visualizations.
    - ContrastiveVisualizationComponent: Base class for contrastive visualizations.
"""

from abc import ABC, abstractmethod
from typing import Any


class VisualizationComponent(ABC):
    """
    Abstract base class for all visualization components in refrakt_viz.

    All visualization components must implement the update, save, and update_from_batch methods.
    """

    @abstractmethod
    def update(self, *args: Any, **kwargs: Any) -> None:
        """
        Update the visualization with new data.

        Args:
            *args: Positional arguments for the update.
            **kwargs: Keyword arguments for the update.
        """
        raise NotImplementedError()

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the visualization to disk.

        Args:
            path (str): Output file path for the visualization.
        """
        raise NotImplementedError()

    @abstractmethod
    def update_from_batch(
        self, model: Any, batch: Any, loss: float, epoch: int
    ) -> None:
        """
        Update the visualization from a model and batch.

        Args:
            model (Any): The model used for the update.
            batch (Any): Input batch for the update.
            loss (float): Loss value (optional).
            epoch (int): Current epoch (optional).
        """
        raise NotImplementedError()


class GenerativeVisualizationComponent(VisualizationComponent):
    """
    Base class for generative visualization components.

    All generative visualizations must implement update_from_batch for generative models.
    """

    @abstractmethod
    def update_from_batch(
        self, model: Any, batch: Any, loss: float, epoch: int
    ) -> None:
        """
        Update the generative visualization from a model and batch.

        Args:
            model (Any): The generative model.
            batch (Any): Input batch for the update.
            loss (float): Loss value (optional).
            epoch (int): Current epoch (optional).
        """
        raise NotImplementedError()


class ContrastiveVisualizationComponent(VisualizationComponent):
    """
    Base class for contrastive visualization components.

    All contrastive visualizations must implement update_from_batch for contrastive models.
    """

    @abstractmethod
    def update_from_batch(
        self, model: Any, batch: Any, loss: float, epoch: int
    ) -> None:
        """
        Update the contrastive visualization from a model and batch.

        Args:
            model (Any): The contrastive model.
            batch (Any): Input batch for the update.
            loss (float): Loss value (optional).
            epoch (int): Current epoch (optional).
        """
        raise NotImplementedError()
