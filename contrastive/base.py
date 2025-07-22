from refrakt_viz.supervised.base import VisualizationComponent
from typing import Any
from abc import abstractmethod

class ContrastiveVisualizationComponent(VisualizationComponent):
    @abstractmethod
    def update_from_batch(self, model: Any, batch: Any, loss: float, epoch: int) -> None:
        ... 