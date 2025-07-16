from typing import Any
from abc import ABC, abstractmethod

class VisualizationComponent(ABC):
    @abstractmethod
    def update(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass 