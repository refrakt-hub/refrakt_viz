"""
Computation graph visualization for supervised learning models.

This module provides a visualization component for displaying computation graphs of models,
allowing users to inspect the structure and flow of computations.

Typical usage:
    from refrakt_viz.supervised import ComputationGraphPlot
    viz = ComputationGraphPlot()
    viz.update(model, input_tensor, model_name="my_model")
    viz.save("computation_graph.png")

Classes:
    - ComputationGraphPlot: Visualize model computation graphs.
"""

from __future__ import annotations

import os
import shutil
from typing import Any, Optional

from refrakt_viz.base import VisualizationComponent
from refrakt_viz.registry import register_viz
from refrakt_viz.utils.computation_graph_utils import (
    extract_tensor_for_graph,
    move_graph_file,
    render_and_save_dot,
)


@register_viz("computation_graph")
class ComputationGraphPlot(VisualizationComponent):
    """
    Visualization component for displaying computation graphs of models.

    This class provides methods to generate and save computation graphs using torchviz.

    Attributes:
        model_name (Optional[str]): Name of the model.
        graph_dir (str): Directory for saving computation graphs.
        graph_path (Optional[str]): Path to the saved computation graph image.
        last_model (Any): Last model used for graph generation.
        last_input (Any): Last input tensor used for graph generation.
        generated (bool): Whether the graph has been generated.
    """

    def __init__(self) -> None:
        """
        Initialize the ComputationGraphPlot visualization.
        """
        self.model_name: Optional[str] = None
        self.graph_dir: str = os.path.join(".", "visualizations")
        os.makedirs(self.graph_dir, exist_ok=True)
        self.graph_path: Optional[str] = None
        self.last_model: Any = None
        self.last_input: Any = None
        self.generated: bool = False

    def update(
        self, model: Any = None, input_tensor: Any = None, model_name: str = "model"
    ) -> None:
        """
        Update the computation graph with a new model and input.

        Args:
            model (Any): The model to visualize.
            input_tensor (Any): Input tensor for the model.
            model_name (str): Name of the model.
        """
        self.last_model = model
        self.last_input = input_tensor
        self.model_name = model_name
        out_dir = os.path.join(self.graph_dir, self.model_name)
        os.makedirs(out_dir, exist_ok=True)
        self.graph_path = os.path.join(out_dir, "computation_graph.png")
        self._generate_graph()
        self.generated = True

    def update_from_batch(self, model, batch, loss, epoch):
        """
        Update from a model and batch. (Stub implementation)
        """
        pass

    def _generate_graph(self) -> None:
        """
        Generate and save the computation graph using torchviz.

        Raises:
            ImportError: If torchviz is not installed.
            Exception: If graph generation fails.
        """
        try:
            if self.last_model is not None and self.last_input is not None:
                output = self.last_model(self.last_input)
                tensor_for_graph = extract_tensor_for_graph(output)
                out_dir = os.path.join(self.graph_dir, self.model_name or "model")
                os.makedirs(out_dir, exist_ok=True)
                temp_path = os.path.join(out_dir, "computation_graph")
                final_path = render_and_save_dot(
                    tensor_for_graph, self.last_model, out_dir, temp_path
                )
                graph_path = os.path.join(out_dir, "computation_graph.png")
                move_graph_file(final_path, graph_path)
                print(f"[ComputationGraphPlot] Computation graph saved to {graph_path}")
        except ImportError:
            print("[ComputationGraphPlot] torchviz not installed.")
        except (RuntimeError, OSError) as e:
            print(f"[ComputationGraphPlot] Failed to generate graph: {e}")

    def save(self, path: Optional[str] = None) -> None:
        """
        Save the computation graph image to disk.

        Args:
            path (Optional[str]): Output file path for the computation graph image.
        """
        if not self.model_name:
            raise ValueError(
                "[ComputationGraphPlot] Model name not set; "
                "cannot determine computation graph file. "
                "Please set the model_name attribute or pass it as an argument."
            )
        out_dir = os.path.join(self.graph_dir, self.model_name)
        graph_path = os.path.join(out_dir, "computation_graph.png")
        if os.path.exists(graph_path):
            if os.path.getsize(graph_path) == 0:
                print(
                    f"[ComputationGraphPlot] Warning: Graph file {graph_path} is empty."
                )
            dest = path if path is not None else graph_path
            shutil.copy(graph_path, dest)
            print(f"[ComputationGraphPlot] Graph copied to {dest}")
        else:
            print(
                f"[ComputationGraphPlot] Graph file does not exist at {graph_path}. "
                "It should have been generated at the start of training."
            )
