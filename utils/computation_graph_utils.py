"""
Computation graph utility functions for supervised visualizations.

This module provides helper functions for extracting tensors, rendering computation graphs,
and moving output files for supervised model visualizations.

Typical usage:
    from refrakt_viz.utils.computation_graph_utils import extract_tensor_for_graph, render_and_save_dot, move_graph_file
    tensor = extract_tensor_for_graph(output)
    path = render_and_save_dot(tensor, model, out_dir, temp_path)
    move_graph_file(path, graph_path)

Functions:
    - extract_tensor_for_graph: Extract a tensor suitable for graph visualization from model output.
    - render_and_save_dot: Render and save a computation graph as a PNG file.
    - move_graph_file: Move the rendered graph file to the final destination.
"""

from typing import Any


def extract_tensor_for_graph(output: Any) -> Any:
    """
    Extract a tensor suitable for graph visualization from model output.

    Args:
        output (Any): Model output object.
    Returns:
        Any: The tensor to use for graph visualization.
    Raises:
        ValueError: If no suitable tensor is found in the output.
    """
    if hasattr(output, "logits") and output.logits is not None:
        return output.logits
    elif hasattr(output, "embeddings") and output.embeddings is not None:
        return output.embeddings
    elif hasattr(output, "reconstruction") and output.reconstruction is not None:
        return output.reconstruction
    elif hasattr(output, "image") and output.image is not None:
        return output.image
    else:
        raise ValueError(
            "ModelOutput does not contain a tensor suitable for graph visualization."
        )


def render_and_save_dot(
    tensor_for_graph: Any, model: Any, out_dir: str, temp_path: str
) -> str:
    """
    Render and save a computation graph as a PNG file using torchviz.

    Args:
        tensor_for_graph (Any): Tensor to visualize.
        model (Any): Model whose parameters are visualized.
        out_dir (str): Output directory for the graph.
        temp_path (str): Temporary file path for rendering.
    Returns:
        str: Path to the saved PNG file.
    """
    from torchviz import make_dot

    dot = make_dot(tensor_for_graph, params=dict(model.named_parameters()))
    dot.format = "png"
    dot.render(temp_path, cleanup=True)
    final_path = temp_path + ".png"
    return final_path


def move_graph_file(final_path: str, graph_path: str) -> None:
    """
    Move the rendered computation graph file to the final destination.

    Args:
        final_path (str): Path to the rendered PNG file.
        graph_path (str): Final destination path for the graph file.
    """
    if final_path != graph_path:
        import shutil

        shutil.move(final_path, graph_path)
