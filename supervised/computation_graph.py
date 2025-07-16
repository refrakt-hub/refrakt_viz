from typing import Any
from refrakt_viz.registry import register_viz
from .base import VisualizationComponent
import matplotlib.pyplot as plt
import os

@register_viz("computation_graph")
class ComputationGraphPlot(VisualizationComponent):
    def __init__(self) -> None:
        self.model_name = None
        # Always use a relative path for the visualizations directory
        self.graph_dir = os.path.join(".", "visualizations")
        os.makedirs(self.graph_dir, exist_ok=True)
        self.graph_path = None
        self.last_model = None
        self.last_input = None
        self.generated = False

    def update(self, model: Any = None, input_tensor: Any = None, model_name: str = "model") -> None:
        self.last_model = model
        self.last_input = input_tensor
        self.model_name = model_name
        out_dir = os.path.join(self.graph_dir, self.model_name)
        os.makedirs(out_dir, exist_ok=True)
        self.graph_path = os.path.join(out_dir, "computation_graph.png")
        self._generate_graph()
        self.generated = True

    def _generate_graph(self):
        try:
            from torchviz import make_dot
            if self.last_model is not None and self.last_input is not None:
                output = self.last_model(self.last_input)
                tensor_for_graph = None
                if hasattr(output, 'logits') and output.logits is not None:
                    tensor_for_graph = output.logits
                elif hasattr(output, 'embeddings') and output.embeddings is not None:
                    tensor_for_graph = output.embeddings
                elif hasattr(output, 'reconstruction') and output.reconstruction is not None:
                    tensor_for_graph = output.reconstruction
                elif hasattr(output, 'image') and output.image is not None:
                    tensor_for_graph = output.image
                else:
                    raise ValueError("ModelOutput does not contain a tensor suitable for graph visualization.")
                # Save to a temp path, then move/rename
                out_dir = os.path.join(self.graph_dir, self.model_name)
                os.makedirs(out_dir, exist_ok=True)
                temp_path = os.path.join(out_dir, "computation_graph")
                dot = make_dot(tensor_for_graph, params=dict(self.last_model.named_parameters()))
                dot.format = "png"
                dot.render(temp_path, cleanup=True)
                # torchviz will save as temp_path + '.png'
                final_path = temp_path + ".png"
                graph_path = os.path.join(out_dir, "computation_graph.png")
                if final_path != graph_path:
                    import shutil
                    shutil.move(final_path, graph_path)
                print(f"[ComputationGraphPlot] Computation graph saved to {graph_path}")
        except ImportError:
            print("[ComputationGraphPlot] torchviz not installed.")
        except Exception as e:
            print(f"[ComputationGraphPlot] Failed to generate graph: {e}")

    def save(self, path: str = None) -> None:
        if not self.model_name:
            print("[ComputationGraphPlot] Model name not set; cannot determine computation graph file. Skipping save().")
            return
        out_dir = os.path.join(self.graph_dir, self.model_name)
        graph_path = os.path.join(out_dir, "computation_graph.png")
        if os.path.exists(graph_path):
            if os.path.getsize(graph_path) == 0:
                print(f"[ComputationGraphPlot] Warning: Graph file {graph_path} is empty.")
            import shutil
            dest = path if path is not None else graph_path
            shutil.copy(graph_path, dest)
            print(f"[ComputationGraphPlot] Graph copied to {dest}")
        else:
            print(f"[ComputationGraphPlot] Graph file does not exist at {graph_path}. It should have been generated at the start of training.") 