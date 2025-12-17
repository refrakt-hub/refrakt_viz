# Refrakt Viz

Unified visualization toolkit for ML/DL model analysis and interpretation. `refrakt_viz` provides a comprehensive suite of visualization components for **generative**, **contrastive**, and **supervised** learning paradigms, enabling researchers and practitioners to visualize, interpret, and analyze their models with a consistent API.

## 🚀 Features

- **Unified Visualization Interface**: Consistent API across all visualization methods
- **Three Visualization Paradigms**: Dedicated components for generative, contrastive, and supervised learning
- **Backend Flexibility**: Matplotlib by default, with optional Plotly/Seaborn for interactive plots
- **Registry System**: Decorator-based registration for easy extensibility
- **XAI Integration**: Seamless integration with `refrakt_xai` attribution outputs
- **Type Safety**: Full type annotations and mypy compliance
- **Training Loop Integration**: `update_from_batch` methods for real-time visualization during training

## 📦 Installation

Since `refrakt_viz` is part of the Refrakt ecosystem, you can install it in several ways:

### Step 1: Clone the repository
```bash
# Clone the repository
git clone https://github.com/refrakt-hub/refrakt_viz.git
cd refrakt_viz
```

### Step 2: Create a virtual environment
```bash
# Option A: Using uv (recommended)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Option B: Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Option C: Using conda
conda create -n refrakt_viz python=3.10
conda activate refrakt_viz
```

### Step 3: Install from requirements
```bash
# Option A (with uv)
uv pip install -e .

# Option B (with pip)
pip install -e .

# With interactive extras (Plotly, Seaborn)
pip install -e ".[interactive]"
```

### Dependencies

**Runtime Dependencies:**
- `matplotlib>=3.0.0` - Static visualization backend
- `numpy>=1.18.0` - Array operations
- `scikit-learn>=0.24.0` - Dimensionality reduction (t-SNE), metrics

**Optional Dependencies (interactive):**
- `seaborn>=0.11.0` - Enhanced statistical plots
- `plotly>=5.0.0` - Interactive visualizations
- `pillow>=8.0.0` - Image processing

**Development Dependencies:**
- `mypy>=1.0.0` - Type checking
- `radon>=5.0.0` - Code complexity analysis
- `lizard>=1.17.10` - Code complexity metrics

## 🏗️ Project Structure

```
refrakt_viz/
├── base.py                 # Base visualization component interface
├── registry.py             # Method registration system
├── contrastive/            # Contrastive learning visualizations
│   ├── cluster_assignments.py
│   ├── embedding_space.py
│   ├── loss_curve.py
│   ├── nearest_neighbors.py
│   └── pair_similarity.py
├── generative/             # Generative model visualizations
│   ├── disentanglement_analysis.py
│   ├── feature_attribution.py
│   ├── latent_space_projection.py
│   ├── reconstruction_viz.py
│   └── sample_generation.py
├── supervised/             # Supervised learning visualizations
│   ├── computation_graph.py
│   ├── confusion_matrix.py
│   ├── loss_accuracy.py
│   ├── per_layer_metrics.py
│   └── sample_predictions.py
└── utils/                  # Utility functions
    ├── computation_graph_utils.py
    ├── disentanglement_analysis_utils.py
    ├── display_image.py
    ├── feature_attribution_utils.py
    ├── nearest_neighbors_utils.py
    ├── reconstruction_viz_utils.py
    └── sample_generation_utils.py
```

## 🎯 Available Visualization Methods

| Category | Method | Description | Use Case |
|----------|--------|-------------|----------|
| **Generative** | `SampleGeneration` | Generate and visualize samples from latent space | VAE/GAN output visualization |
| **Generative** | `LatentInterpolation` | Interpolate between latent vectors | Latent space exploration |
| **Generative** | `FeatureAttribution` | Saliency maps for generated samples | XAI integration |
| **Generative** | `DisentanglementAnalysis` | Analyze latent disentanglement | VAE interpretation |
| **Generative** | `ReconstructionViz` | Input vs reconstruction comparison | Autoencoder evaluation |
| **Generative** | `LatentSpaceProjection` | 2D projection of latent space | t-SNE/UMAP visualization |
| **Contrastive** | `EmbeddingSpacePlot` | Embedding space visualization | SimCLR/DINO analysis |
| **Contrastive** | `NearestNeighborsPlot` | K-nearest neighbors in embedding space | Similarity analysis |
| **Contrastive** | `ClusterAssignmentPlot` | Cluster assignment visualization | Clustering evaluation |
| **Contrastive** | `ContrastiveLossCurvePlot` | Training loss curves | Training monitoring |
| **Contrastive** | `PairSimilarityPlot` | Pairwise similarity matrices | Contrastive pair analysis |
| **Supervised** | `ConfusionMatrixPlot` | Classification confusion matrix | Model evaluation |
| **Supervised** | `LossAccuracyPlot` | Loss and accuracy curves | Training monitoring |
| **Supervised** | `SamplePredictionsPlot` | Sample predictions grid | Qualitative evaluation |
| **Supervised** | `PerLayerMetricsPlot` | Per-layer activation metrics | Model debugging |
| **Supervised** | `ComputationGraphPlot` | Model architecture graph | Architecture visualization |

## 💻 Usage Examples

### Basic Usage - Supervised Learning

```python
from refrakt_viz import ConfusionMatrixPlot, LossAccuracyPlot

# Confusion Matrix
class_names = ["cat", "dog", "bird"]
cm_viz = ConfusionMatrixPlot(class_names=class_names)
cm_viz.update(y_true=[0, 1, 2, 0, 1], y_pred=[0, 1, 1, 0, 2])
cm_viz.save("confusion_matrix.png")

# Loss/Accuracy Curves
loss_viz = LossAccuracyPlot()
for epoch in range(10):
    loss_viz.update(
        epoch=epoch,
        train_loss=0.5 - epoch * 0.04,
        val_loss=0.6 - epoch * 0.03,
        train_acc=0.7 + epoch * 0.02,
        val_acc=0.65 + epoch * 0.02
    )
loss_viz.save("training_curves.png")
```

### Contrastive Learning Visualization

```python
import numpy as np
from refrakt_viz import EmbeddingSpacePlot, NearestNeighborsPlot

# Embedding Space with t-SNE
embeddings = np.random.randn(100, 128)  # 100 samples, 128-dim embeddings
labels = np.random.randint(0, 10, 100)  # 10 classes

viz = EmbeddingSpacePlot()
viz.update(embeddings=embeddings, labels=labels, method="tsne")
viz.save("embedding_space.png")

# Nearest Neighbors Plot
nn_viz = NearestNeighborsPlot()
nn_viz.update(embeddings=embeddings, labels=labels, k=5)
nn_viz.save("nearest_neighbors.png")
```

### Generative Model Visualization

```python
import torch
from refrakt_viz import FeatureAttribution, ReconstructionViz, SampleGeneration

# Feature Attribution (Saliency Maps)
attr_viz = FeatureAttribution(title="VAE Attribution")
attr_viz.update(input_img=input_image, model=vae_model)
attr_viz.save("attribution.png")

# Reconstruction Comparison
recon_viz = ReconstructionViz()
recon_viz.update(original=input_batch, reconstructed=output_batch)
recon_viz.save("reconstruction.png")

# Sample Generation from Latent Space
sample_viz = SampleGeneration()
sample_viz.update(model=vae_model, num_samples=16)
sample_viz.save("generated_samples.png")
```

### Integration with refrakt_xai

```python
from refrakt_xai import SaliencyXAI, IntegratedGradientsXAI
from refrakt_viz import FeatureAttribution

# Generate XAI attribution
model = MyClassifier()
input_tensor = torch.randn(1, 3, 224, 224, requires_grad=True)

xai = SaliencyXAI(model)
attribution = xai.explain(input_tensor, target=0)

# Visualize attribution with refrakt_viz
viz = FeatureAttribution(title="Saliency Map")
viz.update(input_img=input_tensor.detach().numpy(), model=model)
viz.save("xai_visualization.png")
```

### Batch Processing with Training Loop

```python
from refrakt_viz import EmbeddingSpacePlot, ContrastiveLossCurvePlot

# Initialize visualizations
embedding_viz = EmbeddingSpacePlot()
loss_viz = ContrastiveLossCurvePlot()

# Training loop integration
for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(dataloader):
        # Forward pass
        loss = train_step(model, batch)
        
        # Update visualizations from batch
        embedding_viz.update_from_batch(model, batch, loss=loss.item(), epoch=epoch)
        loss_viz.update(epoch=epoch, loss=loss.item())

# Save after training
embedding_viz.save("visualizations/embeddings.png")
loss_viz.save("visualizations/loss_curve.png")
```

### Custom Model Integration

```python
import torchvision.models as models
from refrakt_viz import PerLayerMetricsPlot, ComputationGraphPlot

# Works with any PyTorch model
resnet = models.resnet18(pretrained=True)
resnet.eval()

# Per-layer metrics analysis
layer_viz = PerLayerMetricsPlot()
layer_viz.update(model=resnet, input_tensor=sample_input)
layer_viz.save("per_layer_metrics.png")

# Computation graph visualization
graph_viz = ComputationGraphPlot()
graph_viz.update(model=resnet, input_shape=(1, 3, 224, 224))
graph_viz.save("computation_graph.png")
```

## 🧩 Extending refrakt_viz

### Adding a Custom Visualization

Use the registry system to add new visualization components:

```python
from refrakt_viz.base import VisualizationComponent
from refrakt_viz.registry import register_viz, get_viz

@register_viz("custom_viz")
class CustomVisualization(VisualizationComponent):
    """Custom visualization component."""
    
    def __init__(self, title: str = "Custom Plot"):
        self.title = title
        self.data = []
    
    def update(self, *args, **kwargs) -> None:
        """Update with new data."""
        value = kwargs.get("value", args[0] if args else None)
        self.data.append(value)
    
    def update_from_batch(self, model, batch, loss, epoch) -> None:
        """Update from training batch."""
        self.update(value=loss)
    
    def save(self, path: str) -> None:
        """Save the visualization."""
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(self.data)
        plt.title(self.title)
        plt.savefig(path)
        plt.close()

# Retrieve and use the custom visualization
VizClass = get_viz("custom_viz")
viz = VizClass(title="My Custom Plot")
```

## 📚 Integration with Refrakt

`refrakt_viz` is designed as a core component of the Refrakt ecosystem, providing:

- **refrakt_core Integration**: Visualization components can be attached to training pipelines for real-time monitoring
- **refrakt_xai Integration**: Seamless visualization of attribution maps, saliency outputs, and explanation results
- **refrakt_cli Integration**: Visualizations can be triggered via CLI commands and YAML configurations
- **Scalability**: Methods are optimized for large-scale model analysis and batch processing

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on:

- Setting up the development environment
- Code style and conventions
- Testing requirements
- Pull request process
- Adding new visualization methods

## 📄 License

This project is licensed under the same license as the main Refrakt project. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Built on top of [Matplotlib](https://matplotlib.org/) and [scikit-learn](https://scikit-learn.org/)
- Inspired by visualization tools in the ML/DL research community
- Part of the Refrakt ecosystem for scalable ML/DL workflows

---

**Part of the [Refrakt](https://github.com/refrakt-hub/refrakt) ecosystem** - Natural-language orchestrator for scalable ML/DL workflows.
