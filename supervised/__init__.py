"""
refrakt_viz.supervised package initialization.

This module provides visualization components for supervised learning models in the
Refrakt framework. It includes tools for visualizing computation graphs, sample
predictions, per-layer metrics, confusion matrices, and loss/accuracy curves.

Typical usage:
    from refrakt_viz.supervised import ComputationGraphPlot, SamplePredictionsPlot, \
        PerLayerMetricsPlot, ConfusionMatrixPlot, LossAccuracyPlot

Main classes:
    - ComputationGraphPlot: Visualize model computation graphs.
    - SamplePredictionsPlot: Visualize sample predictions and true labels.
    - PerLayerMetricsPlot: Visualize per-layer activation and gradient metrics.
    - ConfusionMatrixPlot: Visualize confusion matrices for classification.
    - LossAccuracyPlot: Visualize loss and accuracy curves during training.
"""

# No import of the module itself here.
from refrakt_viz.supervised.computation_graph import ComputationGraphPlot
from refrakt_viz.supervised.confusion_matrix import ConfusionMatrixPlot
from refrakt_viz.supervised.loss_accuracy import LossAccuracyPlot
from refrakt_viz.supervised.per_layer_metrics import PerLayerMetricsPlot
from refrakt_viz.supervised.sample_predictions import SamplePredictionsPlot

__all__ = [
    "ComputationGraphPlot",
    "ConfusionMatrixPlot",
    "LossAccuracyPlot",
    "PerLayerMetricsPlot",
    "SamplePredictionsPlot",
]
