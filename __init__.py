"""
refrakt_viz package initialization.

This module exposes the main visualization component classes for convenient import.
It allows users to access all available visualizations directly from the refrakt_viz package.

Typical usage:
    from refrakt_viz import (
        SampleGeneration, LatentInterpolation, FeatureAttribution, DisentanglementAnalysis,
        ReconstructionViz, LatentSpaceProjection, NearestNeighborsPlot, ClusterAssignmentPlot,
        ContrastiveLossCurvePlot, EmbeddingSpacePlot, PairSimilarityPlot, ComputationGraphPlot,
        SamplePredictionsPlot, PerLayerMetricsPlot, ConfusionMatrixPlot, LossAccuracyPlot
    )
"""

from refrakt_viz.contrastive.cluster_assignments import ClusterAssignmentPlot
from refrakt_viz.contrastive.embedding_space import EmbeddingSpacePlot
from refrakt_viz.contrastive.loss_curve import ContrastiveLossCurvePlot
from refrakt_viz.contrastive.nearest_neighbors import NearestNeighborsPlot
from refrakt_viz.contrastive.pair_similarity import PairSimilarityPlot
from refrakt_viz.generative.disentanglement_analysis import DisentanglementAnalysis
from refrakt_viz.generative.feature_attribution import FeatureAttribution
from refrakt_viz.generative.latent_space_projection import LatentSpaceProjection
from refrakt_viz.generative.reconstruction_viz import ReconstructionViz
from refrakt_viz.generative.sample_generation import (
    LatentInterpolation,
    SampleGeneration,
)
from refrakt_viz.supervised.computation_graph import ComputationGraphPlot
from refrakt_viz.supervised.confusion_matrix import ConfusionMatrixPlot
from refrakt_viz.supervised.loss_accuracy import LossAccuracyPlot
from refrakt_viz.supervised.per_layer_metrics import PerLayerMetricsPlot
from refrakt_viz.supervised.sample_predictions import SamplePredictionsPlot

__all__ = [
    "SampleGeneration",
    "LatentInterpolation",
    "FeatureAttribution",
    "DisentanglementAnalysis",
    "ReconstructionViz",
    "LatentSpaceProjection",
    "NearestNeighborsPlot",
    "ClusterAssignmentPlot",
    "ContrastiveLossCurvePlot",
    "EmbeddingSpacePlot",
    "PairSimilarityPlot",
    "ComputationGraphPlot",
    "SamplePredictionsPlot",
    "PerLayerMetricsPlot",
    "ConfusionMatrixPlot",
    "LossAccuracyPlot",
]
