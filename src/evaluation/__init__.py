"""Evaluation metrics, scaffold analysis, and interpretability tools."""
from src.evaluation.moleculenet_metrics import (
    compute_auroc,
    compute_rmse,
    compute_mae,
    compute_r2,
    evaluate_classification,
    evaluate_regression,
    get_metrics_fn,
    ScaffoldPerformanceAnalyzer,
    ApplicabilityDomainAnalyzer,
)
from src.evaluation.interpretability import (
    AttentionVisualizer,
    IntegratedGradients,
    SubstructureImportance,
    GNNGradCAM,
)

__all__ = [
    "compute_auroc",
    "compute_rmse",
    "compute_mae",
    "compute_r2",
    "evaluate_classification",
    "evaluate_regression",
    "get_metrics_fn",
    "ScaffoldPerformanceAnalyzer",
    "ApplicabilityDomainAnalyzer",
    "AttentionVisualizer",
    "IntegratedGradients",
    "SubstructureImportance",
    "GNNGradCAM",
]
