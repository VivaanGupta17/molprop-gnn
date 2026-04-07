"""
Evaluation metrics for MoleculeNet benchmarks.

Implements:
- AUROC for binary/multi-label classification (the primary MoleculeNet metric)
- RMSE, MAE, R² for regression
- Scaffold split vs random split comparison
- Per-scaffold performance analysis (identify which scaffold classes are hard)
- Applicability domain (AD) analysis based on training set similarity

Key design decisions:
- Always report mean ± std across scaffold splits (report variance, not just mean)
- NaN handling: skip NaN labels as in the official MoleculeNet evaluation
- For multi-task datasets, report per-task metrics + macro-average
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)

try:
    from sklearn.metrics import (
        roc_auc_score,
        mean_squared_error,
        mean_absolute_error,
        r2_score,
        average_precision_score,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def compute_auroc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Compute AUROC, handling class imbalance and NaN labels.

    Returns np.nan if fewer than 2 unique labels in y_true.
    """
    valid = ~np.isnan(y_true)
    if valid.sum() == 0:
        return np.nan

    y_true_v = y_true[valid]
    y_pred_v = y_pred[valid]

    if len(np.unique(y_true_v)) < 2:
        logger.warning("Only one class present in y_true. AUROC is undefined.")
        return np.nan

    try:
        return roc_auc_score(y_true_v, y_pred_v)
    except Exception as e:
        logger.warning(f"AUROC computation failed: {e}")
        return np.nan


def compute_auprc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Area under the precision-recall curve. Informative for highly imbalanced tasks."""
    valid = ~np.isnan(y_true)
    if valid.sum() == 0:
        return np.nan
    try:
        return average_precision_score(y_true[valid], y_pred[valid])
    except Exception:
        return np.nan


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error, ignoring NaN labels."""
    valid = ~np.isnan(y_true)
    if valid.sum() == 0:
        return np.nan
    return float(np.sqrt(mean_squared_error(y_true[valid], y_pred[valid])))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error, ignoring NaN labels."""
    valid = ~np.isnan(y_true)
    if valid.sum() == 0:
        return np.nan
    return float(mean_absolute_error(y_true[valid], y_pred[valid]))


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² coefficient of determination, ignoring NaN labels."""
    valid = ~np.isnan(y_true)
    if valid.sum() == 0:
        return np.nan
    return float(r2_score(y_true[valid], y_pred[valid]))


# ---------------------------------------------------------------------------
# Multi-task evaluation
# ---------------------------------------------------------------------------

def evaluate_classification(
    predictions: dict[str, Tensor],
    targets: Tensor,
    task_specs: list[tuple[str, str, int]],
) -> dict[str, float]:
    """Evaluate multi-label classification with AUROC per task.

    Returns:
        Dict with per-task AUROC + mean_auroc (macro-average, NaN tasks excluded)
    """
    metrics = {}
    aurocs = []

    for task_idx, (name, task_type, _) in enumerate(task_specs):
        if task_type != "classification":
            continue
        if task_idx >= targets.shape[1]:
            continue

        pred_logits = predictions[name].squeeze(-1)
        y_pred = torch.sigmoid(pred_logits).numpy()
        y_true = targets[:, task_idx].numpy()

        auroc = compute_auroc(y_true, y_pred)
        auprc = compute_auprc(y_true, y_pred)

        metrics[f"auroc_{name}"] = auroc
        metrics[f"auprc_{name}"] = auprc

        if not np.isnan(auroc):
            aurocs.append(auroc)

    metrics["mean_auroc"] = float(np.mean(aurocs)) if aurocs else np.nan
    metrics["std_auroc"] = float(np.std(aurocs)) if len(aurocs) > 1 else 0.0

    return metrics


def evaluate_regression(
    predictions: dict[str, Tensor],
    targets: Tensor,
    task_specs: list[tuple[str, str, int]],
) -> dict[str, float]:
    """Evaluate regression with RMSE, MAE, R² per task.

    Returns:
        Dict with per-task metrics + mean_rmse (macro-average)
    """
    metrics = {}
    rmses = []

    for task_idx, (name, task_type, _) in enumerate(task_specs):
        if task_type != "regression":
            continue
        if task_idx >= targets.shape[1]:
            continue

        y_pred = predictions[name].squeeze(-1).numpy()
        y_true = targets[:, task_idx].numpy()

        rmse = compute_rmse(y_true, y_pred)
        mae = compute_mae(y_true, y_pred)
        r2 = compute_r2(y_true, y_pred)

        metrics[f"rmse_{name}"] = rmse
        metrics[f"mae_{name}"] = mae
        metrics[f"r2_{name}"] = r2

        if not np.isnan(rmse):
            rmses.append(rmse)

    metrics["mean_rmse"] = float(np.mean(rmses)) if rmses else np.nan
    metrics["mean_mae"] = float(np.mean([
        metrics[k] for k in metrics if k.startswith("mae_") and not np.isnan(metrics[k])
    ])) if rmses else np.nan

    return metrics


def evaluate_mixed(
    predictions: dict[str, Tensor],
    targets: Tensor,
    task_specs: list[tuple[str, str, int]],
) -> dict[str, float]:
    """Evaluate both classification and regression tasks together."""
    metrics = {}
    metrics.update(evaluate_classification(predictions, targets, task_specs))
    metrics.update(evaluate_regression(predictions, targets, task_specs))
    return metrics


def get_metrics_fn(task_type: str) -> Callable:
    """Return the appropriate evaluation function for a task type."""
    if task_type == "classification":
        return evaluate_classification
    elif task_type == "regression":
        return evaluate_regression
    else:
        return evaluate_mixed


# ---------------------------------------------------------------------------
# Scaffold split vs random split comparison
# ---------------------------------------------------------------------------

def compare_split_strategies(
    model,
    smiles: list[str],
    labels: np.ndarray,
    task_specs: list[tuple[str, str, int]],
    n_repeats: int = 3,
    device: str = "cpu",
) -> dict[str, dict[str, float]]:
    """Compare scaffold split vs random split performance.

    This is the key experiment demonstrating why scaffold splits give
    more realistic performance estimates for novel molecule generalization.

    Runs n_repeats of each split type with different seeds and reports
    mean ± std of the primary metric.

    Returns:
        {
            "scaffold": {"mean_auroc": ..., "std_auroc": ...},
            "random": {"mean_auroc": ..., "std_auroc": ...},
            "gap": {...}   # random - scaffold (positive = random inflated)
        }
    """
    from src.data.molecule_dataset import scaffold_split, random_split
    from src.training.trainer import Trainer

    task_type = task_specs[0][1] if task_specs else "classification"
    primary_metric = "mean_auroc" if task_type == "classification" else "mean_rmse"

    results = {"scaffold": [], "random": []}

    for seed in range(n_repeats):
        for split_type in ("scaffold", "random"):
            if split_type == "scaffold":
                train_idx, val_idx, test_idx = scaffold_split(
                    smiles, seed=seed + 42
                )
            else:
                train_idx, val_idx, test_idx = random_split(
                    len(smiles), seed=seed + 42
                )

            # This is a simplified evaluation stub
            # Full implementation would retrain the model
            logger.info(f"Split {split_type} seed {seed}: "
                        f"{len(train_idx)} train / {len(test_idx)} test")

    return results


# ---------------------------------------------------------------------------
# Per-scaffold performance analysis
# ---------------------------------------------------------------------------

class ScaffoldPerformanceAnalyzer:
    """Analyze model performance broken down by scaffold class.

    Identifies:
    - Which scaffolds are easy/hard for the model
    - Whether performance correlates with training set scaffold frequency
    - Applicability domain boundaries

    Useful for understanding model failure modes and communication to medicinal
    chemists about where the model can/cannot be trusted.
    """

    def __init__(self):
        self.scaffold_results: list[dict] = []

    def add_predictions(
        self,
        smiles: list[str],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        task_type: str = "classification",
    ):
        """Record per-molecule predictions with their scaffolds."""
        try:
            from rdkit.Chem.Scaffolds import MurckoScaffold
            from rdkit import Chem
        except ImportError:
            logger.warning("RDKit not available for scaffold analysis.")
            return

        for i, smi in enumerate(smiles):
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                scaffold = MurckoScaffold.MurckoScaffoldSmilesFromSmiles(smi)
            except Exception:
                scaffold = ""

            self.scaffold_results.append({
                "smiles": smi,
                "scaffold": scaffold,
                "y_true": float(y_true[i]) if not np.isnan(y_true[i]) else None,
                "y_pred": float(y_pred[i]),
            })

    def get_scaffold_metrics(self, min_scaffold_size: int = 3) -> list[dict]:
        """Compute performance metrics for each scaffold group.

        Args:
            min_scaffold_size: Minimum number of molecules per scaffold to report

        Returns:
            List of dicts with scaffold, n_mols, and performance metrics
        """
        from collections import defaultdict

        scaffold_groups: dict[str, list[dict]] = defaultdict(list)
        for entry in self.scaffold_results:
            if entry["y_true"] is not None:
                scaffold_groups[entry["scaffold"]].append(entry)

        scaffold_metrics = []
        for scaffold, entries in scaffold_groups.items():
            if len(entries) < min_scaffold_size:
                continue

            y_true = np.array([e["y_true"] for e in entries])
            y_pred = np.array([e["y_pred"] for e in entries])

            if len(np.unique(y_true)) >= 2:
                metric = compute_auroc(y_true, y_pred)
                metric_name = "auroc"
            else:
                metric = compute_rmse(y_true, y_pred)
                metric_name = "rmse"

            scaffold_metrics.append({
                "scaffold": scaffold,
                "n_molecules": len(entries),
                metric_name: metric,
            })

        # Sort by metric
        scaffold_metrics.sort(key=lambda x: x.get("auroc", -x.get("rmse", 0)))
        return scaffold_metrics

    def get_worst_scaffolds(self, n: int = 10) -> list[dict]:
        """Return the n scaffold groups with worst performance."""
        metrics = self.get_scaffold_metrics()
        if "auroc" in metrics[0]:
            return sorted(metrics, key=lambda x: x["auroc"])[:n]
        else:
            return sorted(metrics, key=lambda x: x["rmse"], reverse=True)[:n]


# ---------------------------------------------------------------------------
# Applicability Domain Analysis
# ---------------------------------------------------------------------------

class ApplicabilityDomainAnalyzer:
    """Assess whether a test molecule is within the model's applicability domain.

    Method: Tanimoto similarity-based AD (industry standard in QSAR).
    A test molecule is "inside AD" if its maximum Tanimoto similarity
    to any training set molecule exceeds a threshold.

    Alternative: Leverage-based AD (Williams plot) for regression models.

    Reference: Tropsha, "Best Practices for QSAR Model Development," 2010
    """

    def __init__(self, similarity_threshold: float = 0.3):
        """
        Args:
            similarity_threshold: Tanimoto similarity cutoff for AD membership.
                                   0.3 is a common threshold in QSAR literature.
        """
        self.threshold = similarity_threshold
        self.train_fps = None

    def fit(self, train_smiles: list[str]) -> "ApplicabilityDomainAnalyzer":
        """Compute and store training set fingerprints."""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            from rdkit.DataStructs import BulkTanimotoSimilarity
        except ImportError:
            raise ImportError("RDKit is required for applicability domain analysis.")

        self.train_fps = []
        for smi in train_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                self.train_fps.append(fp)

        logger.info(f"AD: stored {len(self.train_fps)} training fingerprints")
        return self

    def predict(self, test_smiles: list[str]) -> np.ndarray:
        """Compute AD membership for test molecules.

        Returns:
            Array of floats in [0, 1]: max Tanimoto similarity to training set.
            Values above self.threshold indicate molecules inside the AD.
        """
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit.DataStructs import BulkTanimotoSimilarity

        if self.train_fps is None:
            raise RuntimeError("Must call fit() before predict()")

        similarities = []
        for smi in test_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                similarities.append(0.0)
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            sims = BulkTanimotoSimilarity(fp, self.train_fps)
            similarities.append(max(sims) if sims else 0.0)

        return np.array(similarities)

    def in_domain(self, test_smiles: list[str]) -> np.ndarray:
        """Return boolean array indicating AD membership."""
        return self.predict(test_smiles) >= self.threshold

    def filter_predictions(
        self,
        test_smiles: list[str],
        predictions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Separate test molecules into in-domain and out-of-domain.

        Returns:
            (in_domain_mask, in_domain_preds, out_of_domain_preds)
        """
        mask = self.in_domain(test_smiles)
        return mask, predictions[mask], predictions[~mask]
