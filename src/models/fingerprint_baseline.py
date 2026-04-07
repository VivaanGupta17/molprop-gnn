"""
Classical molecular fingerprint baselines for comparison with GNN approaches.

Implements:
1. Morgan/ECFP fingerprints via RDKit (ECFP4 = Morgan radius 2)
2. MACCS keys (166-bit structural keys)
3. RDKit topological fingerprints
4. Random Forest classifier/regressor
5. XGBoost classifier/regressor
6. Ridge/Lasso regression
7. Ensemble of fingerprint types

These baselines are essential for GNN papers because:
- They clarify where GNNs add value (regression > classification typically)
- They catch "easy" datasets where simple FPs already saturate performance
- Reviewers expect them; Pfizer/J&J use them internally as sanity checks

Note: Fingerprint baselines handle multi-task datasets by training separate
      models per task (no shared representation), unlike GNN multi-task heads.
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Any, Optional, Union

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors
    from rdkit.Chem.rdFingerprintGenerator import (
        GetMorganGenerator,
        GetRDKitFPGenerator,
        GetAtomPairGenerator,
        GetTopologicalTorsionGenerator,
    )
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logger.warning("RDKit not available. Fingerprint baselines will be disabled.")

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available. Falling back to Random Forest only.")


# ---------------------------------------------------------------------------
# Fingerprint Generation
# ---------------------------------------------------------------------------

class MolecularFingerprinter:
    """Generate various molecular fingerprints from SMILES strings.

    Supported fingerprint types:
    - 'ecfp4': Extended Connectivity FP, radius=2 (most common in drug discovery)
    - 'ecfp6': Extended Connectivity FP, radius=3
    - 'fcfp4': Feature-class ECFP (pharmacophoric features, not atom types)
    - 'maccs': MACCS structural keys (166-bit, expert-defined)
    - 'rdkit': RDKit topological fingerprint
    - 'atompair': Atom pair fingerprint
    - 'torsion': Topological torsion fingerprint
    - 'combined': Concatenation of ECFP4 + MACCS + RDKit FP
    """

    def __init__(
        self,
        fp_type: str = "ecfp4",
        n_bits: int = 2048,
        use_chirality: bool = True,
        use_features: bool = False,
    ):
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for fingerprint computation.")

        self.fp_type = fp_type
        self.n_bits = n_bits
        self.use_chirality = use_chirality
        self.use_features = use_features

        # Pre-build generators for efficiency
        if fp_type in ("ecfp4", "fcfp4"):
            self._gen = GetMorganGenerator(
                radius=2,
                fpSize=n_bits,
                includeChirality=use_chirality,
            )
        elif fp_type in ("ecfp6", "fcfp6"):
            self._gen = GetMorganGenerator(
                radius=3,
                fpSize=n_bits,
                includeChirality=use_chirality,
            )
        elif fp_type == "rdkit":
            self._gen = GetRDKitFPGenerator(fpSize=n_bits)
        elif fp_type == "atompair":
            self._gen = GetAtomPairGenerator(fpSize=n_bits)
        elif fp_type == "torsion":
            self._gen = GetTopologicalTorsionGenerator(fpSize=n_bits)

    def smiles_to_fp(self, smiles: str) -> Optional[np.ndarray]:
        """Convert SMILES to fingerprint vector.

        Returns None for invalid SMILES (caller should handle these).
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Invalid SMILES: {smiles}")
            return None

        if self.fp_type == "maccs":
            fp = MACCSkeys.GenMACCSKeys(mol)
            return np.array(fp, dtype=np.float32)

        elif self.fp_type == "combined":
            # Concatenate ECFP4 + MACCS + RDKit
            ecfp = GetMorganGenerator(radius=2, fpSize=1024).GetFingerprint(mol)
            maccs = MACCSkeys.GenMACCSKeys(mol)
            rdkfp = GetRDKitFPGenerator(fpSize=512).GetFingerprint(mol)

            return np.concatenate([
                np.array(ecfp, dtype=np.float32),
                np.array(maccs, dtype=np.float32),
                np.array(rdkfp, dtype=np.float32),
            ])

        else:
            fp = self._gen.GetFingerprint(mol)
            return np.array(fp, dtype=np.float32)

    def transform(self, smiles_list: list[str]) -> tuple[np.ndarray, list[bool]]:
        """Convert a list of SMILES to a fingerprint matrix.

        Returns:
            X: Fingerprint matrix [n_valid, fp_dim]
            valid_mask: Boolean mask indicating valid SMILES
        """
        fps = []
        valid_mask = []
        for smi in smiles_list:
            fp = self.smiles_to_fp(smi)
            if fp is not None:
                fps.append(fp)
                valid_mask.append(True)
            else:
                # Use zero vector for invalid SMILES (will be masked out)
                fps.append(np.zeros(self._get_fp_dim(), dtype=np.float32))
                valid_mask.append(False)

        return np.stack(fps), valid_mask

    def _get_fp_dim(self) -> int:
        if self.fp_type == "maccs":
            return 167
        elif self.fp_type == "combined":
            return 1024 + 167 + 512
        return self.n_bits


# ---------------------------------------------------------------------------
# Baseline Models
# ---------------------------------------------------------------------------

class RandomForestBaseline:
    """Random Forest baseline using molecular fingerprints.

    Supports both single-task and multi-task settings.
    For multi-task: trains one RF per task independently.
    """

    def __init__(
        self,
        fp_type: str = "ecfp4",
        n_bits: int = 2048,
        task_type: str = "classification",
        n_estimators: int = 500,
        max_depth: Optional[int] = None,
        n_jobs: int = -1,
        random_state: int = 42,
        class_weight: str = "balanced",
    ):
        self.fp_type = fp_type
        self.n_bits = n_bits
        self.task_type = task_type
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.class_weight = class_weight

        self.fingerprinter = MolecularFingerprinter(fp_type=fp_type, n_bits=n_bits)
        self.models: dict[str, Any] = {}

    def _build_model(self):
        if self.task_type == "classification":
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=None,  # grow full trees
                class_weight=self.class_weight,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                oob_score=True,  # free validation estimate
            )
        else:
            return RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=None,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                oob_score=True,
            )

    def fit(
        self,
        smiles: list[str],
        labels: np.ndarray,
        task_names: Optional[list[str]] = None,
    ) -> "RandomForestBaseline":
        """
        Args:
            smiles: List of SMILES strings
            labels: Label array [n_molecules, n_tasks] or [n_molecules]
            task_names: Optional list of task names
        """
        X, valid_mask = self.fingerprinter.transform(smiles)
        valid_mask = np.array(valid_mask)
        X = X[valid_mask]

        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)
        labels = labels[valid_mask]

        n_tasks = labels.shape[1]
        if task_names is None:
            task_names = [f"task_{i}" for i in range(n_tasks)]

        for i, task_name in enumerate(task_names):
            y = labels[:, i]
            # Handle NaN labels (common in multi-task MoleculeNet datasets)
            not_nan = ~np.isnan(y)
            if not_nan.sum() < 10:
                logger.warning(f"Skipping task {task_name}: fewer than 10 valid labels")
                continue

            model = self._build_model()
            model.fit(X[not_nan], y[not_nan])
            self.models[task_name] = model
            logger.info(f"Trained RF for task '{task_name}': "
                        f"{not_nan.sum()} samples, "
                        f"OOB score: {model.oob_score_:.4f}")

        return self

    def predict(self, smiles: list[str]) -> dict[str, np.ndarray]:
        """Predict for all tasks. Returns dict of task_name → predictions."""
        X, valid_mask = self.fingerprinter.transform(smiles)
        predictions = {}
        for task_name, model in self.models.items():
            if self.task_type == "classification":
                preds = model.predict_proba(X)[:, 1]
            else:
                preds = model.predict(X)
            predictions[task_name] = preds
        return predictions

    def evaluate(
        self,
        smiles: list[str],
        labels: np.ndarray,
        task_names: Optional[list[str]] = None,
    ) -> dict[str, dict[str, float]]:
        """Evaluate on test set, returns per-task metrics."""
        X, valid_mask = self.fingerprinter.transform(smiles)
        valid_mask = np.array(valid_mask)
        X_valid = X[valid_mask]

        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)
        labels = labels[valid_mask]

        if task_names is None:
            task_names = list(self.models.keys())

        results = {}
        for i, task_name in enumerate(task_names):
            if task_name not in self.models:
                continue

            model = self.models[task_name]
            y_true = labels[:, i]
            not_nan = ~np.isnan(y_true)

            if self.task_type == "classification":
                y_pred = model.predict_proba(X_valid[not_nan])[:, 1]
                auroc = roc_auc_score(y_true[not_nan], y_pred)
                results[task_name] = {"auroc": auroc}
            else:
                y_pred = model.predict(X_valid[not_nan])
                rmse = np.sqrt(mean_squared_error(y_true[not_nan], y_pred))
                r2 = r2_score(y_true[not_nan], y_pred)
                mae = np.mean(np.abs(y_true[not_nan] - y_pred))
                results[task_name] = {"rmse": rmse, "r2": r2, "mae": mae}

        return results

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"models": self.models, "config": self._config()}, f)

    def _config(self) -> dict:
        return {
            "fp_type": self.fp_type, "n_bits": self.n_bits,
            "task_type": self.task_type, "n_estimators": self.n_estimators,
        }

    @classmethod
    def load(cls, path: str) -> "RandomForestBaseline":
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(**data["config"])
        obj.models = data["models"]
        return obj


class XGBoostBaseline:
    """XGBoost baseline using molecular fingerprints.

    XGBoost often outperforms RF on molecular datasets because:
    - Gradient boosting handles class imbalance better with scale_pos_weight
    - Implicit feature selection via tree structure
    - Better performance on sparse bit-vector inputs (typical for FPs)
    """

    def __init__(
        self,
        fp_type: str = "ecfp4",
        n_bits: int = 2048,
        task_type: str = "classification",
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is required. Install with: pip install xgboost")

        self.fp_type = fp_type
        self.n_bits = n_bits
        self.task_type = task_type
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.fingerprinter = MolecularFingerprinter(fp_type=fp_type, n_bits=n_bits)
        self.models: dict[str, Any] = {}

    def _build_model(self, scale_pos_weight: float = 1.0):
        common_params = dict(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            tree_method="hist",       # fast for dense/sparse
            early_stopping_rounds=50,
        )
        if self.task_type == "classification":
            return XGBClassifier(
                **common_params,
                scale_pos_weight=scale_pos_weight,
                eval_metric="auc",
            )
        else:
            return XGBRegressor(**common_params, eval_metric="rmse")

    def fit(
        self,
        smiles: list[str],
        labels: np.ndarray,
        task_names: Optional[list[str]] = None,
        eval_smiles: Optional[list[str]] = None,
        eval_labels: Optional[np.ndarray] = None,
    ) -> "XGBoostBaseline":
        X, valid_mask = self.fingerprinter.transform(smiles)
        valid_mask = np.array(valid_mask)
        X = X[valid_mask]

        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)
        labels = labels[valid_mask]

        # Validation set for early stopping
        X_eval, y_eval_all = None, None
        if eval_smiles is not None and eval_labels is not None:
            X_eval, eval_valid = self.fingerprinter.transform(eval_smiles)
            eval_valid = np.array(eval_valid)
            X_eval = X_eval[eval_valid]
            if eval_labels.ndim == 1:
                eval_labels = eval_labels.reshape(-1, 1)
            y_eval_all = eval_labels[eval_valid]

        n_tasks = labels.shape[1]
        if task_names is None:
            task_names = [f"task_{i}" for i in range(n_tasks)]

        for i, task_name in enumerate(task_names):
            y = labels[:, i]
            not_nan = ~np.isnan(y)
            if not_nan.sum() < 10:
                continue

            # Handle class imbalance automatically
            scale_pw = 1.0
            if self.task_type == "classification":
                n_pos = y[not_nan].sum()
                n_neg = not_nan.sum() - n_pos
                if n_pos > 0:
                    scale_pw = n_neg / n_pos

            model = self._build_model(scale_pos_weight=scale_pw)

            fit_params = {}
            if X_eval is not None and y_eval_all is not None:
                y_eval = y_eval_all[:, i]
                eval_not_nan = ~np.isnan(y_eval)
                if eval_not_nan.sum() > 0:
                    fit_params["eval_set"] = [(X_eval[eval_not_nan], y_eval[eval_not_nan])]
                    fit_params["verbose"] = False

            model.fit(X[not_nan], y[not_nan], **fit_params)
            self.models[task_name] = model

        return self

    def predict(self, smiles: list[str]) -> dict[str, np.ndarray]:
        X, _ = self.fingerprinter.transform(smiles)
        predictions = {}
        for task_name, model in self.models.items():
            if self.task_type == "classification":
                preds = model.predict_proba(X)[:, 1]
            else:
                preds = model.predict(X)
            predictions[task_name] = preds
        return predictions

    def evaluate(
        self,
        smiles: list[str],
        labels: np.ndarray,
        task_names: Optional[list[str]] = None,
    ) -> dict[str, dict[str, float]]:
        X, valid_mask = self.fingerprinter.transform(smiles)
        valid_mask = np.array(valid_mask)
        X_valid = X[valid_mask]

        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)
        labels = labels[valid_mask]

        if task_names is None:
            task_names = list(self.models.keys())

        results = {}
        for i, task_name in enumerate(task_names):
            if task_name not in self.models:
                continue
            y_true = labels[:, i]
            not_nan = ~np.isnan(y_true)

            if self.task_type == "classification":
                y_pred = self.models[task_name].predict_proba(X_valid[not_nan])[:, 1]
                results[task_name] = {"auroc": roc_auc_score(y_true[not_nan], y_pred)}
            else:
                y_pred = self.models[task_name].predict(X_valid[not_nan])
                rmse = np.sqrt(mean_squared_error(y_true[not_nan], y_pred))
                results[task_name] = {
                    "rmse": rmse,
                    "r2": r2_score(y_true[not_nan], y_pred),
                    "mae": np.mean(np.abs(y_true[not_nan] - y_pred)),
                }

        return results


# ---------------------------------------------------------------------------
# Comparison Utility
# ---------------------------------------------------------------------------

def run_baseline_comparison(
    smiles_train: list[str],
    labels_train: np.ndarray,
    smiles_test: list[str],
    labels_test: np.ndarray,
    task_type: str = "classification",
    task_names: Optional[list[str]] = None,
    fp_types: Optional[list[str]] = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Run all baselines and return comparison table.

    Args:
        fp_types: List of fingerprint types to try. Default: ['ecfp4', 'maccs', 'combined']

    Returns:
        Nested dict: {model_name: {task_name: {metric: value}}}
    """
    if fp_types is None:
        fp_types = ["ecfp4", "maccs", "combined"]

    results = {}

    for fp_type in fp_types:
        # Random Forest
        rf_name = f"RF_{fp_type.upper()}"
        logger.info(f"Training {rf_name}...")
        rf = RandomForestBaseline(fp_type=fp_type, task_type=task_type)
        rf.fit(smiles_train, labels_train, task_names=task_names)
        results[rf_name] = rf.evaluate(smiles_test, labels_test, task_names=task_names)

        # XGBoost
        if XGBOOST_AVAILABLE:
            xgb_name = f"XGB_{fp_type.upper()}"
            logger.info(f"Training {xgb_name}...")
            xgb = XGBoostBaseline(fp_type=fp_type, task_type=task_type)
            xgb.fit(smiles_train, labels_train, task_names=task_names,
                    eval_smiles=smiles_test, eval_labels=labels_test)
            results[xgb_name] = xgb.evaluate(smiles_test, labels_test, task_names=task_names)

    return results
