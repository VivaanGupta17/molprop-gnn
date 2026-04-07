"""
Training pipeline for molecular property prediction GNNs.

Features:
- Multi-task training with separate loss weighting per task
- Mixed classification (BCEWithLogitsLoss) and regression (MSELoss) losses
- Class imbalance handling via weighted BCE
- Learning rate scheduling (cosine annealing, plateau, warmup)
- Gradient clipping for training stability
- Early stopping with patience
- TensorBoard / Weights & Biases logging
- Optuna hyperparameter optimization
- Checkpoint management
- Reproducible training via seed setting
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    LinearLR,
    SequentialLR,
)
from torch_geometric.loader import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------

class MultiTaskLoss(nn.Module):
    """Combined loss for multi-task molecular property prediction.

    Handles:
    - NaN labels (common in MoleculeNet multi-task datasets, just mask them out)
    - Class imbalance for binary classification via pos_weight
    - Task weighting (can weight classification vs regression tasks differently)
    - Uncertainty-based task weighting (Kendall et al. 2018, optional)

    Args:
        task_specs: List of (name, type, n_out) per task_type description
        pos_weights: Dict of task_name → positive class weight (for imbalanced tasks)
        task_weights: Dict of task_name → loss weight multiplier
        uncertainty_weighting: Use learned task uncertainty weights (Kendall 2018)
    """

    def __init__(
        self,
        task_specs: list[tuple[str, str, int]],
        pos_weights: Optional[dict[str, float]] = None,
        task_weights: Optional[dict[str, float]] = None,
        uncertainty_weighting: bool = False,
    ):
        super().__init__()
        self.task_specs = task_specs
        self.pos_weights = pos_weights or {}
        self.task_weights = task_weights or {name: 1.0 for name, _, _ in task_specs}
        self.uncertainty_weighting = uncertainty_weighting

        if uncertainty_weighting:
            # Log variance parameters (one per task)
            # From Kendall et al., "Multi-Task Learning Using Uncertainty" (NeurIPS 2018)
            self.log_vars = nn.ParameterDict({
                name.replace("-", "_"): nn.Parameter(torch.zeros(1))
                for name, _, _ in task_specs
            })

        # Build per-task loss functions
        self.loss_fns = {}
        for name, task_type, n_out in task_specs:
            if task_type == "classification":
                pos_weight = None
                if name in self.pos_weights:
                    pos_weight = torch.tensor([self.pos_weights[name]])
                self.loss_fns[name] = nn.BCEWithLogitsLoss(
                    pos_weight=pos_weight, reduction="none"
                )
            else:
                self.loss_fns[name] = nn.MSELoss(reduction="none")

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            predictions: Dict of task_name → model output [batch, n_out]
            targets: Label tensor [batch, n_tasks]

        Returns:
            total_loss: Scalar loss
            task_losses: Dict of task_name → loss value (for logging)
        """
        total_loss = torch.tensor(0.0, device=targets.device)
        task_losses = {}

        for task_idx, (name, task_type, n_out) in enumerate(self.task_specs):
            if task_idx >= targets.shape[1]:
                continue

            pred = predictions[name].squeeze(-1)  # [batch]
            y = targets[:, task_idx]              # [batch]

            # Mask out NaN labels (very important for multi-task MoleculeNet!)
            valid_mask = ~torch.isnan(y)
            if valid_mask.sum() == 0:
                continue

            # Compute per-sample loss, then mask and average
            loss_fn = self.loss_fns[name]
            # Move pos_weight to same device if needed
            if isinstance(loss_fn, nn.BCEWithLogitsLoss) and loss_fn.pos_weight is not None:
                loss_fn.pos_weight = loss_fn.pos_weight.to(targets.device)

            per_sample_loss = loss_fn(pred[valid_mask], y[valid_mask])
            task_loss = per_sample_loss.mean()

            # Task weighting
            w = self.task_weights.get(name, 1.0)

            if self.uncertainty_weighting:
                safe_name = name.replace("-", "_")
                log_var = self.log_vars[safe_name]
                # Regression: loss / (2σ²) + log(σ)
                # Classification: loss / σ² + log(σ)
                precision = torch.exp(-log_var)
                if task_type == "regression":
                    task_loss = precision * task_loss / 2 + log_var / 2
                else:
                    task_loss = precision * task_loss + log_var

            total_loss = total_loss + w * task_loss
            task_losses[name] = task_loss.item()

        return total_loss, task_losses


def compute_pos_weights(
    labels: np.ndarray,
    task_names: list[str],
    max_weight: float = 20.0,
) -> dict[str, float]:
    """Compute positive class weights for imbalanced binary classification.

    Weight = n_negative / n_positive (capped at max_weight).
    This is the standard approach for handling class imbalance in drug
    activity prediction, where active compounds are rare.
    """
    weights = {}
    for i, name in enumerate(task_names):
        col = labels[:, i]
        col = col[~np.isnan(col)]
        if len(col) == 0:
            continue
        n_pos = col.sum()
        n_neg = len(col) - n_pos
        if n_pos > 0:
            weights[name] = min(n_neg / n_pos, max_weight)
        else:
            weights[name] = 1.0
    return weights


# ---------------------------------------------------------------------------
# Learning Rate Scheduling
# ---------------------------------------------------------------------------

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    warmup_steps: int = 100,
    total_steps: int = 10000,
    min_lr: float = 1e-6,
    patience: int = 10,
):
    """Build learning rate scheduler with optional linear warmup.

    Warmup prevents large gradient updates early in training when the
    model weights are far from optimal.
    """
    if warmup_steps > 0:
        warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                          total_iters=warmup_steps)
        if scheduler_type == "cosine":
            main = CosineAnnealingLR(
                optimizer, T_max=total_steps - warmup_steps, eta_min=min_lr
            )
        elif scheduler_type == "plateau":
            # ReduceLROnPlateau doesn't chain well with SequentialLR
            # Return warmup only, caller handles plateau separately
            return warmup
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")

        return SequentialLR(optimizer, schedulers=[warmup, main],
                            milestones=[warmup_steps])
    else:
        if scheduler_type == "cosine":
            return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=min_lr)
        elif scheduler_type == "plateau":
            return ReduceLROnPlateau(optimizer, mode="min", patience=patience,
                                     factor=0.5, min_lr=min_lr)
        raise ValueError(f"Unknown scheduler: {scheduler_type}")


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Complete training pipeline for GNN molecular property prediction.

    Supports:
    - Multi-task training
    - Mixed classification + regression
    - Epoch-based and step-based training loops
    - Gradient accumulation for large effective batch sizes
    - Mixed precision (AMP)
    - Early stopping
    - Checkpoint management
    - TensorBoard / W&B logging

    Args:
        model: PyTorch model (MPNN, GAT, GIN, etc.)
        task_specs: List of (name, type, n_out) task descriptors
        device: torch device
        learning_rate: Initial learning rate
        weight_decay: AdamW weight decay (L2 regularization)
        max_grad_norm: Gradient clipping norm
        gradient_accumulation_steps: Accumulate gradients over N batches
        use_amp: Use automatic mixed precision (fp16 forward, fp32 backward)
        early_stopping_patience: Stop if val metric doesn't improve
        checkpoint_dir: Directory for saving checkpoints
    """

    def __init__(
        self,
        model: nn.Module,
        task_specs: list[tuple[str, str, int]],
        device: Union[str, torch.device] = "cuda",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        use_amp: bool = True,
        early_stopping_patience: int = 30,
        checkpoint_dir: str = "./checkpoints",
        pos_weights: Optional[dict[str, float]] = None,
        task_weights: Optional[dict[str, float]] = None,
        scheduler_type: str = "cosine",
        warmup_epochs: int = 5,
    ):
        self.model = model.to(device)
        self.task_specs = task_specs
        self.device = torch.device(device)
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_amp = use_amp and torch.cuda.is_available()
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer (AdamW is standard for GNNs)
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            amsgrad=True,   # more stable in molecular property prediction
        )

        # Loss function
        self.loss_fn = MultiTaskLoss(
            task_specs=task_specs,
            pos_weights=pos_weights,
            task_weights=task_weights,
        )

        # AMP scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # State
        self.epoch = 0
        self.best_val_metric = float("inf")
        self.best_epoch = 0
        self.patience_counter = 0
        self.history: list[dict] = []

        # Scheduler set up in train() once total_steps is known
        self.scheduler = None
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs

    def train_epoch(
        self,
        loader: DataLoader,
        epoch: int,
    ) -> dict[str, float]:
        """Run one training epoch.

        Returns:
            Dict of training metrics for this epoch
        """
        self.model.train()
        total_loss = 0.0
        task_loss_sums: dict[str, float] = {name: 0.0 for name, _, _ in self.task_specs}
        n_batches = 0

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(loader):
            batch = batch.to(self.device)

            # Forward pass
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                predictions = self.model(batch)
                loss, task_losses = self.loss_fn(predictions, batch.y.view(len(batch), -1))
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient step after accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

                # Step-based LR schedulers
                if self.scheduler is not None and self.scheduler_type == "cosine":
                    self.scheduler.step()

            total_loss += loss.item() * self.gradient_accumulation_steps
            for name, val in task_losses.items():
                task_loss_sums[name] += val
            n_batches += 1

        metrics = {"loss": total_loss / n_batches}
        for name in task_loss_sums:
            metrics[f"loss_{name}"] = task_loss_sums[name] / n_batches

        return metrics

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        compute_metrics_fn: Optional[Callable] = None,
    ) -> dict[str, float]:
        """Evaluate on a validation/test set.

        Returns:
            Dict of evaluation metrics
        """
        self.model.eval()

        all_preds: dict[str, list] = {name: [] for name, _, _ in self.task_specs}
        all_targets: list[torch.Tensor] = []
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            batch = batch.to(self.device)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                predictions = self.model(batch)
                loss, _ = self.loss_fn(predictions, batch.y.view(len(batch), -1))

            total_loss += loss.item()
            n_batches += 1

            for name, _, _ in self.task_specs:
                all_preds[name].append(predictions[name].cpu())
            all_targets.append(batch.y.cpu())

        metrics = {"loss": total_loss / n_batches}

        # Concatenate predictions
        concat_preds = {name: torch.cat(preds, dim=0) for name, preds in all_preds.items()}
        concat_targets = torch.cat(all_targets, dim=0).view(-1, len(self.task_specs))

        # Compute task-specific metrics
        if compute_metrics_fn is not None:
            task_metrics = compute_metrics_fn(concat_preds, concat_targets, self.task_specs)
            metrics.update(task_metrics)

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = 100,
        compute_metrics_fn: Optional[Callable] = None,
        val_metric: str = "loss",
        val_metric_mode: str = "min",  # 'min' or 'max'
        model_name: str = "model",
    ) -> dict[str, Any]:
        """Full training loop with validation and early stopping.

        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            n_epochs: Maximum number of epochs
            compute_metrics_fn: Function(preds, targets, task_specs) → metrics dict
            val_metric: Metric name to use for early stopping and model selection
            val_metric_mode: 'min' for loss/RMSE, 'max' for AUROC/R²
            model_name: Used for checkpoint naming

        Returns:
            Training history dict
        """
        # Set up scheduler now that we know n_epochs
        total_steps = n_epochs * len(train_loader)
        warmup_steps = self.warmup_epochs * len(train_loader)
        self.scheduler = build_scheduler(
            self.optimizer,
            scheduler_type=self.scheduler_type,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )

        best_val_score = float("inf") if val_metric_mode == "min" else float("-inf")
        patience_counter = 0

        logger.info(
            f"Starting training: {n_epochs} epochs, "
            f"{len(train_loader)} batches/epoch, "
            f"device={self.device}"
        )

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            self.epoch = epoch

            # Train
            train_metrics = self.train_epoch(train_loader, epoch)

            # Validate
            val_metrics = self.evaluate(val_loader, compute_metrics_fn)

            # Epoch-based LR update (plateau scheduler)
            if self.scheduler_type == "plateau":
                self.scheduler.step(val_metrics.get(val_metric, val_metrics["loss"]))

            epoch_time = time.time() - t0
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Log
            log_msg = (
                f"Epoch {epoch:03d}/{n_epochs} [{epoch_time:.1f}s] "
                f"train_loss={train_metrics['loss']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"lr={current_lr:.2e}"
            )
            if val_metric in val_metrics:
                log_msg += f" val_{val_metric}={val_metrics[val_metric]:.4f}"
            logger.info(log_msg)

            # Record history
            history_entry = {
                "epoch": epoch,
                "lr": current_lr,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            self.history.append(history_entry)

            # Model selection
            current_score = val_metrics.get(val_metric, val_metrics["loss"])
            improved = (
                (val_metric_mode == "min" and current_score < best_val_score) or
                (val_metric_mode == "max" and current_score > best_val_score)
            )

            if improved:
                best_val_score = current_score
                patience_counter = 0
                self.best_epoch = epoch

                # Save best checkpoint
                ckpt_path = self.checkpoint_dir / f"{model_name}_best.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_metric": {val_metric: best_val_score},
                    "history": self.history,
                }, ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    logger.info(
                        f"Early stopping at epoch {epoch}. "
                        f"Best {val_metric}={best_val_score:.4f} at epoch {self.best_epoch}"
                    )
                    break

        logger.info(
            f"Training complete. Best {val_metric}={best_val_score:.4f} at epoch {self.best_epoch}"
        )

        return {
            "best_epoch": self.best_epoch,
            "best_val_score": best_val_score,
            "val_metric": val_metric,
            "history": self.history,
        }


# ---------------------------------------------------------------------------
# Optuna Hyperparameter Optimization
# ---------------------------------------------------------------------------

def run_hparam_sweep(
    model_class,
    dataset_name: str,
    n_trials: int = 50,
    timeout: Optional[int] = None,
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    n_epochs_per_trial: int = 50,
    device: str = "cuda",
) -> dict:
    """Run Optuna hyperparameter sweep.

    Searches over:
    - hidden_dim: [128, 256, 512]
    - num_layers: [2, 3, 4, 5, 6]
    - learning_rate: log-uniform [1e-4, 1e-2]
    - dropout: [0.0, 0.1, 0.2, 0.3]
    - weight_decay: log-uniform [1e-6, 1e-3]
    - batch_size: [32, 64, 128]

    Args:
        model_class: Model class (MPNNModel, GATModel, GINModel)
        dataset_name: MoleculeNet dataset name
        n_trials: Number of Optuna trials
        n_epochs_per_trial: Max epochs per trial (use early stopping)

    Returns:
        Best hyperparameters dict
    """
    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError:
        raise ImportError("Optuna is required for hyperparameter sweeps. "
                          "Install with: pip install optuna")

    from src.data.molecule_dataset import get_dataloaders
    from src.evaluation.moleculenet_metrics import get_metrics_fn

    def objective(trial: optuna.Trial) -> float:
        # Sample hyperparameters
        hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 300, 512])
        num_layers = trial.suggest_int("num_layers", 2, 6)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.3, step=0.05)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

        # Get data
        loaders = get_dataloaders(
            dataset_name, split="scaffold", batch_size=batch_size
        )

        # Build model
        from src.data.molecule_dataset import MOLECULENET_DATASETS
        config = MOLECULENET_DATASETS[dataset_name]
        label_cols = config["label_cols"] or [f"task_{i}" for i in range(27)]
        task_type = config["task_type"]
        task_specs = [(name, task_type, 1) for name in label_cols]

        model = model_class(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            task_specs=task_specs,
        )

        trainer = Trainer(
            model=model,
            task_specs=task_specs,
            device=device,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            early_stopping_patience=10,
            checkpoint_dir=f"/tmp/optuna_trial_{trial.number}",
        )

        metrics_fn = get_metrics_fn(task_type)
        val_metric = "mean_auroc" if task_type == "classification" else "mean_rmse"
        val_mode = "max" if task_type == "classification" else "min"

        results = trainer.train(
            loaders["train"], loaders["val"],
            n_epochs=n_epochs_per_trial,
            compute_metrics_fn=metrics_fn,
            val_metric=val_metric,
            val_metric_mode=val_mode,
            model_name=f"trial_{trial.number}",
        )

        return results["best_val_score"]

    sampler = TPESampler(seed=42)
    direction = "maximize" if "auroc" in dataset_name else "minimize"

    from src.data.molecule_dataset import MOLECULENET_DATASETS
    task_type = MOLECULENET_DATASETS[dataset_name]["task_type"]
    direction = "maximize" if task_type == "classification" else "minimize"

    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        study_name=study_name or f"{dataset_name}_sweep",
        storage=storage,
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=1)

    logger.info(f"Best trial: {study.best_trial.params}")
    logger.info(f"Best value: {study.best_value:.4f}")

    return study.best_trial.params
