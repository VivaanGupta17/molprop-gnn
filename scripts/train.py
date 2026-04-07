#!/usr/bin/env python3
"""
Training script for MolProp-GNN.

Usage:
    # Train MPNN on BBBP with scaffold split
    python scripts/train.py --dataset bbbp --model mpnn --split scaffold

    # Train GIN on ESOL with custom hyperparameters
    python scripts/train.py --dataset esol --model gin --lr 5e-4 --hidden_dim 300 --epochs 300

    # Run hyperparameter sweep with Optuna
    python scripts/train.py --dataset tox21 --model gat --hparam-sweep --n-trials 50

    # Use config file (override individual args still apply)
    python scripts/train.py --config configs/moleculenet_config.yaml --dataset bbbp
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.featurizer import MolecularFeaturizer
from src.data.molecule_dataset import get_dataloaders, MOLECULENET_DATASETS
from src.evaluation.moleculenet_metrics import get_metrics_fn


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(model_type: str, task_specs: list, args: argparse.Namespace):
    """Instantiate the requested GNN architecture."""
    kwargs = {
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "task_specs": task_specs,
    }

    if model_type == "mpnn":
        from src.models.mpnn import MPNNModel
        return MPNNModel(
            **kwargs,
            edge_dim=args.edge_dim,
            readout=args.readout,
        )
    elif model_type == "gat":
        from src.models.gat_model import GATModel
        # For GAT, hidden_dim is per-head; total = hidden_dim * heads
        gat_kwargs = kwargs.copy()
        gat_kwargs["hidden_dim"] = args.hidden_dim // args.heads  # ensure total = hidden_dim
        return GATModel(
            **gat_kwargs,
            heads=args.heads,
            edge_dim=args.edge_dim,
            jk_mode=args.jk_mode,
            update_edges=True,
        )
    elif model_type == "gin":
        from src.models.gin_model import GINModel
        return GINModel(
            **kwargs,
            edge_dim=args.edge_dim,
            use_virtual_node=True,
            residual=True,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose mpnn | gat | gin")


def get_task_specs(dataset_name: str) -> tuple[list, str]:
    """Get task specifications for a MoleculeNet dataset."""
    config = MOLECULENET_DATASETS[dataset_name]
    task_type = config["task_type"]
    label_cols = config["label_cols"]

    if label_cols is None:
        label_cols = [f"sider_{i}" for i in range(27)]

    task_specs = [(name, task_type, 1) for name in label_cols]
    return task_specs, task_type


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train GNN for molecular property prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument("--dataset", type=str, default="bbbp",
                        choices=list(MOLECULENET_DATASETS.keys()),
                        help="MoleculeNet dataset")
    parser.add_argument("--data-root", type=str, default="./data",
                        help="Root directory for datasets")
    parser.add_argument("--split", type=str, default="scaffold",
                        choices=["scaffold", "random"],
                        help="Dataset split strategy (scaffold recommended)")
    parser.add_argument("--seed", type=int, default=42)

    # Model
    parser.add_argument("--model", type=str, default="gin",
                        choices=["mpnn", "gat", "gin"],
                        help="GNN architecture")
    parser.add_argument("--hidden-dim", dest="hidden_dim", type=int, default=256)
    parser.add_argument("--edge-dim", dest="edge_dim", type=int, default=64)
    parser.add_argument("--num-layers", dest="num_layers", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--readout", type=str, default="attention",
                        choices=["sum", "mean", "attention"],
                        help="Graph readout (MPNN only)")
    parser.add_argument("--heads", type=int, default=4,
                        help="Number of attention heads (GAT only)")
    parser.add_argument("--jk-mode", dest="jk_mode", type=str, default="cat",
                        choices=["cat", "max", "lstm"],
                        help="Jumping Knowledge mode (GAT only)")

    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", dest="weight_decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=30,
                        help="Early stopping patience")
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "plateau"])
    parser.add_argument("--use-amp", dest="use_amp", action="store_true",
                        help="Use automatic mixed precision")
    parser.add_argument("--grad-accum", dest="gradient_accumulation_steps",
                        type=int, default=1)

    # Output
    parser.add_argument("--checkpoint-dir", dest="checkpoint_dir",
                        type=str, default="./checkpoints")
    parser.add_argument("--model-name", dest="model_name", type=str, default=None,
                        help="Name for checkpoint files (default: dataset_model_split)")

    # Config file
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config file (CLI args override config values)")

    # Optuna sweep
    parser.add_argument("--hparam-sweep", dest="hparam_sweep", action="store_true")
    parser.add_argument("--n-trials", dest="n_trials", type=int, default=50)

    # Misc
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cpu/cuda, auto-detect if None)")
    parser.add_argument("--log-level", dest="log_level", type=str, default="INFO")

    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Load config file if provided
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        # Config values are defaults; CLI args override
        dataset_override = cfg.get("dataset_overrides", {}).get(args.dataset, {})
        # Apply dataset-specific defaults (not overriding explicit CLI args)
        logger.info(f"Loaded config from {args.config}")

    # Auto-detect device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {args.device}")

    # Set seed for reproducibility
    set_seed(args.seed)

    # Model name for checkpointing
    if args.model_name is None:
        args.model_name = f"{args.dataset}_{args.model}_{args.split}"

    # -------------------------------------------------------------------
    # Hyperparameter sweep mode
    # -------------------------------------------------------------------
    if args.hparam_sweep:
        logger.info(f"Starting Optuna hyperparameter sweep for {args.dataset} / {args.model}")
        from src.training.trainer import run_hparam_sweep

        model_classes = {
            "mpnn": __import__("src.models.mpnn", fromlist=["MPNNModel"]).MPNNModel,
            "gat": __import__("src.models.gat_model", fromlist=["GATModel"]).GATModel,
            "gin": __import__("src.models.gin_model", fromlist=["GINModel"]).GINModel,
        }

        best_params = run_hparam_sweep(
            model_class=model_classes[args.model],
            dataset_name=args.dataset,
            n_trials=args.n_trials,
            device=args.device,
        )
        logger.info(f"Best hyperparameters: {best_params}")
        return

    # -------------------------------------------------------------------
    # Standard training
    # -------------------------------------------------------------------
    logger.info(f"Dataset: {args.dataset} | Model: {args.model} | Split: {args.split}")

    # Get task specs
    task_specs, task_type = get_task_specs(args.dataset)
    logger.info(f"Task type: {task_type} | {len(task_specs)} tasks")

    # Build model
    model = build_model(args.model, task_specs, args)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {args.model.upper()} | Parameters: {n_params:,}")

    # Load data
    featurizer = MolecularFeaturizer(
        include_pharmacophore=True,
        include_chirality=True,
        include_global=True,
    )
    loaders = get_dataloaders(
        dataset_name=args.dataset,
        root=args.data_root,
        split=args.split,
        batch_size=args.batch_size,
        featurizer=featurizer,
        seed=args.seed,
    )
    logger.info(
        f"Batches per epoch: {len(loaders['train'])} train / "
        f"{len(loaders['val'])} val / {len(loaders['test'])} test"
    )

    # Compute pos weights for imbalanced classification
    pos_weights = None
    if task_type == "classification":
        import numpy as np
        # Collect all training labels
        all_labels = []
        for batch in loaders["train"]:
            all_labels.append(batch.y.numpy())
        labels_array = np.concatenate(all_labels, axis=0).reshape(-1, len(task_specs))
        from src.training.trainer import compute_pos_weights
        task_names = [name for name, _, _ in task_specs]
        pos_weights = compute_pos_weights(labels_array, task_names, max_weight=20.0)
        logger.info(f"Positive class weights (first 5): "
                    f"{dict(list(pos_weights.items())[:5])}")

    # Build trainer
    from src.training.trainer import Trainer
    trainer = Trainer(
        model=model,
        task_specs=task_specs,
        device=args.device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=1.0,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_amp=args.use_amp and torch.cuda.is_available(),
        early_stopping_patience=args.patience,
        checkpoint_dir=args.checkpoint_dir,
        pos_weights=pos_weights,
        scheduler_type=args.scheduler,
    )

    # Evaluation function
    metrics_fn = get_metrics_fn(task_type)
    val_metric = "mean_auroc" if task_type == "classification" else "mean_rmse"
    val_mode = "max" if task_type == "classification" else "min"

    # Train
    results = trainer.train(
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        n_epochs=args.epochs,
        compute_metrics_fn=metrics_fn,
        val_metric=val_metric,
        val_metric_mode=val_mode,
        model_name=args.model_name,
    )

    logger.info(f"Training complete. Best epoch: {results['best_epoch']} | "
                f"Best val {val_metric}: {results['best_val_score']:.4f}")

    # -------------------------------------------------------------------
    # Final test set evaluation
    # -------------------------------------------------------------------
    logger.info("Loading best checkpoint for test evaluation...")
    best_ckpt = Path(args.checkpoint_dir) / f"{args.model_name}_best.pt"
    if best_ckpt.exists():
        checkpoint = torch.load(str(best_ckpt), map_location=args.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Best checkpoint loaded.")

    test_metrics = trainer.evaluate(loaders["test"], metrics_fn)
    logger.info("=" * 60)
    logger.info("TEST SET RESULTS:")
    for key, val in sorted(test_metrics.items()):
        logger.info(f"  {key}: {val:.4f}")
    logger.info("=" * 60)

    # Save results summary
    import json
    results_path = Path(args.checkpoint_dir) / f"{args.model_name}_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "dataset": args.dataset,
            "model": args.model,
            "split": args.split,
            "best_epoch": results["best_epoch"],
            "best_val_score": results["best_val_score"],
            "test_metrics": test_metrics,
            "config": vars(args),
        }, f, indent=2)
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
