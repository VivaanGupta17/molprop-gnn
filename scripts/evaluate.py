#!/usr/bin/env python3
"""
Evaluation script for trained MolProp-GNN models.

Usage:
    # Evaluate on test set with full metrics
    python scripts/evaluate.py \\
        --checkpoint checkpoints/bbbp_mpnn_scaffold_best.pt \\
        --dataset bbbp --split scaffold

    # Compare scaffold vs random split performance
    python scripts/evaluate.py \\
        --checkpoint checkpoints/bbbp_gin_scaffold_best.pt \\
        --dataset bbbp --compare-splits

    # Run applicability domain analysis
    python scripts/evaluate.py \\
        --checkpoint checkpoints/esol_gin_scaffold_best.pt \\
        --dataset esol --ad-analysis

    # Full interpretability report for a specific molecule
    python scripts/evaluate.py \\
        --checkpoint checkpoints/esol_gin_scaffold_best.pt \\
        --dataset esol --smiles "CC(=O)Oc1ccccc1C(=O)O" --interpret
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.featurizer import MolecularFeaturizer
from src.data.molecule_dataset import get_dataloaders, MOLECULENET_DATASETS


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_checkpoint(path: str, device: str = "cpu"):
    """Load checkpoint and return model + metadata."""
    checkpoint = torch.load(path, map_location=device)
    return checkpoint


def build_model_from_checkpoint(checkpoint: dict, device: str = "cpu"):
    """Reconstruct model from checkpoint metadata."""
    config = checkpoint.get("config", {})
    model_type = config.get("model", "gin")
    task_specs = config.get("task_specs", [("property", "regression", 1)])

    if model_type == "mpnn":
        from src.models.mpnn import MPNNModel
        model = MPNNModel(task_specs=task_specs)
    elif model_type == "gat":
        from src.models.gat_model import GATModel
        model = GATModel(task_specs=task_specs)
    elif model_type == "gin":
        from src.models.gin_model import GINModel
        model = GINModel(task_specs=task_specs)
    else:
        raise ValueError(f"Unknown model type in checkpoint: {model_type}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, model_type


def print_metrics_table(metrics: dict, title: str = "Results"):
    """Pretty-print a metrics dictionary as a table."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    for key, val in sorted(metrics.items()):
        if isinstance(val, float):
            print(f"  {key:<40} {val:.4f}")
        else:
            print(f"  {key:<40} {val}")
    print(f"{'='*60}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained MolProp-GNN model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=list(MOLECULENET_DATASETS.keys()))
    parser.add_argument("--data-root", dest="data_root", type=str, default="./data")
    parser.add_argument("--split", type=str, default="scaffold",
                        choices=["scaffold", "random"])
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", dest="output_dir", type=str, default="./results")

    # Analysis options
    parser.add_argument("--compare-splits", dest="compare_splits", action="store_true",
                        help="Compare scaffold vs random split performance")
    parser.add_argument("--ad-analysis", dest="ad_analysis", action="store_true",
                        help="Run applicability domain analysis")
    parser.add_argument("--scaffold-analysis", dest="scaffold_analysis", action="store_true",
                        help="Per-scaffold performance breakdown")
    parser.add_argument("--smiles", type=str, default=None,
                        help="Single SMILES for interpretability analysis")
    parser.add_argument("--interpret", action="store_true",
                        help="Run interpretability analysis on --smiles")

    return parser.parse_args()


def run_comparison_experiment(
    model,
    task_specs,
    task_type,
    dataset_name,
    data_root,
    batch_size,
    device,
    featurizer,
):
    """Systematically compare scaffold vs random split performance."""
    from src.evaluation.moleculenet_metrics import get_metrics_fn
    from src.training.trainer import Trainer

    metrics_fn = get_metrics_fn(task_type)
    val_metric = "mean_auroc" if task_type == "classification" else "mean_rmse"

    results = {}
    for split_type in ("scaffold", "random"):
        loaders = get_dataloaders(
            dataset_name=dataset_name,
            root=data_root,
            split=split_type,
            batch_size=batch_size,
            featurizer=featurizer,
        )

        trainer = Trainer(model=model, task_specs=task_specs, device=device)
        test_metrics = trainer.evaluate(loaders["test"], metrics_fn)
        results[split_type] = test_metrics.get(val_metric, test_metrics.get("loss"))

        print_metrics_table(test_metrics, f"{split_type.upper()} Split Test Results")

    # Print comparison
    if "scaffold" in results and "random" in results:
        scaffold_score = results["scaffold"]
        random_score = results["random"]
        print("\n" + "="*60)
        print("SCAFFOLD vs RANDOM SPLIT COMPARISON")
        print("="*60)
        print(f"  Scaffold split {val_metric}: {scaffold_score:.4f}")
        print(f"  Random split   {val_metric}: {random_score:.4f}")
        if task_type == "classification":
            delta = random_score - scaffold_score
            print(f"  Inflation (random - scaffold): +{delta:.4f}")
            print(f"\n  Random split OVERESTIMATES performance by {delta:.4f} AUROC units")
            print(f"  This is data leakage — scaffold-similar molecules in train AND test")
        else:
            delta = scaffold_score - random_score  # higher RMSE = worse
            print(f"  Degradation (scaffold - random): {delta:.4f}")
            print(f"\n  Random split UNDERESTIMATES error by {delta:.4f} RMSE units")
        print("="*60)

    return results


def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    args = parse_args()

    # Device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint, device=args.device)

    # Get task specs from dataset
    config = MOLECULENET_DATASETS[args.dataset]
    task_type = config["task_type"]
    label_cols = config["label_cols"]
    if label_cols is None:
        label_cols = [f"sider_{i}" for i in range(27)]
    task_specs = [(name, task_type, 1) for name in label_cols]

    # Build model
    # Note: if checkpoint has config, use that; otherwise rebuild from dataset
    if "model_state_dict" in checkpoint:
        # Try to infer model type from checkpoint filename
        ckpt_name = Path(args.checkpoint).stem
        if "mpnn" in ckpt_name:
            from src.models.mpnn import MPNNModel
            model = MPNNModel(task_specs=task_specs)
        elif "gat" in ckpt_name:
            from src.models.gat_model import GATModel
            model = GATModel(task_specs=task_specs)
        else:
            from src.models.gin_model import GINModel
            model = GINModel(task_specs=task_specs)

        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raise ValueError("Checkpoint does not contain model_state_dict")

    model = model.to(args.device)
    model.eval()

    # Featurizer
    featurizer = MolecularFeaturizer()

    # -------------------------------------------------------------------
    # Standard test evaluation
    # -------------------------------------------------------------------
    from src.evaluation.moleculenet_metrics import get_metrics_fn
    from src.training.trainer import Trainer

    loaders = get_dataloaders(
        dataset_name=args.dataset,
        root=args.data_root,
        split=args.split,
        batch_size=args.batch_size,
        featurizer=featurizer,
    )

    metrics_fn = get_metrics_fn(task_type)
    trainer = Trainer(model=model, task_specs=task_specs, device=args.device)
    test_metrics = trainer.evaluate(loaders["test"], metrics_fn)

    print_metrics_table(
        test_metrics,
        f"{args.dataset.upper()} Test Results ({args.split} split)"
    )

    # Save metrics
    results_path = output_dir / f"{args.dataset}_test_metrics.json"
    with open(results_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    logger.info(f"Saved test metrics to {results_path}")

    # -------------------------------------------------------------------
    # Split comparison experiment
    # -------------------------------------------------------------------
    if args.compare_splits:
        logger.info("Running scaffold vs random split comparison...")
        run_comparison_experiment(
            model, task_specs, task_type, args.dataset,
            args.data_root, args.batch_size, args.device, featurizer
        )

    # -------------------------------------------------------------------
    # Applicability domain analysis
    # -------------------------------------------------------------------
    if args.ad_analysis:
        logger.info("Running applicability domain analysis...")
        from src.evaluation.moleculenet_metrics import ApplicabilityDomainAnalyzer

        # Get training SMILES
        train_loader = loaders["train"]
        train_smiles = [batch.smiles for batch in train_loader
                        for smiles in batch.smiles]

        test_loader = loaders["test"]
        test_smiles_list = []
        for batch in test_loader:
            if hasattr(batch, "smiles"):
                test_smiles_list.extend(batch.smiles)

        if train_smiles and test_smiles_list:
            ad = ApplicabilityDomainAnalyzer(similarity_threshold=0.3)
            ad.fit(train_smiles)
            in_domain_mask = ad.in_domain(test_smiles_list)

            print(f"\nApplicability Domain Analysis:")
            print(f"  Test molecules in domain (Tanimoto ≥ 0.3): "
                  f"{in_domain_mask.sum()} / {len(in_domain_mask)} "
                  f"({in_domain_mask.mean()*100:.1f}%)")
            print(f"  Consider flagging out-of-domain predictions as unreliable.")

    # -------------------------------------------------------------------
    # Interpretability for single SMILES
    # -------------------------------------------------------------------
    if args.smiles and args.interpret:
        logger.info(f"Running interpretability analysis for: {args.smiles}")

        from src.evaluation.interpretability import (
            IntegratedGradients,
            SubstructureImportance,
        )

        # Integrated Gradients
        ig = IntegratedGradients(model, task_name=task_specs[0][0])
        try:
            attributions = ig.attribute(args.smiles, n_steps=50)
            print(f"\nIntegrated Gradients — Atom Importance:")
            print(f"  Top 5 important atoms (by IG norm):")
            imp = attributions["node_importance"]
            for rank, idx in enumerate(np.argsort(imp)[::-1][:5]):
                print(f"    Atom {idx}: importance = {imp[idx]:.4f}")
        except Exception as e:
            logger.warning(f"IG analysis failed: {e}")

        # Substructure importance
        subst_analyzer = SubstructureImportance(model, task_name=task_specs[0][0])
        try:
            top_substructures = subst_analyzer.rank_substructures(args.smiles, top_n=5)
            print(f"\nSubstructure Importance (top 5):")
            for name, score in top_substructures:
                direction = "increases" if score > 0 else "decreases"
                print(f"  {name}: removing {direction} prediction by {abs(score):.4f}")
        except Exception as e:
            logger.warning(f"Substructure analysis failed: {e}")


if __name__ == "__main__":
    main()
