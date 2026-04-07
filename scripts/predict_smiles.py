#!/usr/bin/env python3
"""
Predict molecular properties from SMILES strings using trained MolProp-GNN models.

Usage:
    # Single molecule prediction
    python scripts/predict_smiles.py \\
        --smiles "CC(=O)Oc1ccccc1C(=O)O" \\
        --checkpoint checkpoints/esol_gin_scaffold_best.pt \\
        --property esol

    # Predict from CSV file
    python scripts/predict_smiles.py \\
        --input compounds.csv --smiles-col SMILES \\
        --checkpoint checkpoints/tox21_gat_scaffold_best.pt \\
        --property tox21 \\
        --output predictions.csv

    # Predict all loaded ADMET endpoints at once
    python scripts/predict_smiles.py \\
        --smiles "CN1CCC[C@H]1c2cccnc2" \\
        --checkpoint-dir checkpoints/ \\
        --all-endpoints \\
        --output-format json

    # Batch prediction with applicability domain check
    python scripts/predict_smiles.py \\
        --input candidates.csv \\
        --checkpoint checkpoints/bbbp_gin_scaffold_best.pt \\
        --property bbbp \\
        --ad-check \\
        --output bbbp_predictions.csv
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.featurizer import MolecularFeaturizer
from src.data.molecule_dataset import smiles_to_graph, MOLECULENET_DATASETS


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_model(checkpoint_path: str, dataset_name: str, device: str = "cpu"):
    """Load a trained model from checkpoint."""
    config = MOLECULENET_DATASETS[dataset_name]
    task_type = config["task_type"]
    label_cols = config["label_cols"] or [f"sider_{i}" for i in range(27)]
    task_specs = [(name, task_type, 1) for name in label_cols]

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Infer model type from filename
    ckpt_stem = Path(checkpoint_path).stem
    if "mpnn" in ckpt_stem:
        from src.models.mpnn import MPNNModel
        model = MPNNModel(task_specs=task_specs)
    elif "gat" in ckpt_stem:
        from src.models.gat_model import GATModel
        model = GATModel(task_specs=task_specs)
    else:
        from src.models.gin_model import GINModel
        model = GINModel(task_specs=task_specs)

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, task_specs, task_type


@torch.no_grad()
def predict_single(
    smiles: str,
    model,
    task_specs: list,
    task_type: str,
    featurizer: MolecularFeaturizer,
    device: str = "cpu",
) -> dict:
    """Predict properties for a single SMILES string.

    Returns:
        Dict of task_name → prediction value (probability for classification,
        raw value for regression)
    """
    graph = smiles_to_graph(smiles, featurizer)
    if graph is None:
        return {"error": f"Invalid SMILES: {smiles}"}

    graph = graph.to(device)
    # Add batch dimension (single graph)
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)

    outputs = model(graph)

    predictions = {}
    for task_name, tt, _ in task_specs:
        pred = outputs[task_name].squeeze()
        if tt == "classification":
            # Convert logit to probability
            prob = torch.sigmoid(pred).item()
            predictions[task_name] = {
                "probability": round(prob, 4),
                "predicted_class": int(prob >= 0.5),
                "interpretation": "active" if prob >= 0.5 else "inactive",
            }
        else:
            predictions[task_name] = {
                "value": round(pred.item(), 4),
            }

    return predictions


@torch.no_grad()
def predict_batch(
    smiles_list: list[str],
    model,
    task_specs: list,
    task_type: str,
    featurizer: MolecularFeaturizer,
    device: str = "cpu",
    batch_size: int = 64,
) -> list[dict]:
    """Predict properties for a list of SMILES strings."""
    from torch_geometric.data import Batch as PyGBatch

    all_predictions = []

    for i in range(0, len(smiles_list), batch_size):
        batch_smiles = smiles_list[i:i + batch_size]
        batch_graphs = []
        batch_valid = []

        for smi in batch_smiles:
            graph = smiles_to_graph(smi, featurizer)
            if graph is not None:
                batch_graphs.append(graph)
                batch_valid.append(True)
            else:
                batch_valid.append(False)

        if not batch_graphs:
            all_predictions.extend([{"error": "invalid SMILES"}] * len(batch_smiles))
            continue

        # Batch the valid graphs
        pyg_batch = PyGBatch.from_data_list(batch_graphs).to(device)
        outputs = model(pyg_batch)

        # Unpack per-molecule predictions
        valid_idx = 0
        for j, is_valid in enumerate(batch_valid):
            if not is_valid:
                all_predictions.append({"error": f"Invalid SMILES: {batch_smiles[j]}"})
                continue

            pred_dict = {}
            for task_name, tt, _ in task_specs:
                pred = outputs[task_name][valid_idx].squeeze()
                if tt == "classification":
                    prob = torch.sigmoid(pred).item()
                    pred_dict[task_name] = {
                        "probability": round(prob, 4),
                        "predicted_class": int(prob >= 0.5),
                    }
                else:
                    pred_dict[task_name] = {"value": round(pred.item(), 4)}
            all_predictions.append(pred_dict)
            valid_idx += 1

    return all_predictions


def format_prediction_output(
    smiles: str,
    predictions: dict,
    task_type: str,
    include_interpretation: bool = True,
) -> str:
    """Format predictions for human-readable output."""
    lines = [f"\nSMILES: {smiles}"]
    lines.append("-" * 50)

    if "error" in predictions:
        lines.append(f"  ERROR: {predictions['error']}")
        return "\n".join(lines)

    for task_name, pred in predictions.items():
        if task_type == "classification":
            prob = pred["probability"]
            cls = pred["predicted_class"]
            label = "ACTIVE  ✓" if cls == 1 else "INACTIVE ✗"
            lines.append(f"  {task_name}:")
            lines.append(f"    Probability: {prob:.4f}  →  {label}")
        else:
            val = pred["value"]
            lines.append(f"  {task_name}: {val:.4f}")

    return "\n".join(lines)


def check_lipinski(smiles: str) -> dict:
    """Check Lipinski's Rule of Five for a SMILES string.

    Rule of Five (Ro5) is a quick filter for oral bioavailability:
    - MW ≤ 500 Da
    - logP ≤ 5
    - H-bond donors ≤ 5
    - H-bond acceptors ≤ 10

    Returns pass/fail + individual values.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": "invalid SMILES"}

        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)

        ro5_violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])

        return {
            "MW": round(mw, 2),
            "logP": round(logp, 2),
            "HBD": hbd,
            "HBA": hba,
            "TPSA": round(tpsa, 2),
            "RotBonds": rot_bonds,
            "Ro5_violations": ro5_violations,
            "Ro5_pass": ro5_violations <= 1,
            "Veber_pass": rot_bonds <= 10 and tpsa <= 140,
        }
    except ImportError:
        return {"error": "RDKit not available"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict molecular properties from SMILES",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--smiles", type=str,
                             help="Single SMILES string to predict")
    input_group.add_argument("--input", type=str,
                             help="CSV file with SMILES column")

    parser.add_argument("--smiles-col", dest="smiles_col", type=str, default="smiles",
                        help="Column name for SMILES in input CSV")

    # Model
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--property", type=str, default=None,
                        choices=list(MOLECULENET_DATASETS.keys()),
                        help="Property/dataset the model was trained on")

    # Output
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV file for batch predictions")
    parser.add_argument("--output-format", dest="output_format", type=str,
                        default="table", choices=["table", "json", "csv"])

    # Analysis options
    parser.add_argument("--ad-check", dest="ad_check", action="store_true",
                        help="Flag out-of-applicability-domain predictions")
    parser.add_argument("--lipinski", action="store_true",
                        help="Print Lipinski/Veber rule checks alongside predictions")
    parser.add_argument("--device", type=str, default=None)

    return parser.parse_args()


def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    args = parse_args()

    # Device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.checkpoint is None or args.property is None:
        logger.error("Must specify both --checkpoint and --property")
        sys.exit(1)

    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model, task_specs, task_type = load_model(
        args.checkpoint, args.property, device=args.device
    )
    featurizer = MolecularFeaturizer()
    logger.info(f"Model loaded. Predicting {len(task_specs)} tasks ({task_type})")

    # -------------------------------------------------------------------
    # Single SMILES prediction
    # -------------------------------------------------------------------
    if args.smiles:
        smiles = args.smiles
        logger.info(f"Predicting properties for: {smiles}")

        predictions = predict_single(
            smiles, model, task_specs, task_type, featurizer, device=args.device
        )

        if args.output_format == "json":
            print(json.dumps({"smiles": smiles, "predictions": predictions}, indent=2))
        else:
            print(format_prediction_output(smiles, predictions, task_type))

        if args.lipinski:
            lipinski_result = check_lipinski(smiles)
            print("\nLipinski / Drug-likeness:")
            print(f"  MW:            {lipinski_result.get('MW', 'N/A')} Da")
            print(f"  logP:          {lipinski_result.get('logP', 'N/A')}")
            print(f"  H-bond donors: {lipinski_result.get('HBD', 'N/A')}")
            print(f"  H-bond accept: {lipinski_result.get('HBA', 'N/A')}")
            print(f"  TPSA:          {lipinski_result.get('TPSA', 'N/A')} Å²")
            print(f"  Ro5 pass:      {lipinski_result.get('Ro5_pass', 'N/A')}")
            print(f"  Veber pass:    {lipinski_result.get('Veber_pass', 'N/A')}")

    # -------------------------------------------------------------------
    # Batch prediction from CSV
    # -------------------------------------------------------------------
    elif args.input:
        import pandas as pd

        logger.info(f"Loading input from {args.input}")
        df = pd.read_csv(args.input)

        if args.smiles_col not in df.columns:
            logger.error(f"Column '{args.smiles_col}' not found. "
                         f"Available columns: {list(df.columns)}")
            sys.exit(1)

        smiles_list = df[args.smiles_col].tolist()
        logger.info(f"Predicting {len(smiles_list)} molecules...")

        predictions = predict_batch(
            smiles_list, model, task_specs, task_type,
            featurizer, device=args.device,
        )

        # Build output DataFrame
        results_df = df.copy()
        for task_name, tt, _ in task_specs:
            col_vals = []
            for pred in predictions:
                if "error" in pred or task_name not in pred:
                    col_vals.append(np.nan)
                elif tt == "classification":
                    col_vals.append(pred[task_name]["probability"])
                else:
                    col_vals.append(pred[task_name]["value"])
            results_df[f"pred_{task_name}"] = col_vals

        if args.lipinski:
            lipinski_results = [check_lipinski(smi) for smi in smiles_list]
            for key in ["MW", "logP", "HBD", "HBA", "TPSA", "Ro5_violations", "Ro5_pass"]:
                results_df[key] = [r.get(key, np.nan) for r in lipinski_results]

        if args.output:
            results_df.to_csv(args.output, index=False)
            logger.info(f"Saved {len(results_df)} predictions to {args.output}")
        else:
            print(results_df.to_string(index=False))

    logger.info("Done.")


if __name__ == "__main__":
    main()
