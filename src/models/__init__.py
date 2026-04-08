"""GNN model architectures for molecular property prediction."""
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn

from src.models.mpnn import MPNNModel, build_mpnn_for_dataset
from src.models.gat_model import GATModel, build_gat_for_dataset
from src.models.gin_model import GINModel, build_gin_for_dataset
from src.models.fingerprint_baseline import (
    RandomForestBaseline,
    XGBoostBaseline,
    MolecularFingerprinter,
    run_baseline_comparison,
)


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
    strict: bool = True,
) -> dict:
    """Load a model checkpoint saved by the GNNTrainer.

    Args:
        model: Model instance to load weights into.
        checkpoint_path: Path to the .pt checkpoint file.
        device: Device to map tensors to.
        strict: Whether to require exact key matching.

    Returns:
        The full checkpoint dict (includes epoch, metrics, config).
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=strict)
    model.eval()
    return ckpt


__all__ = [
    "MPNNModel",
    "GATModel",
    "GINModel",
    "build_mpnn_for_dataset",
    "build_gat_for_dataset",
    "build_gin_for_dataset",
    "RandomForestBaseline",
    "XGBoostBaseline",
    "MolecularFingerprinter",
    "run_baseline_comparison",
    "load_checkpoint",
]
