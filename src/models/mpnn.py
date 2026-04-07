"""
Message Passing Neural Network (MPNN) for molecular property prediction.

Architecture follows Gilmer et al., "Neural Message Passing for Quantum Chemistry"
(ICML 2017), adapted for ADMET property prediction on MoleculeNet benchmarks.

The key design choices:
- Edge features are first-class citizens (not just node features)
- GRU-based node update for better gradient flow
- Multiple readout options (sum, mean, attention-weighted)
- Multi-task head for simultaneous prediction of correlated endpoints
"""

from __future__ import annotations

import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool
from torch_geometric.utils import softmax


# ---------------------------------------------------------------------------
# Atom and Bond Encoders
# ---------------------------------------------------------------------------

class AtomEncoder(nn.Module):
    """Encode atom feature vectors into a fixed-size embedding.

    Handles mixed one-hot (categorical) + continuous atom features.
    Categorical features use learned embeddings; continuous features use linear projection.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        # Categorical feature embedding sizes (must match featurizer.py output)
        # atom_type(44), degree(11), formal_charge(11), hybridization(8), num_Hs(5)
        self.atom_type_emb = nn.Embedding(44 + 1, 32)
        self.degree_emb = nn.Embedding(11 + 1, 16)
        self.formal_charge_emb = nn.Embedding(11 + 1, 8)
        self.hybridization_emb = nn.Embedding(8 + 1, 8)
        self.num_hs_emb = nn.Embedding(5 + 1, 8)

        # Continuous features: aromaticity (1), in_ring (1), mass (1) = 3 dims
        categorical_dim = 32 + 16 + 8 + 8 + 8  # = 72
        continuous_dim = 3
        self.proj = nn.Linear(categorical_dim + continuous_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Atom feature matrix [num_atoms, num_atom_features]
               Columns: atom_type_idx, degree_idx, formal_charge_idx,
                        hybridization_idx, num_hs_idx, aromaticity, in_ring, mass
        Returns:
            Atom embeddings [num_atoms, hidden_dim]
        """
        atom_type = self.atom_type_emb(x[:, 0].long())
        degree = self.degree_emb(x[:, 1].long())
        formal_charge = self.formal_charge_emb(x[:, 2].long())
        hybridization = self.hybridization_emb(x[:, 3].long())
        num_hs = self.num_hs_emb(x[:, 4].long())

        continuous = x[:, 5:8].float()  # aromaticity, in_ring, mass

        combined = torch.cat([atom_type, degree, formal_charge,
                               hybridization, num_hs, continuous], dim=-1)
        return self.norm(F.silu(self.proj(combined)))


class BondEncoder(nn.Module):
    """Encode bond (edge) feature vectors into a fixed-size embedding.

    Bond features: bond_type(4), conjugated(1), in_ring(1), stereo(6)
    """

    def __init__(self, edge_dim: int):
        super().__init__()
        self.bond_type_emb = nn.Embedding(4 + 1, 16)
        self.stereo_emb = nn.Embedding(6 + 1, 8)

        # Continuous: conjugated, in_ring
        categorical_dim = 16 + 8
        continuous_dim = 2
        self.proj = nn.Linear(categorical_dim + continuous_dim, edge_dim)
        self.norm = nn.LayerNorm(edge_dim)

    def forward(self, edge_attr: Tensor) -> Tensor:
        """
        Args:
            edge_attr: Bond feature matrix [num_edges, num_bond_features]
        Returns:
            Bond embeddings [num_edges, edge_dim]
        """
        bond_type = self.bond_type_emb(edge_attr[:, 0].long())
        stereo = self.stereo_emb(edge_attr[:, 3].long())
        continuous = edge_attr[:, 1:3].float()  # conjugated, in_ring

        combined = torch.cat([bond_type, stereo, continuous], dim=-1)
        return self.norm(F.silu(self.proj(combined)))


# ---------------------------------------------------------------------------
# Message Passing Layer
# ---------------------------------------------------------------------------

class MPNNLayer(MessagePassing):
    """Single MPNN message passing layer with edge features.

    Message function: m_ij = MLP([h_i || h_j || e_ij])
    Update function: h_i' = GRU(h_i, sum_j(m_ij))
    """

    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float = 0.0):
        super().__init__(aggr="add")
        self.hidden_dim = hidden_dim

        # Message MLP: combines source, dest, and edge features
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # GRU-based update for better memory of original node state
        self.update_gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        """
        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
        Returns:
            Updated node features [num_nodes, hidden_dim]
        """
        aggr = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        x_new = self.update_gru(aggr, x)
        return self.dropout(x_new)

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        # x_i: destination, x_j: source
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.message_mlp(msg_input)


# ---------------------------------------------------------------------------
# Attention-Weighted Readout
# ---------------------------------------------------------------------------

class AttentionReadout(nn.Module):
    """Computes graph-level representation via soft attention over nodes.

    h_G = Σ_i softmax(a_i) * transform(h_i)

    This allows the model to focus on the most task-relevant atoms
    (e.g., pharmacophore atoms, reactive groups) when forming the
    molecular fingerprint.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        # Gate network (scalar attention score per atom)
        self.gate_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Value transform
        self.value_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        """
        Args:
            x: Node features [num_nodes, hidden_dim]
            batch: Batch assignment vector [num_nodes]
        Returns:
            Graph-level features [num_graphs, hidden_dim]
        """
        gates = self.gate_nn(x)           # [num_nodes, 1]
        gates = softmax(gates, batch)     # normalize per graph
        values = self.value_nn(x)         # [num_nodes, hidden_dim]

        # Weighted sum per graph
        out = global_add_pool(gates * values, batch)
        return out


# ---------------------------------------------------------------------------
# Multi-Task Prediction Head
# ---------------------------------------------------------------------------

class MultiTaskHead(nn.Module):
    """Prediction head supporting simultaneous classification and regression tasks.

    Args:
        in_dim: Input dimension (graph embedding size)
        task_specs: List of (task_name, task_type, n_classes) tuples
                    task_type: "classification" or "regression"
                    n_classes: 1 for regression/binary, >1 for multiclass
    """

    def __init__(self, in_dim: int, task_specs: list[tuple[str, str, int]],
                 dropout: float = 0.2):
        super().__init__()
        self.task_specs = task_specs

        # Shared encoder before task-specific heads
        self.shared = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(in_dim // 2),
        )

        # One output head per task
        self.heads = nn.ModuleDict()
        for name, task_type, n_out in task_specs:
            safe_name = name.replace("-", "_")
            self.heads[safe_name] = nn.Sequential(
                nn.Linear(in_dim // 2, in_dim // 4),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(in_dim // 4, n_out),
            )

    def forward(self, h_graph: Tensor) -> dict[str, Tensor]:
        """
        Returns:
            Dict mapping task_name → raw logits/predictions
        """
        shared = self.shared(h_graph)
        outputs = {}
        for name, task_type, n_out in self.task_specs:
            safe_name = name.replace("-", "_")
            outputs[name] = self.heads[safe_name](shared)
        return outputs


# ---------------------------------------------------------------------------
# Full MPNN Model
# ---------------------------------------------------------------------------

class MPNNModel(nn.Module):
    """Message Passing Neural Network for molecular property prediction.

    Combines:
    1. Atom/bond encoding
    2. L message passing layers (with residual connections after layer 1)
    3. Graph-level readout (sum, mean, or attention)
    4. Multi-task prediction head

    Args:
        hidden_dim: Node feature dimension throughout the network
        edge_dim: Edge feature dimension
        num_layers: Number of message passing rounds
        readout: Pooling strategy ('sum', 'mean', 'attention')
        task_specs: List of (name, type, n_out) tuples
        dropout: Dropout probability
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        edge_dim: int = 64,
        num_layers: int = 4,
        readout: str = "attention",
        task_specs: Optional[list[tuple[str, str, int]]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.readout_type = readout

        # Default: single regression task
        if task_specs is None:
            task_specs = [("property", "regression", 1)]
        self.task_specs = task_specs

        # Encoders
        self.atom_encoder = AtomEncoder(hidden_dim)
        self.bond_encoder = BondEncoder(edge_dim)

        # Message passing stack
        self.mp_layers = nn.ModuleList([
            MPNNLayer(hidden_dim, edge_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Layer norms between MP layers
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Readout
        if readout == "attention":
            self.readout_fn = AttentionReadout(hidden_dim)
        elif readout in ("sum", "mean"):
            self.readout_fn = None  # handled in forward
        else:
            raise ValueError(f"Unknown readout: {readout}. Choose 'sum', 'mean', 'attention'.")

        # Prediction head
        self.head = MultiTaskHead(hidden_dim, task_specs, dropout=dropout)

    def forward(self, data: Union[Data, Batch]) -> dict[str, Tensor]:
        """
        Args:
            data: PyG Data or Batch object with x, edge_index, edge_attr, batch
        Returns:
            Dict of task_name → predictions [batch_size, n_out]
        """
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )

        # Encode atoms and bonds
        h = self.atom_encoder(x)              # [N, hidden_dim]
        e = self.bond_encoder(edge_attr)       # [E, edge_dim]

        # Message passing with residual connections (after first layer)
        for i, (mp_layer, norm) in enumerate(zip(self.mp_layers, self.layer_norms)):
            h_new = mp_layer(h, edge_index, e)
            if i > 0:
                h_new = h_new + h  # residual skip
            h = norm(h_new)

        # Graph-level readout
        if self.readout_type == "attention":
            h_graph = self.readout_fn(h, batch)
        elif self.readout_type == "sum":
            h_graph = global_add_pool(h, batch)
        else:
            h_graph = global_mean_pool(h, batch)

        return self.head(h_graph)

    def get_node_embeddings(self, data: Union[Data, Batch]) -> Tensor:
        """Return final node embeddings (useful for interpretability)."""
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        h = self.atom_encoder(x)
        e = self.bond_encoder(edge_attr)

        for i, (mp_layer, norm) in enumerate(zip(self.mp_layers, self.layer_norms)):
            h_new = mp_layer(h, edge_index, e)
            if i > 0:
                h_new = h_new + h
            h = norm(h_new)

        return h

    @classmethod
    def load_from_checkpoint(cls, path: str, device: str = "cpu") -> "MPNNModel":
        """Load a trained model from a checkpoint file."""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint["config"]
        model = cls(**config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    def save_checkpoint(self, path: str, config: dict, extra: Optional[dict] = None):
        """Save model checkpoint with config for reproducibility."""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": config,
        }
        if extra:
            checkpoint.update(extra)
        torch.save(checkpoint, path)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Factory function for common configurations
# ---------------------------------------------------------------------------

def build_mpnn_for_dataset(dataset_name: str, **kwargs) -> MPNNModel:
    """Construct MPNN with appropriate task heads for a MoleculeNet dataset.

    Args:
        dataset_name: One of 'bbbp', 'hiv', 'tox21', 'sider',
                      'esol', 'freesolv', 'lipophilicity'
    """
    DATASET_CONFIGS = {
        "bbbp": [("bbbp", "classification", 1)],
        "hiv": [("hiv_active", "classification", 1)],
        "esol": [("logS", "regression", 1)],
        "freesolv": [("dG_hyd", "regression", 1)],
        "lipophilicity": [("logD", "regression", 1)],
        "tox21": [(f"tox21_{i}", "classification", 1) for i in range(12)],
        "sider": [(f"sider_{i}", "classification", 1) for i in range(27)],
    }

    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                         f"Choose from {list(DATASET_CONFIGS.keys())}")

    task_specs = DATASET_CONFIGS[dataset_name]
    return MPNNModel(task_specs=task_specs, **kwargs)
