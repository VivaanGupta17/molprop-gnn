"""
Graph Attention Network (GAT) for molecular property prediction.

Implements multi-head attention on molecular graphs with edge feature integration.
Based on Veličković et al., "Graph Attention Networks" (ICLR 2018) and
Brody et al., "How Attentive are Graph Attention Networks?" (ICLR 2022) — GATv2.

Design choices for molecular data:
- GATv2 (dynamic attention) rather than original GAT (static attention).
  GATv2 uses the more expressive e(h_i, h_j) rather than e(h_i) + e(h_j).
- Edge features explicitly modify attention scores (crucial for bond type info).
- Multi-head concatenation for first L-1 layers, average for last layer.
- Jumping knowledge aggregation: concatenate all layer outputs before readout.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data, Batch
from torch_geometric.nn import (
    GATv2Conv,
    global_add_pool,
    global_mean_pool,
    JumpingKnowledge,
)
from torch_geometric.nn import AttentionalAggregation

from src.models.mpnn import AtomEncoder, BondEncoder, MultiTaskHead


class GATv2Layer(nn.Module):
    """Single GATv2 layer with edge feature support.

    Attention coefficients incorporate edge features:
        e_ij = LeakyReLU(a^T · W[h_i || h_j || e_ij])

    This is important for molecules because bond type (single/double/aromatic)
    should directly modulate how much a neighbor contributes to a node's update.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        edge_dim: int = 64,
        concat: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv = GATv2Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            edge_dim=edge_dim,
            concat=concat,
            dropout=dropout,
            add_self_loops=True,
            bias=True,
        )
        self.concat = concat
        self.heads = heads
        self.out_channels = out_channels

        # Output dim depends on concat mode
        layer_out_dim = out_channels * heads if concat else out_channels
        self.norm = nn.LayerNorm(layer_out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Returns:
            h: Updated node features
            alpha: Attention weights [num_edges, heads] for interpretability
        """
        h, alpha = self.conv(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        h = self.norm(F.silu(h))
        h = self.dropout(h)
        return h, alpha[1]  # alpha is (edge_index, attention_weights)


class EdgeFeatureGATLayer(nn.Module):
    """GAT layer that produces updated edge features alongside node features.

    Useful for tasks where edge states carry meaningful information
    (e.g., reaction site prediction, bond order change).
    """

    def __init__(self, hidden_dim: int, edge_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.node_layer = GATv2Layer(
            hidden_dim, hidden_dim // heads, heads=heads,
            edge_dim=edge_dim, concat=True, dropout=dropout
        )
        # Edge update: MLP on concatenated endpoint states + old edge
        self.edge_update = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, edge_dim),
            nn.SiLU(),
            nn.LayerNorm(edge_dim),
        )

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Returns updated (x, edge_attr, alpha)."""
        h_new, alpha = self.node_layer(x, edge_index, edge_attr)

        # Update edge features using endpoint node representations
        src, dst = edge_index
        edge_in = torch.cat([h_new[src], h_new[dst], edge_attr], dim=-1)
        edge_new = self.edge_update(edge_in)

        return h_new, edge_new, alpha


class GATModel(nn.Module):
    """Graph Attention Network for molecular property prediction.

    Architecture:
    1. Atom/bond encoding
    2. L GATv2 layers with edge features
    3. Jumping Knowledge aggregation (concatenate all layer outputs)
    4. Attentional global pooling
    5. Multi-task prediction head

    Args:
        hidden_dim: Per-head output dimension (total = hidden_dim * heads)
        num_layers: Number of GAT layers
        heads: Number of attention heads
        edge_dim: Edge feature embedding dimension
        jk_mode: Jumping Knowledge aggregation ('cat', 'max', 'lstm')
        task_specs: List of (name, type, n_out) tuples
        dropout: Dropout probability
        update_edges: Whether to propagate updated edge features between layers
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 4,
        heads: int = 4,
        edge_dim: int = 64,
        jk_mode: str = "cat",
        task_specs: Optional[list[tuple[str, str, int]]] = None,
        dropout: float = 0.1,
        update_edges: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.heads = heads
        self.hidden_dim = hidden_dim
        self.jk_mode = jk_mode
        self.update_edges = update_edges

        if task_specs is None:
            task_specs = [("property", "regression", 1)]
        self.task_specs = task_specs

        # Encoders
        self.atom_encoder = AtomEncoder(hidden_dim * heads)
        self.bond_encoder = BondEncoder(edge_dim)

        node_dim = hidden_dim * heads  # after concat attention

        # GAT layers
        if update_edges:
            self.gat_layers = nn.ModuleList([
                EdgeFeatureGATLayer(node_dim, edge_dim, heads=heads, dropout=dropout)
                for _ in range(num_layers)
            ])
        else:
            self.gat_layers = nn.ModuleList([
                GATv2Layer(node_dim, hidden_dim, heads=heads, edge_dim=edge_dim,
                           concat=True, dropout=dropout)
                for _ in range(num_layers)
            ])

        # Jumping Knowledge for multi-scale representation
        # JK-cat concatenates all layer outputs → richer features
        if jk_mode == "cat":
            jk_out_dim = node_dim * num_layers
        else:
            jk_out_dim = node_dim
        self.jk = JumpingKnowledge(mode=jk_mode, channels=node_dim, num_layers=num_layers)

        # Project JK output back to hidden_dim for readout
        self.jk_proj = nn.Sequential(
            nn.Linear(jk_out_dim, node_dim),
            nn.SiLU(),
            nn.LayerNorm(node_dim),
        )

        # Attentional global pooling (learn which atoms matter for the task)
        gate_nn = nn.Sequential(
            nn.Linear(node_dim, node_dim // 2),
            nn.SiLU(),
            nn.Linear(node_dim // 2, 1),
        )
        self.pool = AttentionalAggregation(gate_nn)

        # Prediction head
        self.head = MultiTaskHead(node_dim, task_specs, dropout=dropout)

    def forward(self, data: Union[Data, Batch]) -> dict[str, Tensor]:
        """
        Args:
            data: PyG Data/Batch with x, edge_index, edge_attr, batch
        Returns:
            Dict of task predictions
        """
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )

        h = self.atom_encoder(x)
        e = self.bond_encoder(edge_attr)

        layer_outputs = []
        for layer in self.gat_layers:
            if self.update_edges:
                h, e, _ = layer(h, edge_index, e)
            else:
                h, _ = layer(h, edge_index, e)
            layer_outputs.append(h)

        # Jumping knowledge aggregation
        h_jk = self.jk(layer_outputs)
        h = self.jk_proj(h_jk)

        # Graph readout via attentional pooling
        h_graph = self.pool(h, batch)

        return self.head(h_graph)

    def get_attention_weights(
        self, data: Union[Data, Batch]
    ) -> list[Tensor]:
        """Return per-layer attention weights for interpretability.

        Returns:
            List of attention tensors, one per layer.
            Each tensor: [num_edges, num_heads]
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        h = self.atom_encoder(x)
        e = self.bond_encoder(edge_attr)

        all_alphas = []
        for layer in self.gat_layers:
            if self.update_edges:
                h, e, alpha = layer(h, edge_index, e)
            else:
                h, alpha = layer(h, edge_index, e)
            all_alphas.append(alpha)

        return all_alphas

    @classmethod
    def load_from_checkpoint(cls, path: str, device: str = "cpu") -> "GATModel":
        checkpoint = torch.load(path, map_location=device)
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_gat_for_dataset(dataset_name: str, **kwargs) -> GATModel:
    """Construct GAT with appropriate task heads for a MoleculeNet dataset."""
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
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return GATModel(task_specs=DATASET_CONFIGS[dataset_name], **kwargs)
