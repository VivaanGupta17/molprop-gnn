"""
Graph Isomorphism Network (GIN) for molecular property prediction.

Based on Xu et al., "How Powerful are Graph Neural Networks?" (ICLR 2019).
GIN is theoretically as powerful as the Weisfeiler-Lehman graph isomorphism test,
making it the most expressive standard GNN for distinguishing graph structures.

Key design choices for molecules:
- epsilon is learned (not fixed) — allows the model to balance self vs. neighbor info
- MLP in aggregation is 2-layer (required for GIN expressiveness theorem)
- Virtual node augmentation: adds a global node connected to all atoms
  to enable long-range information flow (important for ADMET where whole-molecule
  features like MW, logP matter alongside local structure)
- Batch normalization (not layer norm) after each GIN layer, following Hu et al.
  "Strategies for Pre-training GNNs" (ICLR 2020)

Note: Pure GIN ignores edge features. We use GINEConv (the edge-feature extension)
from Hu et al. 2020, which passes edge features through a linear layer before
adding to neighbor messages.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool
from torch_geometric.nn.aggr import AttentionalAggregation

from src.models.mpnn import AtomEncoder, BondEncoder, MultiTaskHead


# ---------------------------------------------------------------------------
# Virtual Node
# ---------------------------------------------------------------------------

class VirtualNodeMixin(nn.Module):
    """Augments a molecule graph with a virtual node connected to all real atoms.

    The virtual node aggregates global information and broadcasts it back to all
    atoms in subsequent MP rounds. This helps with long-range dependencies
    (e.g., ADMET properties that depend on whole-molecule physicochemistry).

    Reference: Gilmer et al. 2017; Hu et al. 2020
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        # Embedding for virtual node initialized to zero
        self.vn_embedding = nn.Embedding(1, hidden_dim)
        nn.init.constant_(self.vn_embedding.weight, 0)

        # MLP to update virtual node: aggregates all atom embeddings
        self.vn_update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def update_vn(
        self, h: Tensor, vn_emb: Tensor, batch: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Update virtual node and add its embedding to all atom nodes.

        Args:
            h: Atom features [num_atoms, hidden_dim]
            vn_emb: Virtual node features [num_graphs, hidden_dim]
            batch: Batch assignment [num_atoms]
        Returns:
            Updated (h, vn_emb)
        """
        # Aggregate atom features into virtual node
        vn_in = global_add_pool(h, batch) + vn_emb
        vn_emb_new = self.vn_update(vn_in)

        # Broadcast virtual node embedding to all atoms
        h_new = h + self.dropout(vn_emb_new[batch])
        return h_new, vn_emb_new


# ---------------------------------------------------------------------------
# GIN Layer
# ---------------------------------------------------------------------------

class GINELayer(nn.Module):
    """Single GINEConv layer with batch normalization.

    GINEConv: h_i' = MLP((1 + ε) * h_i + Σ_{j∈N(i)} ReLU(h_j + e_ij))

    Using 2-layer MLP as required for GIN's theoretical expressiveness guarantee.
    """

    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float = 0.0):
        super().__init__()
        # 2-layer MLP (required for expressiveness, Xu et al. Theorem 3)
        mlp = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.BatchNorm1d(2 * hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.conv = GINEConv(
            nn=mlp,
            eps=0.0,          # initial epsilon
            train_eps=True,   # learn epsilon
            edge_dim=edge_dim,
        )
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        h = self.conv(x, edge_index, edge_attr)
        h = self.bn(h)
        return self.dropout(F.silu(h))


# ---------------------------------------------------------------------------
# GIN Model
# ---------------------------------------------------------------------------

class GINModel(nn.Module):
    """Graph Isomorphism Network for molecular property prediction.

    Architecture:
    1. Atom/bond encoding (same as MPNN for fair comparison)
    2. L GINEConv layers with optional virtual node augmentation
    3. Pooling: sum over all layers (not just last layer — captures multi-scale info)
    4. Multi-task prediction head

    The sum-over-layers readout (also called "virtual node" trick in some papers)
    gives each layer its own contribution to the final representation.

    Args:
        hidden_dim: Node embedding dimension
        edge_dim: Edge embedding dimension
        num_layers: Number of GIN layers (typically 5 works well, Hu et al. 2020)
        use_virtual_node: Add virtual node for global info aggregation
        residual: Whether to use residual connections between GIN layers
        task_specs: List of (name, type, n_out) task descriptors
        dropout: Dropout probability
    """

    def __init__(
        self,
        hidden_dim: int = 300,
        edge_dim: int = 64,
        num_layers: int = 5,
        use_virtual_node: bool = True,
        residual: bool = True,
        task_specs: Optional[list[tuple[str, str, int]]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_virtual_node = use_virtual_node
        self.residual = residual

        if task_specs is None:
            task_specs = [("property", "regression", 1)]
        self.task_specs = task_specs

        # Encoders
        self.atom_encoder = AtomEncoder(hidden_dim)
        self.bond_encoder = BondEncoder(edge_dim)

        # GIN layers
        self.gin_layers = nn.ModuleList([
            GINELayer(hidden_dim, edge_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Virtual node (optional)
        if use_virtual_node:
            self.virtual_node = VirtualNodeMixin(hidden_dim, dropout=dropout)

        # Linear projections for each layer's pooled output (for sum readout)
        self.layer_pools = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # Final projection after summing layer readouts
        self.readout_norm = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # Prediction head
        self.head = MultiTaskHead(hidden_dim, task_specs, dropout=dropout)

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

        # Initialize virtual node embedding
        if self.use_virtual_node:
            num_graphs = batch.max().item() + 1
            vn_emb = self.virtual_node.vn_embedding(
                torch.zeros(num_graphs, dtype=torch.long, device=x.device)
            )

        # Accumulate readouts from each layer
        h_graph_sum = torch.zeros(
            batch.max().item() + 1, self.hidden_dim, device=x.device
        )

        for i, gin_layer in enumerate(self.gin_layers):
            # Virtual node broadcast (before each layer)
            if self.use_virtual_node and i > 0:
                h, vn_emb = self.virtual_node.update_vn(h, vn_emb, batch)

            h_new = gin_layer(h, edge_index, e)

            # Residual connection
            if self.residual and i > 0:
                h_new = h_new + h
            h = h_new

            # Pool and accumulate this layer's graph representation
            h_pool = global_add_pool(h, batch)
            h_graph_sum = h_graph_sum + self.layer_pools[i](h_pool)

        h_graph = self.readout_norm(h_graph_sum)
        return self.head(h_graph)

    def get_node_embeddings(self, data: Union[Data, Batch]) -> list[Tensor]:
        """Return node embeddings at each layer (for multi-scale analysis)."""
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )

        h = self.atom_encoder(x)
        e = self.bond_encoder(edge_attr)

        if self.use_virtual_node:
            num_graphs = batch.max().item() + 1
            vn_emb = self.virtual_node.vn_embedding(
                torch.zeros(num_graphs, dtype=torch.long, device=x.device)
            )

        layer_embeddings = []
        for i, gin_layer in enumerate(self.gin_layers):
            if self.use_virtual_node and i > 0:
                h, vn_emb = self.virtual_node.update_vn(h, vn_emb, batch)
            h_new = gin_layer(h, edge_index, e)
            if self.residual and i > 0:
                h_new = h_new + h
            h = h_new
            layer_embeddings.append(h.detach())

        return layer_embeddings

    @classmethod
    def load_from_checkpoint(cls, path: str, device: str = "cpu") -> "GINModel":
        checkpoint = torch.load(path, map_location=device)
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    def save_checkpoint(self, path: str, config: dict, extra: Optional[dict] = None):
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
# Factory
# ---------------------------------------------------------------------------

def build_gin_for_dataset(dataset_name: str, **kwargs) -> GINModel:
    """Construct GIN with appropriate task heads for a MoleculeNet dataset."""
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
    return GINModel(task_specs=DATASET_CONFIGS[dataset_name], **kwargs)
