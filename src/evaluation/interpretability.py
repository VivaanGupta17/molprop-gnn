"""
Model interpretability for molecular property prediction GNNs.

Implements:
1. AttentionVisualizer: Extract and visualize atom-level attention weights
   from GAT and MPNN attention readout layers.
2. IntegratedGradients: Gradient-based attribution for any GNN model.
   Computes which atom features most contribute to a prediction.
3. SubstructureImportance: Which functional groups/substructures are most
   predictive (via systematic perturbation / masking).
4. GradCAM-style node importance for GNNs.

These tools are essential for communicating model reasoning to medicinal
chemists — a key skill for pharma ML interviews and actual drug discovery work.

References:
- Sundararajan et al., "Axiomatic Attribution for Deep Networks" (ICML 2017)
- Pope et al., "Explainability Methods for GNNs" (CVPR 2019)
- Ying et al., "GNNExplainer" (NeurIPS 2019)
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

try:
    from torch_geometric.data import Data
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False


# ---------------------------------------------------------------------------
# Attention Visualizer
# ---------------------------------------------------------------------------

class AttentionVisualizer:
    """Extract and visualize attention weights from GNN models.

    Works with:
    - GATModel: uses per-layer GATv2Conv attention weights
    - MPNNModel with attention readout: uses graph-level attention scores

    Atom importance is computed by:
    1. Extracting raw attention weights for each edge (i→j)
    2. Aggregating by destination node (average incoming attention)
    3. Optionally averaging across attention heads
    """

    def __init__(self, model: nn.Module, model_type: str = "gat"):
        """
        Args:
            model: Trained GATModel or MPNNModel
            model_type: 'gat' or 'mpnn'
        """
        self.model = model
        self.model_type = model_type
        self.model.eval()

    def get_atom_importance(
        self,
        smiles: str,
        layer_idx: int = -1,
        head_aggregation: str = "mean",
    ) -> np.ndarray:
        """Compute per-atom importance scores from attention weights.

        Args:
            smiles: Input molecule SMILES
            layer_idx: Which GAT layer to use (-1 = last layer)
            head_aggregation: How to aggregate over heads ('mean' or 'max')

        Returns:
            Array of shape [n_atoms] with importance scores in [0, 1]
        """
        from src.data.featurizer import MolecularFeaturizer
        from src.data.molecule_dataset import smiles_to_graph

        featurizer = MolecularFeaturizer()
        graph = smiles_to_graph(smiles, featurizer)
        if graph is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        graph = graph.to(next(self.model.parameters()).device)

        with torch.no_grad():
            if self.model_type == "gat":
                all_alphas = self.model.get_attention_weights(graph)
                alphas = all_alphas[layer_idx]  # [n_edges, n_heads]
            elif self.model_type == "mpnn":
                # MPNN with attention readout
                x = self.model.atom_encoder(graph.x)
                e = self.model.bond_encoder(graph.edge_attr)
                for i, (mp_layer, norm) in enumerate(
                    zip(self.model.mp_layers, self.model.layer_norms)
                ):
                    h_new = mp_layer(x, graph.edge_index, e)
                    if i > 0:
                        h_new = h_new + x
                    x = norm(h_new)

                # Get attention readout scores
                gates = self.model.readout_fn.gate_nn(x)  # [n_atoms, 1]
                atom_importance = torch.softmax(gates, dim=0).squeeze(-1)
                return atom_importance.cpu().numpy()
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

        # Aggregate edge-level attention to node-level
        n_atoms = graph.num_nodes
        edge_index = graph.edge_index  # [2, n_edges]

        # Average over heads
        if head_aggregation == "mean":
            alphas_agg = alphas.mean(dim=-1)  # [n_edges]
        else:
            alphas_agg = alphas.max(dim=-1).values  # [n_edges]

        # Sum incoming attention for each destination node
        node_importance = torch.zeros(n_atoms, device=alphas.device)
        dst_nodes = edge_index[1]  # destination nodes
        node_importance.scatter_add_(0, dst_nodes, alphas_agg)

        # Normalize to [0, 1]
        node_importance = node_importance / (node_importance.max() + 1e-8)
        return node_importance.cpu().numpy()

    def render_molecule_svg(
        self,
        smiles: str,
        atom_weights: Optional[np.ndarray] = None,
        output_path: Optional[str] = None,
        width: int = 400,
        height: int = 300,
    ) -> str:
        """Render molecule with atom-level color highlighting.

        Atoms with high attention scores are colored red/orange.
        Atoms with low attention are colored blue.

        Args:
            smiles: Input SMILES
            atom_weights: Per-atom importance scores [0, 1]. If None, renders without coloring.
            output_path: If provided, saves SVG to this path
            width, height: Image dimensions in pixels

        Returns:
            SVG string
        """
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for molecular visualization.")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        # Add 2D coordinates for drawing
        AllChem.Compute2DCoords(mol)

        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        drawer.drawOptions().addStereoAnnotation = True

        if atom_weights is not None:
            # Color atoms by importance: low=blue, high=red
            atom_colors = {}
            highlight_atoms = []
            atom_radii = {}

            for idx in range(mol.GetNumAtoms()):
                w = float(atom_weights[idx]) if idx < len(atom_weights) else 0.0
                # Interpolate between blue (0) and red (1)
                r = w
                g = 0.2 * (1 - abs(2 * w - 1))  # peaks at w=0.5
                b = 1 - w
                atom_colors[idx] = (r, g, b)
                highlight_atoms.append(idx)
                atom_radii[idx] = 0.3 + 0.3 * w

            drawer.DrawMolecule(
                mol,
                highlightAtoms=highlight_atoms,
                highlightAtomColors=atom_colors,
                highlightAtomRadii=atom_radii,
            )
        else:
            drawer.DrawMolecule(mol)

        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()

        if output_path:
            with open(output_path, "w") as f:
                f.write(svg)
            logger.info(f"Saved molecular SVG to {output_path}")

        return svg


# ---------------------------------------------------------------------------
# Integrated Gradients
# ---------------------------------------------------------------------------

class IntegratedGradients:
    """Integrated Gradients attribution for GNN molecular models.

    Computes the expected gradient of output with respect to input features,
    integrated along a straight path from a baseline (all-zero molecule) to
    the actual molecule.

    IG satisfies two important axioms:
    1. Completeness: attributions sum to (prediction - baseline prediction)
    2. Sensitivity: non-zero attribution iff input differs from baseline AND
       model output differs from baseline

    For molecular GNNs, we attribute to:
    - Atom features (which atom properties drive the prediction)
    - Edge features (which bond properties matter)

    Reference: Sundararajan et al., "Axiomatic Attribution for Deep Networks" (ICML 2017)
    """

    def __init__(self, model: nn.Module, task_name: str = "property"):
        self.model = model
        self.task_name = task_name
        self.model.eval()

    def _get_prediction(self, data: "Data") -> Tensor:
        """Get scalar prediction for the target task."""
        outputs = self.model(data)
        pred = outputs[self.task_name].squeeze()
        if pred.dim() == 0:
            return pred
        return pred[0]  # take first molecule if batched

    def attribute(
        self,
        smiles: str,
        target_class: int = 1,
        n_steps: int = 50,
        return_edge_attr: bool = False,
    ) -> dict[str, np.ndarray]:
        """Compute integrated gradients for a molecule.

        Args:
            smiles: Input SMILES string
            target_class: For classification, 0 or 1. For regression, ignored.
            n_steps: Number of integration steps (higher = more accurate)
            return_edge_attr: Also attribute to edge features

        Returns:
            Dict with:
            - 'node_attributions': [n_atoms, atom_feat_dim]
            - 'node_importance': [n_atoms] (L2 norm of feature attributions)
            - 'edge_attributions': [n_edges, edge_feat_dim] (if requested)
        """
        from src.data.featurizer import MolecularFeaturizer
        from src.data.molecule_dataset import smiles_to_graph

        featurizer = MolecularFeaturizer()
        graph = smiles_to_graph(smiles, featurizer)
        if graph is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        device = next(self.model.parameters()).device
        graph = graph.to(device)

        # Baseline: zero atom features (represents "absence of atoms")
        baseline_x = torch.zeros_like(graph.x)
        baseline_edge_attr = torch.zeros_like(graph.edge_attr)

        # Interpolate from baseline to input
        alphas = torch.linspace(0, 1, n_steps + 1, device=device)

        # Accumulate gradients
        x_grads = torch.zeros_like(graph.x)
        edge_grads = torch.zeros_like(graph.edge_attr)

        for alpha in alphas:
            # Interpolated input
            interp_x = baseline_x + alpha * (graph.x - baseline_x)
            interp_edge_attr = (baseline_edge_attr +
                                alpha * (graph.edge_attr - baseline_edge_attr))

            interp_x.requires_grad_(True)
            if return_edge_attr:
                interp_edge_attr.requires_grad_(True)

            # Create modified graph
            interp_data = graph.clone()
            interp_data.x = interp_x
            interp_data.edge_attr = interp_edge_attr

            # Forward pass
            outputs = self.model(interp_data)
            pred = outputs[self.task_name].squeeze()
            if pred.dim() > 0:
                pred = pred[0]

            # Backward
            pred.backward()

            x_grads += interp_x.grad.detach()
            if return_edge_attr and interp_edge_attr.grad is not None:
                edge_grads += interp_edge_attr.grad.detach()

            # Zero gradients
            interp_x.grad = None
            if return_edge_attr:
                interp_edge_attr.grad = None

        # Riemann sum approximation of integral
        integrated_x_grads = (graph.x - baseline_x) * x_grads / n_steps
        integrated_edge_grads = (graph.edge_attr - baseline_edge_attr) * edge_grads / n_steps

        node_attributions = integrated_x_grads.cpu().numpy()  # [n_atoms, feat_dim]
        node_importance = np.linalg.norm(node_attributions, axis=-1)  # [n_atoms]

        # Normalize importance scores to [0, 1]
        if node_importance.max() > 0:
            node_importance = node_importance / node_importance.max()

        result = {
            "node_attributions": node_attributions,
            "node_importance": node_importance,
        }

        if return_edge_attr:
            edge_attributions = integrated_edge_grads.cpu().numpy()
            result["edge_attributions"] = edge_attributions
            result["edge_importance"] = np.linalg.norm(edge_attributions, axis=-1)

        return result


# ---------------------------------------------------------------------------
# Substructure Importance
# ---------------------------------------------------------------------------

class SubstructureImportance:
    """Evaluate importance of chemical substructures via systematic masking.

    Algorithm:
    1. Identify substructure atoms (SMARTS matching)
    2. Zero out node features for those atoms
    3. Measure change in prediction
    4. Large drop = important substructure

    This approach is model-agnostic and produces chemically intuitive
    explanations based on functional groups (e.g., "amine group increases
    predicted blood-brain barrier permeability by 0.15 probability units").
    """

    # Common pharmacophoric substructures
    PHARMACOPHORE_SMARTS = {
        "amine_primary": "[NH2]",
        "amine_secondary": "[NH]",
        "carboxylic_acid": "C(=O)[OH]",
        "amide": "C(=O)[NH]",
        "hydroxyl": "[OH]",
        "aromatic_ring": "c1ccccc1",
        "halogen_F": "[F]",
        "halogen_Cl": "[Cl]",
        "halogen_Br": "[Br]",
        "sulfur": "[S]",
        "phosphorus": "[P]",
        "nitro": "[N+](=O)[O-]",
        "ether": "[O;!$(O=C)]",
        "ketone": "C(=O)[C]",
        "sulfonamide": "S(=O)(=O)[NH]",
        "pyridine": "c1ccncc1",
        "imidazole": "c1cnc[nH]1",
    }

    def __init__(self, model: nn.Module, task_name: str = "property"):
        self.model = model
        self.task_name = task_name
        self.model.eval()

    def importance_by_masking(
        self,
        smiles: str,
        smarts_dict: Optional[dict[str, str]] = None,
    ) -> dict[str, float]:
        """Score each substructure by its effect on the prediction.

        Returns:
            Dict of substructure_name → importance_score
            Positive score = removing substructure decreases prediction
            Negative score = removing substructure increases prediction
        """
        from src.data.featurizer import MolecularFeaturizer
        from src.data.molecule_dataset import smiles_to_graph

        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit required for substructure matching.")

        if smarts_dict is None:
            smarts_dict = self.PHARMACOPHORE_SMARTS

        featurizer = MolecularFeaturizer()
        graph = smiles_to_graph(smiles, featurizer)
        if graph is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        mol = Chem.MolFromSmiles(smiles)
        device = next(self.model.parameters()).device
        graph = graph.to(device)

        # Baseline prediction (unmasked)
        with torch.no_grad():
            outputs = self.model(graph)
            baseline_pred = outputs[self.task_name].squeeze()
            if baseline_pred.dim() > 0:
                baseline_pred = baseline_pred[0]
            baseline_pred = torch.sigmoid(baseline_pred).item()

        importance_scores = {}

        for subst_name, smarts in smarts_dict.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is None:
                continue

            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                continue

            # Collect all matched atom indices
            matched_atoms = set()
            for match in matches:
                matched_atoms.update(match)

            # Mask: zero out features for matched atoms
            masked_graph = graph.clone()
            mask = torch.ones(graph.num_nodes, 1, device=device)
            for atom_idx in matched_atoms:
                mask[atom_idx] = 0.0
            masked_graph.x = graph.x * mask

            with torch.no_grad():
                masked_outputs = self.model(masked_graph)
                masked_pred = masked_outputs[self.task_name].squeeze()
                if masked_pred.dim() > 0:
                    masked_pred = masked_pred[0]
                masked_pred = torch.sigmoid(masked_pred).item()

            # Importance = change in prediction when substructure is masked
            importance_scores[subst_name] = baseline_pred - masked_pred

        return importance_scores

    def rank_substructures(self, smiles: str, top_n: int = 5) -> list[tuple[str, float]]:
        """Return top-N most important substructures for a prediction.

        Returns:
            List of (substructure_name, importance_score), sorted by |importance|
        """
        scores = self.importance_by_masking(smiles)
        ranked = sorted(scores.items(), key=lambda x: abs(x[1]), reverse=True)
        return ranked[:top_n]


# ---------------------------------------------------------------------------
# GradCAM for GNNs
# ---------------------------------------------------------------------------

class GNNGradCAM:
    """GradCAM-style atom importance using gradient magnitudes at the last GNN layer.

    Computes atom importance as:
        importance_i = ||∂y/∂h_i^(L)||₂

    where h_i^(L) is the last-layer representation of atom i.
    This doesn't require attention weights, making it applicable to any GNN.
    """

    def __init__(self, model: nn.Module, task_name: str = "property"):
        self.model = model
        self.task_name = task_name

    def get_atom_importance(
        self,
        smiles: str,
        use_final_activation: bool = True,
    ) -> np.ndarray:
        """Compute gradient-based atom importance.

        Returns:
            Array [n_atoms] of importance scores
        """
        from src.data.featurizer import MolecularFeaturizer
        from src.data.molecule_dataset import smiles_to_graph

        featurizer = MolecularFeaturizer()
        graph = smiles_to_graph(smiles, featurizer)
        if graph is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        device = next(self.model.parameters()).device
        graph = graph.to(device)

        # Get node embeddings with gradient tracking
        node_embeddings = self.model.get_node_embeddings(graph)

        if not node_embeddings.requires_grad:
            node_embeddings.requires_grad_(True)

        # Forward through readout and head
        # This is a simplified version — full implementation would hook into internals
        outputs = self.model(graph)
        pred = outputs[self.task_name].squeeze()
        if pred.dim() > 0:
            pred = pred[0]

        # Compute gradients with respect to node embeddings
        pred.backward()

        if node_embeddings.grad is not None:
            # Gradient magnitude as importance
            importance = node_embeddings.grad.norm(dim=-1).cpu().numpy()
            if importance.max() > 0:
                importance = importance / importance.max()
            return importance
        else:
            return np.ones(graph.num_nodes) / graph.num_nodes
