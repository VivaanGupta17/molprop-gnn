"""
MoleculeNet dataset loading, SMILES-to-graph conversion, and dataset splitting.

Supported datasets:
- BBBP (blood-brain barrier permeability) — binary classification
- HIV — binary classification
- Tox21 — multi-label classification (12 tasks)
- SIDER — multi-label classification (27 tasks)
- ESOL — regression (aqueous solubility, logS)
- FreeSolv — regression (hydration free energy)
- Lipophilicity — regression (logD at pH 7.4)

Split strategies:
- scaffold: Bemis-Murcko scaffold split (industry standard, recommended)
- random: Random split (optimistic, for comparison only)
- stratified: Stratified split for classification tasks

The scaffold split implementation follows:
Bemis & Murcko, "The Properties of Known Drugs," J. Med. Chem. 1996
Wu et al., "MoleculeNet," Chem. Sci. 2018
Hu et al., "Strategies for Pre-Training GNNs," ICLR 2020
"""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.request
from collections import defaultdict
from typing import Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader as PyGDataLoader

from src.data.featurizer import MolecularFeaturizer

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Dataset configurations
# ---------------------------------------------------------------------------

MOLECULENET_DATASETS = {
    "bbbp": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
        "smiles_col": "smiles",
        "label_cols": ["p_np"],
        "task_type": "classification",
        "description": "Blood-brain barrier permeability (1=permeable, 0=not)",
    },
    "hiv": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv",
        "smiles_col": "smiles",
        "label_cols": ["HIV_active"],
        "task_type": "classification",
        "description": "HIV replication inhibition",
    },
    "esol": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
        "smiles_col": "smiles",
        "label_cols": ["measured log solubility in mols per litre"],
        "task_type": "regression",
        "description": "Aqueous solubility (logS mol/L)",
    },
    "freesolv": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv",
        "smiles_col": "smiles",
        "label_cols": ["expt"],
        "task_type": "regression",
        "description": "Hydration free energy (kcal/mol)",
    },
    "lipophilicity": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv",
        "smiles_col": "smiles",
        "label_cols": ["exp"],
        "task_type": "regression",
        "description": "logD at pH 7.4",
    },
    "tox21": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
        "smiles_col": "smiles",
        "label_cols": [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
            "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
            "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
        ],
        "task_type": "classification",
        "description": "12 toxicology assays (Tox21 challenge)",
    },
    "sider": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz",
        "smiles_col": "smiles",
        "label_cols": None,  # all non-smiles columns
        "task_type": "classification",
        "description": "27 drug side-effect categories",
    },
}


# ---------------------------------------------------------------------------
# SMILES → PyG Data conversion
# ---------------------------------------------------------------------------

def smiles_to_graph(
    smiles: str,
    featurizer: MolecularFeaturizer,
    labels: Optional[np.ndarray] = None,
    mol_id: Optional[str] = None,
) -> Optional[Data]:
    """Convert a SMILES string to a PyG Data object.

    Args:
        smiles: SMILES string
        featurizer: MolecularFeaturizer instance
        labels: Optional label vector [n_tasks]
        mol_id: Optional molecule identifier (for tracking)

    Returns:
        PyG Data object or None if SMILES is invalid
    """
    feat = featurizer.featurize_smiles(smiles)
    if feat is None:
        return None

    data = Data(
        x=torch.tensor(feat["node_features"], dtype=torch.float32),
        edge_index=torch.tensor(feat["edge_index"], dtype=torch.long),
        edge_attr=torch.tensor(feat["edge_features"], dtype=torch.float32),
        smiles=smiles,
    )

    if labels is not None:
        data.y = torch.tensor(labels, dtype=torch.float32).unsqueeze(0)

    if "global_features" in feat:
        data.global_features = torch.tensor(feat["global_features"], dtype=torch.float32)

    if mol_id is not None:
        data.mol_id = mol_id

    return data


# ---------------------------------------------------------------------------
# Scaffold Split Implementation
# ---------------------------------------------------------------------------

def get_bemis_murcko_scaffold(smiles: str, include_chirality: bool = False) -> str:
    """Compute the Bemis-Murcko scaffold for a SMILES string.

    The Bemis-Murcko scaffold retains:
    - All ring systems
    - Linker chains between rings
    - Side chains are stripped

    This is the standard scaffold definition used in drug discovery for
    lead series identification and diversity analysis.

    Args:
        smiles: Input SMILES
        include_chirality: Whether to preserve chirality in scaffold

    Returns:
        Scaffold SMILES string, or empty string if computation fails
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for scaffold computation.")

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        scaffold = MurckoScaffold.MurckoScaffoldSmilesFromSmiles(
            smiles, includeChirality=include_chirality
        )
        return scaffold
    except Exception as e:
        logger.warning(f"Scaffold computation failed for {smiles}: {e}")
        return ""


def scaffold_split(
    smiles_list: list[str],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    balanced: bool = False,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Split a molecular dataset by Bemis-Murcko scaffold.

    Algorithm:
    1. Compute scaffold for each molecule
    2. Group molecules by scaffold
    3. Sort scaffold groups by size (largest first)
    4. Greedily assign scaffold groups to train/val/test to meet size targets

    This ensures:
    - All molecules with the same scaffold are in the same split
    - Test set contains scaffolds NOT seen during training
    - Generalization estimate is honest

    Args:
        smiles_list: List of SMILES strings
        train_frac: Fraction for training set
        val_frac: Fraction for validation set
        test_frac: Fraction for test set
        balanced: If True, put largest scaffolds in test (harder generalization)
                  If False, largest scaffolds go to train (standard protocol)
        seed: Random seed for shuffling within scaffold groups of equal size

    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    rng = np.random.default_rng(seed)

    # Step 1: Compute scaffolds
    scaffold_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, smiles in enumerate(smiles_list):
        scaffold = get_bemis_murcko_scaffold(smiles)
        scaffold_to_indices[scaffold].append(idx)

    # Step 2: Sort scaffold groups
    scaffold_sets = list(scaffold_to_indices.values())

    if balanced:
        # Balanced: larger scaffolds → test (more challenging, used in some papers)
        scaffold_sets = sorted(scaffold_sets, key=len, reverse=True)
    else:
        # Standard: larger scaffolds → train (more training data)
        scaffold_sets = sorted(scaffold_sets, key=len)
        # Shuffle groups of equal size for randomization
        # (preserves that rare scaffolds go to val/test)

    # Step 3: Greedy assignment
    n_total = len(smiles_list)
    n_train = int(train_frac * n_total)
    n_val = int(val_frac * n_total)

    train_idx, val_idx, test_idx = [], [], []

    for group in scaffold_sets:
        if len(train_idx) + len(val_idx) + len(group) <= n_train + n_val:
            if len(val_idx) < n_val:
                val_idx.extend(group)
            else:
                train_idx.extend(group)
        else:
            if len(train_idx) < n_train:
                train_idx.extend(group)
            else:
                test_idx.extend(group)

    # Shuffle within splits
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    logger.info(
        f"Scaffold split: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test "
        f"({len(scaffold_to_indices)} unique scaffolds)"
    )

    return train_idx, val_idx, test_idx


def random_split(
    n: int,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Random split for comparison with scaffold split.

    WARNING: Random split leads to data leakage when molecules sharing
    scaffolds end up in both train and test. Use only for comparison purposes.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    n_train = int(train_frac * n)
    n_val = int(val_frac * n)

    train_idx = indices[:n_train].tolist()
    val_idx = indices[n_train:n_train + n_val].tolist()
    test_idx = indices[n_train + n_val:].tolist()

    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# MoleculeNet Dataset Class
# ---------------------------------------------------------------------------

class MoleculeNetDataset(InMemoryDataset):
    """PyTorch Geometric dataset for MoleculeNet benchmarks.

    Handles downloading, caching, featurization, and splitting.
    Processed graphs are cached to disk for fast reloading.

    Args:
        dataset_name: One of the keys in MOLECULENET_DATASETS
        root: Root directory for data storage
        featurizer: MolecularFeaturizer instance
        split: 'scaffold', 'random', or 'all'
        split_idx: 'train', 'val', 'test' (when split != 'all')
        transform: Optional transform applied on-the-fly
    """

    def __init__(
        self,
        dataset_name: str,
        root: str = "./data",
        featurizer: Optional[MolecularFeaturizer] = None,
        split: str = "scaffold",
        split_idx: str = "all",
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        seed: int = 42,
        transform=None,
        pre_transform=None,
    ):
        if dataset_name not in MOLECULENET_DATASETS:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. "
                f"Choose from: {list(MOLECULENET_DATASETS.keys())}"
            )

        self.dataset_name = dataset_name
        self.config = MOLECULENET_DATASETS[dataset_name]
        self.featurizer = featurizer or MolecularFeaturizer()
        self.split = split
        self.split_idx = split_idx
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.seed = seed

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

        # Apply split
        if split_idx != "all":
            self._apply_split()

    def _apply_split(self):
        """Apply the requested split and subset the dataset."""
        all_smiles = [self.get(i).smiles for i in range(len(self))]

        if self.split == "scaffold":
            train_idx, val_idx, test_idx = scaffold_split(
                all_smiles, self.train_frac, self.val_frac, self.test_frac, seed=self.seed
            )
        elif self.split == "random":
            train_idx, val_idx, test_idx = random_split(
                len(all_smiles), self.train_frac, self.val_frac, self.test_frac, seed=self.seed
            )
        else:
            raise ValueError(f"Unknown split: {self.split}")

        split_map = {"train": train_idx, "val": val_idx, "test": test_idx}
        indices = split_map[self.split_idx]

        # Subset the dataset
        self.data, self.slices = self.collate([self.get(i) for i in indices])

    @property
    def raw_file_names(self) -> list[str]:
        url = self.config["url"]
        filename = url.split("/")[-1]
        return [filename]

    @property
    def processed_file_names(self) -> list[str]:
        return [f"{self.dataset_name}_processed.pt"]

    def download(self):
        """Download dataset from DeepChem S3 storage."""
        url = self.config["url"]
        filename = url.split("/")[-1]
        out_path = os.path.join(self.raw_dir, filename)

        if not os.path.exists(out_path):
            logger.info(f"Downloading {self.dataset_name} from {url}...")
            urllib.request.urlretrieve(url, out_path)
            logger.info(f"Downloaded to {out_path}")

    def process(self):
        """Process raw CSV into PyG Data objects and save."""
        import pandas as pd

        raw_path = self.raw_paths[0]

        # Handle gzip
        if raw_path.endswith(".gz"):
            df = pd.read_csv(raw_path, compression="gzip")
        else:
            df = pd.read_csv(raw_path)

        smiles_col = self.config["smiles_col"]
        label_cols = self.config["label_cols"]

        # For SIDER, use all non-smiles columns as labels
        if label_cols is None:
            label_cols = [c for c in df.columns if c != smiles_col]

        logger.info(f"Processing {len(df)} molecules for {self.dataset_name}...")

        data_list = []
        n_failed = 0

        for idx, row in df.iterrows():
            smiles = str(row[smiles_col])
            labels = row[label_cols].values.astype(np.float32)

            graph = smiles_to_graph(smiles, self.featurizer, labels, mol_id=str(idx))
            if graph is None:
                n_failed += 1
                continue

            data_list.append(graph)

        logger.info(
            f"Processed {len(data_list)} molecules "
            f"({n_failed} invalid SMILES skipped)"
        )

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    @property
    def task_type(self) -> str:
        return self.config["task_type"]

    @property
    def num_tasks(self) -> int:
        label_cols = self.config["label_cols"]
        if label_cols is None:
            # SIDER: need to load to count
            return 27
        return len(label_cols)

    @property
    def task_names(self) -> list[str]:
        label_cols = self.config["label_cols"]
        if label_cols is None:
            return [f"sider_{i}" for i in range(27)]
        return label_cols


# ---------------------------------------------------------------------------
# Data Loaders
# ---------------------------------------------------------------------------

def get_dataloaders(
    dataset_name: str,
    root: str = "./data",
    split: str = "scaffold",
    batch_size: int = 32,
    num_workers: int = 0,
    featurizer: Optional[MolecularFeaturizer] = None,
    seed: int = 42,
) -> dict[str, PyGDataLoader]:
    """Create train/val/test DataLoaders for a MoleculeNet dataset.

    Args:
        dataset_name: MoleculeNet dataset name
        root: Data root directory
        split: 'scaffold' (recommended) or 'random'
        batch_size: Batch size for DataLoader
        num_workers: Number of DataLoader workers

    Returns:
        Dict with 'train', 'val', 'test' DataLoaders
    """
    common_kwargs = dict(
        dataset_name=dataset_name,
        root=root,
        featurizer=featurizer,
        split=split,
        seed=seed,
    )

    train_ds = MoleculeNetDataset(**common_kwargs, split_idx="train")
    val_ds = MoleculeNetDataset(**common_kwargs, split_idx="val")
    test_ds = MoleculeNetDataset(**common_kwargs, split_idx="test")

    loaders = {
        "train": PyGDataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
        ),
        "val": PyGDataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        ),
        "test": PyGDataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        ),
    }

    logger.info(
        f"Dataset '{dataset_name}' ({split} split): "
        f"{len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test"
    )

    return loaders


# ---------------------------------------------------------------------------
# Scaffold analysis utilities
# ---------------------------------------------------------------------------

def analyze_scaffold_overlap(
    smiles_train: list[str],
    smiles_test: list[str],
) -> dict:
    """Quantify scaffold overlap between train and test sets.

    Used to verify split quality and demonstrate the leakage problem
    with random splits vs scaffold splits.

    Returns:
        Dict with overlap statistics
    """
    train_scaffolds = set(get_bemis_murcko_scaffold(s) for s in smiles_train if s)
    test_scaffolds = set(get_bemis_murcko_scaffold(s) for s in smiles_test if s)

    # Remove empty scaffolds (acyclic molecules)
    train_scaffolds.discard("")
    test_scaffolds.discard("")

    overlap = train_scaffolds & test_scaffolds
    union = train_scaffolds | test_scaffolds

    return {
        "n_train_scaffolds": len(train_scaffolds),
        "n_test_scaffolds": len(test_scaffolds),
        "n_overlapping_scaffolds": len(overlap),
        "scaffold_overlap_fraction": len(overlap) / len(test_scaffolds) if test_scaffolds else 0,
        "jaccard_similarity": len(overlap) / len(union) if union else 0,
    }
