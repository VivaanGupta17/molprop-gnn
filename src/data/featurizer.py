"""
Molecular featurization: atom features, bond features, and global descriptors.

Atom and bond featurization is central to how well a GNN can learn molecular
structure. The choices here reflect what's standard in the MoleculeNet/GNN
literature (Hu et al. 2020, Yang et al. 2019 Chemprop).

Design principles:
- Use one-hot encoding for discrete features (atom type, hybridization)
- Use continuous values only where they have meaningful scale (mass, partial charge)
- Be consistent — all features should be normalized to similar magnitude
- Store indices for categorical features (used by embedding layers in models)
- Include pharmacophoric features (H-bond donor/acceptor) — critical for ADMET
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
    from rdkit.Chem.rdchem import (
        Atom, Bond, BondType, HybridizationType, ChiralType, StereoType
    )
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logger.warning("RDKit not available. Molecular featurization will not work.")


# ---------------------------------------------------------------------------
# Atom feature vocabularies
# ---------------------------------------------------------------------------

# Elements commonly seen in drug-like molecules (+ UNK)
ATOM_TYPES = [
    "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca",
    "Fe", "As", "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb", "Sn", "Ag",
    "Pd", "Co", "Se", "Ti", "Zn", "H", "Li", "Ge", "Cu", "Au", "Ni",
    "Cd", "In", "Mn", "Zr", "Cr", "Pt", "Hg", "Pb",
]  # len=43, index 43=UNK

HYBRIDIZATION_TYPES = [
    HybridizationType.SP,
    HybridizationType.SP2,
    HybridizationType.SP3,
    HybridizationType.SP3D,
    HybridizationType.SP3D2,
    HybridizationType.S,
    HybridizationType.OTHER,
]  # len=7, index 7=UNK

CHIRALITY_TYPES = [
    ChiralType.CHI_UNSPECIFIED,
    ChiralType.CHI_TETRAHEDRAL_CW,
    ChiralType.CHI_TETRAHEDRAL_CCW,
    ChiralType.CHI_OTHER,
]

# Bond type vocabulary
BOND_TYPES = [
    BondType.SINGLE,
    BondType.DOUBLE,
    BondType.TRIPLE,
    BondType.AROMATIC,
]

BOND_STEREO = [
    StereoType.STEREONONE,
    StereoType.STEREOANY,
    StereoType.STEREOZ,
    StereoType.STEREOE,
    StereoType.STEREOCIS,
    StereoType.STEREOTRANS,
]


def safe_index(alist: list, elem) -> int:
    """Return index of elem in alist, or last index (UNK) if not found."""
    try:
        return alist.index(elem)
    except ValueError:
        return len(alist)  # UNK index


# ---------------------------------------------------------------------------
# Atom Featurizer
# ---------------------------------------------------------------------------

class AtomFeaturizer:
    """Compute feature vector for a single RDKit atom.

    Output format (used by AtomEncoder in models):
    Index 0: atom type index (0-43, 44=UNK)
    Index 1: degree index (0-10, 11=UNK)
    Index 2: formal charge index (offset by 5, so -5..+5 → 0..10, 11=UNK)
    Index 3: hybridization index (0-7, 7=UNK)
    Index 4: num_Hs index (0-4, 5=UNK)
    Index 5: is_aromatic (float 0/1)
    Index 6: is_in_ring (float 0/1)
    Index 7: normalized mass (mass / 100)

    Total: 8 features (5 categorical indices + 3 continuous)

    Extended version adds pharmacophoric features (H-bond donor/acceptor,
    charged, hydrophobic) — toggle with include_pharmacophore=True.
    """

    def __init__(self, include_pharmacophore: bool = True, include_chirality: bool = True):
        self.include_pharmacophore = include_pharmacophore
        self.include_chirality = include_chirality

    def featurize(self, atom: "Atom") -> np.ndarray:
        """Compute feature vector for a single atom.

        Returns:
            Feature array of shape [8] (or [11] with pharmacophore/chirality)
        """
        features = [
            # Categorical (stored as indices for embedding layers)
            safe_index(ATOM_TYPES, atom.GetSymbol()),
            safe_index(list(range(11)), atom.GetDegree()),
            safe_index(list(range(-5, 6)), atom.GetFormalCharge()),
            safe_index(HYBRIDIZATION_TYPES, atom.GetHybridization()),
            safe_index(list(range(5)), atom.GetTotalNumHs()),
            # Continuous
            float(atom.GetIsAromatic()),
            float(atom.IsInRing()),
            atom.GetMass() / 100.0,   # normalize: C=0.12, I=1.27
        ]

        if self.include_pharmacophore:
            # Pharmacophoric features critical for ADMET prediction
            features.extend([
                float(atom.GetTotalNumHs() > 0 and atom.GetAtomicNum() in (7, 8)),  # H-bond donor
                float(atom.GetAtomicNum() in (7, 8)),                                # H-bond acceptor
                float(atom.GetFormalCharge() != 0),                                 # charged
            ])

        if self.include_chirality:
            chirality_idx = safe_index(CHIRALITY_TYPES, atom.GetChiralTag())
            features.append(float(chirality_idx) / len(CHIRALITY_TYPES))

        return np.array(features, dtype=np.float32)

    @property
    def feature_dim(self) -> int:
        dim = 8
        if self.include_pharmacophore:
            dim += 3
        if self.include_chirality:
            dim += 1
        return dim

    @property
    def feature_names(self) -> list[str]:
        names = [
            "atom_type_idx", "degree_idx", "formal_charge_idx",
            "hybridization_idx", "num_hs_idx",
            "is_aromatic", "is_in_ring", "mass_normalized",
        ]
        if self.include_pharmacophore:
            names += ["hb_donor", "hb_acceptor", "is_charged"]
        if self.include_chirality:
            names += ["chirality_normalized"]
        return names


# ---------------------------------------------------------------------------
# Bond Featurizer
# ---------------------------------------------------------------------------

class BondFeaturizer:
    """Compute feature vector for a single RDKit bond.

    Output format (used by BondEncoder in models):
    Index 0: bond type index (0=single, 1=double, 2=triple, 3=aromatic)
    Index 1: is_conjugated (float 0/1)
    Index 2: is_in_ring (float 0/1)
    Index 3: stereo index (0-5)

    Total: 4 features
    """

    def featurize(self, bond: "Bond") -> np.ndarray:
        features = [
            safe_index(BOND_TYPES, bond.GetBondType()),
            float(bond.GetIsConjugated()),
            float(bond.IsInRing()),
            safe_index(BOND_STEREO, bond.GetStereo()),
        ]
        return np.array(features, dtype=np.float32)

    @property
    def feature_dim(self) -> int:
        return 4

    @property
    def feature_names(self) -> list[str]:
        return ["bond_type_idx", "is_conjugated", "is_in_ring", "stereo_idx"]


# ---------------------------------------------------------------------------
# Global Molecular Descriptor Featurizer
# ---------------------------------------------------------------------------

class GlobalDescriptorFeaturizer:
    """Compute whole-molecule descriptors for augmenting graph representations.

    These descriptors capture global molecular properties that are difficult
    to learn from local message passing alone:
    - Molecular weight
    - LogP (lipophilicity — Wildman-Crippen)
    - TPSA (topological polar surface area — critical for BBB/absorption)
    - Rotatable bonds (conformational flexibility)
    - H-bond donors/acceptors
    - Ring count
    - Fraction sp3 carbons (Fsp3 — 3D character, linked to solubility)
    - Formal charge

    These are core Lipinski/Veber rule descriptors used daily in drug discovery.
    """

    DESCRIPTOR_NAMES = [
        "MolWt", "MolLogP", "TPSA", "NumRotatableBonds",
        "NumHDonors", "NumHAcceptors", "RingCount", "FractionCSP3",
        "NumAromaticRings", "NumAliphaticRings", "FormalCharge",
        "NumHeavyAtoms", "NumStereocenters",
    ]

    def featurize(self, mol: "Chem.Mol") -> np.ndarray:
        """Compute normalized global descriptors for a molecule."""
        try:
            desc = [
                Descriptors.MolWt(mol) / 500.0,                            # normalize to ~1 for 500 Da
                Descriptors.MolLogP(mol) / 5.0,                            # normalize to ~1 for logP=5
                rdMolDescriptors.CalcTPSA(mol) / 140.0,                    # Veber rule: <140 Å²
                rdMolDescriptors.CalcNumRotatableBonds(mol) / 10.0,
                rdMolDescriptors.CalcNumHBD(mol) / 5.0,                    # Lipinski: ≤5
                rdMolDescriptors.CalcNumHBA(mol) / 10.0,                   # Lipinski: ≤10
                rdMolDescriptors.CalcNumRings(mol) / 5.0,
                rdMolDescriptors.CalcFractionCSP3(mol),                    # already 0-1
                rdMolDescriptors.CalcNumAromaticRings(mol) / 5.0,
                rdMolDescriptors.CalcNumAliphaticRings(mol) / 5.0,
                float(Chem.GetFormalCharge(mol)) / 2.0,
                mol.GetNumHeavyAtoms() / 50.0,
                float(len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))) / 5.0,
            ]
        except Exception as e:
            logger.warning(f"Descriptor computation failed: {e}")
            desc = [0.0] * len(self.DESCRIPTOR_NAMES)

        return np.array(desc, dtype=np.float32)

    @property
    def feature_dim(self) -> int:
        return len(self.DESCRIPTOR_NAMES)


# ---------------------------------------------------------------------------
# Unified Featurizer
# ---------------------------------------------------------------------------

class MolecularFeaturizer:
    """Unified featurizer for GNN input construction.

    Wraps atom, bond, and global descriptor featurizers.
    Used by molecule_dataset.py to convert SMILES into PyG Data objects.

    Args:
        include_pharmacophore: Add H-bond donor/acceptor/charge atom features
        include_chirality: Encode chirality in atom features
        include_global: Compute whole-molecule descriptors as a global feature
    """

    def __init__(
        self,
        include_pharmacophore: bool = True,
        include_chirality: bool = True,
        include_global: bool = True,
    ):
        self.atom_featurizer = AtomFeaturizer(
            include_pharmacophore=include_pharmacophore,
            include_chirality=include_chirality,
        )
        self.bond_featurizer = BondFeaturizer()
        self.global_featurizer = GlobalDescriptorFeaturizer() if include_global else None

    def featurize_mol(self, mol: "Chem.Mol") -> Optional[dict]:
        """Compute all features for a molecule.

        Returns:
            Dict with keys:
            - 'node_features': np.ndarray [num_atoms, atom_feat_dim]
            - 'edge_features': np.ndarray [num_bonds*2, bond_feat_dim]
            - 'edge_index': np.ndarray [2, num_bonds*2]
            - 'global_features': np.ndarray [global_feat_dim] (optional)
        Returns None if mol is None or has no atoms.
        """
        if mol is None or mol.GetNumAtoms() == 0:
            return None

        # Atom features
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(self.atom_featurizer.featurize(atom))
        node_features = np.stack(atom_features)  # [N, atom_feat_dim]

        # Bond features (each bond appears twice: i→j and j→i)
        edge_index_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_feat = self.bond_featurizer.featurize(bond)

            # Add both directions
            edge_index_list.extend([[i, j], [j, i]])
            edge_features_list.extend([bond_feat, bond_feat])

        if len(edge_index_list) == 0:
            # Single atom — add self-loop
            edge_index_list = [[0, 0]]
            edge_features_list = [np.zeros(self.bond_featurizer.feature_dim, dtype=np.float32)]

        edge_index = np.array(edge_index_list, dtype=np.int64).T   # [2, E]
        edge_features = np.stack(edge_features_list)                # [E, bond_feat_dim]

        result = {
            "node_features": node_features,
            "edge_index": edge_index,
            "edge_features": edge_features,
        }

        if self.global_featurizer is not None:
            result["global_features"] = self.global_featurizer.featurize(mol)

        return result

    def featurize_smiles(self, smiles: str) -> Optional[dict]:
        """Featurize a SMILES string. Returns None for invalid SMILES."""
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for molecular featurization.")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Cannot parse SMILES: {smiles}")
            return None
        return self.featurize_mol(mol)

    @property
    def atom_feature_dim(self) -> int:
        return self.atom_featurizer.feature_dim

    @property
    def bond_feature_dim(self) -> int:
        return self.bond_featurizer.feature_dim

    @property
    def global_feature_dim(self) -> Optional[int]:
        return self.global_featurizer.feature_dim if self.global_featurizer else None

# Validate SMILES before featurization
