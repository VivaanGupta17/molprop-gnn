"""
Setup configuration for molprop-gnn package.

Installs the src/ package and scripts as command-line entry points.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
this_dir = Path(__file__).parent
long_description = (this_dir / "README.md").read_text(encoding="utf-8")

setup(
    name="molprop-gnn",
    version="0.1.0",
    description=(
        "Graph Neural Networks for ADMET molecular property prediction "
        "on MoleculeNet benchmarks"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/molprop-gnn",
    license="MIT",
    python_requires=">=3.9",

    packages=find_packages(exclude=["tests*", "notebooks*", "docs*"]),

    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "rdkit>=2023.3.1",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "PyYAML>=6.0",
        "tqdm>=4.65.0",
    ],

    extras_require={
        "xgboost": ["xgboost>=1.7.0"],
        "optuna": ["optuna>=3.2.0"],
        "viz": ["matplotlib>=3.7.0", "seaborn>=0.12.0", "plotly>=5.15.0"],
        "wandb": ["wandb>=0.15.0"],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ],
        "all": [
            "xgboost>=1.7.0",
            "optuna>=3.2.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
    },

    entry_points={
        "console_scripts": [
            "molprop-train=scripts.train:main",
            "molprop-evaluate=scripts.evaluate:main",
            "molprop-predict=scripts.predict_smiles:main",
        ],
    },

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],

    keywords=[
        "drug discovery", "machine learning", "graph neural networks",
        "molecular property prediction", "ADMET", "cheminformatics",
        "MoleculeNet", "MPNN", "GAT", "GIN", "PyTorch Geometric",
    ],
)
