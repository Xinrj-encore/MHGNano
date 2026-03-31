# HyperGraph Meta-Learning for Protein-Protein Interaction Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the implementation of **HyperGraph Meta-Learning Network** for cross-species Protein-Protein Interaction (PPI) prediction using Model-Agnostic Meta-Learning (MAML) and Hypergraph Neural Networks.

## Overview

Protein-protein interactions are fundamental to biological processes. This project addresses the challenge of predicting PPIs in species with limited labeled data by leveraging meta-learning across multiple source species. The model uses:

- **Hypergraph Neural Networks (HGNN)** to model complex protein relationships
- **Model-Agnostic Meta-Learning (MAML)** for fast adaptation to new species
- **Cross-species transfer learning** to handle data scarcity

## Key Features

- Meta-learning framework for cross-species PPI prediction
- Hypergraph-based protein interaction modeling
- Support for multiple species adaptation
- No BatchNorm/Dropout variants for stable meta-learning
- Comprehensive evaluation metrics (AUC-ROC, AUC-PR, F1, etc.)

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.12.1
- CUDA

### Setup

```bash

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Data Preparation

Organize your data in the following structure:
```
data/
├── yuanhe/
│   ├── species_1/
│   │   ├── PPI_cliques_Hyperedge_species_1_8M.pkl
│   │   └── protein_feature_CT_species_1_8M.pt
│   └── species_2/
│       └── ...
└── zhenhe/
    └── ...
```

### 2. Training

```bash
# Train on Yuanhe dataset
python -m src.train --dataset yuanhe --epochs 20000 --inner_steps 5

# Train on Zhenhe dataset
python -m src.train --dataset zhenhe --epochs 20000 --inner_lr 1e-3
```

### 3. Evaluation

```bash
python -m src.evaluate --checkpoint output/best_model.pt --dataset yuanhe
```

## Project Structure

```
.
├── src/                    # Source code
│   ├── models/            # Model architectures
│   │   ├── hypergraph.py  # HyperGraphMetaNetwork
│   │   ├── layers.py      # HGNNConv, Concentration
│   │   └── meta.py        # MetaLearner (MAML)
│   ├── data/              # Data loading and processing
│   │   └── dataset.py     # Dataset utilities
│   ├── utils/             # Helper functions
│   │   ├── metrics.py     # Evaluation metrics
│   │   └── checkpoint.py  # Model saving/loading
│   ├── train.py           # Training script
│   └── evaluate.py        # Evaluation script
├── configs/               # Configuration files
├── examples/              # Example scripts
├── tests/                 # Unit tests
├── requirements.txt       # Python dependencies
├── setup.py              # Package setup
└── README.md             # This file
```

## Model Architecture

### HyperGraphMetaNetwork

The core model consists of:

1. **HGNN Encoder**: Two-layer hypergraph convolutional network
   - Input: Protein features (320-dim ESM embeddings)
   - Hidden: 256-dim hypergraph representation
   - Output: 128-dim protein embeddings

2. **Concentration Layer**: Aggregates hyperedge information

3. **Classifier**: 3-layer MLP (512 → 1024 → 256 → 1)

### Meta-Learning (MAML)

- **Inner Loop**: Task-specific adaptation with gradient descent
- **Outer Loop**: Meta-parameter optimization across tasks
- **Support/Query Split**: Few-shot learning setup

## Datasets

### Supported Datasets

- **Yuanhe**: 4 species (1326, 41514, 77635, 208964)
- **Zhenhe**: Multiple species with varying sizes
- **Nano**: Small-scale species for few-shot testing

### Data Format

- **Hypergraph**: Pickled hyperedge lists (`PPI_cliques_Hyperedge_*.pkl`)
- **Features**: PyTorch tensors (`protein_feature_CT_*.pt`)
- **Labels**: Train/val/test indices in pickle format

## Training Configuration

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `inner_lr` | 1e-3 | Inner loop learning rate |
| `inner_steps` | 5 | Number of inner loop updates |
| `meta_lr` | 1e-4 | Meta learning rate |
| `meta_batch_size` | 3 | Number of tasks per meta-batch |
| `n_support` | 64 | Support set size |
| `n_query` | 64 | Query set size |

### Example Configurations

See `configs/` directory for dataset-specific configurations.

```
