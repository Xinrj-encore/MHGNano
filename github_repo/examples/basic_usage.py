# Example: Training on Yuanhe dataset

from models import HyperGraphMetaNetwork, MetaLearner
from data import load_species_data, sample_meta_batch
from utils import Checkpointer, EarlyStopping

import torch
import random

# Set seeds
random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Species list
species_list = ["1326", "41514", "77635", "208964"]
train_species = species_list[:-1]

# Load data
species_data = {}
num_hyperedges = 0
for sp in train_species:
    X, hg, y = load_species_data(sp, "./data/yuanhe", "8M")
    species_data[sp] = (X, hg, y)
    num_hyperedges = max(num_hyperedges, len(hg.e))

# Create model
base = HyperGraphMetaNetwork(
    in_channels=320,
    hid_channels=256,
    num_classes=128,
    num_hyperedges=num_hyperedges,
).to(device)

meta = MetaLearner(
    base_model=base,
    inner_lr=1e-3,
    inner_steps=5,
    meta_lr=1e-4,
)

print(f"Model has {sum(p.numel() for p in base.parameters()):,} parameters")

# Training would continue here...
# See src/train.py for complete training loop
