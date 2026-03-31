# Example: Loading and evaluating a trained model

import torch
from models import HyperGraphMetaNetwork, MetaLearner
from data import load_species_data

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load checkpoint
checkpoint_path = "./output/best_model.pt"
checkpoint = torch.load(checkpoint_path, map_location=device)

# Create model
base = HyperGraphMetaNetwork(
    in_channels=320,
    hid_channels=256,
    num_classes=128,
    num_hyperedges=10000,
).to(device)

# Load weights
base.load_state_dict(checkpoint["model_state_dict"])

# Create meta-learner
meta = MetaLearner(
    base_model=base,
    inner_lr=1e-3,
    inner_steps=5,
    meta_lr=1e-4,
)

# Load test data
X, hg, y = load_species_data("208964", "./data/yuanhe", "8M")

print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
print(f"Checkpoint metrics: {checkpoint.get('metrics', {})}")

# Evaluation would continue here...
# See src/evaluate.py for complete evaluation
