"""Minimal example for loading a species task and building MHGNano."""

from data import load_species_data
from models import HyperGraphMetaNetwork, MetaLearner


features, hypergraph, splits = load_species_data(
    "208964",
    data_root="./data/yuanhe",
    model_size_tag="8M",
    fold=0,
)
model = HyperGraphMetaNetwork(in_channels=features.shape[1])
meta = MetaLearner(model, inner_lr=1e-3, inner_steps=5, meta_lr=1e-4)

print(f"features: {tuple(features.shape)}")
print(f"hypergraph vertices: {hypergraph.num_v}")
print(f"train pairs: {len(splits['train'][0])}")
print(f"validation pairs: {len(splits['val'][0])}")
print(f"test pairs: {len(splits['test'][0])}")
print(f"trainable parameters: {sum(p.numel() for p in model.parameters()):,}")
