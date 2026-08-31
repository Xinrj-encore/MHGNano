"""Load a checkpoint and report its recorded evaluation metadata."""

import argparse
import json
from pathlib import Path

import torch


parser = argparse.ArgumentParser()
parser.add_argument("checkpoint")
args = parser.parse_args()

payload = torch.load(args.checkpoint, map_location="cpu")
print(
    json.dumps(
        {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "epoch": payload.get("epoch"),
            "metrics": payload.get("metrics", {}),
            "hyperparams": payload.get("hyperparams", {}),
        },
        indent=2,
    )
)
