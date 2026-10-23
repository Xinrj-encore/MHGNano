#!/usr/bin/env python3
"""Command-line entry point for deterministic NAI pair preprocessing."""

from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.prepare_nai_dataset import main


if __name__ == "__main__":
    main()
