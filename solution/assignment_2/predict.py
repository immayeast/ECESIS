from __future__ import annotations

import argparse
from pathlib import Path
import sys


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR / "src"))

from operational import load_config, predict_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Assignment 2 operational forecast inference.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--train-if-missing", action="store_true", help="Train the model if the artifact is missing.")
    args = parser.parse_args()
    metadata = predict_from_config(load_config(args.config), train_if_missing=args.train_if_missing)
    for key, value in metadata.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
