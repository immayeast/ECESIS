from __future__ import annotations

import argparse
from pathlib import Path
import sys


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR / "src"))

from operational import load_config, train_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an Assignment 2 operational forecast model.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()
    metadata = train_from_config(load_config(args.config))
    for key, value in metadata.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
