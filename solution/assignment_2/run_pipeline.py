from __future__ import annotations

import argparse
from pathlib import Path
import sys


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR / "src"))

from operational import run_train_and_predict


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and predict with an Assignment 2 operational config.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()
    result = run_train_and_predict(args.config)
    print("train:")
    for key, value in result["train"].items():
        print(f"  {key}: {value}")
    print("predict:")
    for key, value in result["predict"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
