"""Evaluator Weco calls after each proposed edit.

Loads train.py fresh each run (Weco rewrites it in place), executes the
pipeline, and prints a single `auc_roc: 0.xxxxxx` line that Weco parses as
the metric.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_module(path: str):
    spec = importlib.util.spec_from_file_location("train_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    train = load_module(str(Path(__file__).parent / "train.py"))
    auc = train.run_pipeline()

    if not (0.0 <= auc <= 1.0):
        print(f"Constraint violated: AUC-ROC out of range ({auc})")
        return 1

    print(f"auc_roc: {auc:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
