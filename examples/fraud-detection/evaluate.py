"""Evaluator — FROZEN. Loads data, runs FeatureBuilder + train_and_evaluate,
prints `auc_roc: 0.xxxxxx`.

This file is the API enforcement boundary. Weco never edits it.

The interface contract this file enforces:
- isFraud and TransactionID are stripped before X reaches FeatureBuilder.
- val data is never passed to fit() — only X_train + y_train.
- transform() is called once each on X_train and X_val with no `y`.
- Model code receives only ndarrays — no DataFrame metadata to peek at.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    here = Path(__file__).resolve().parent
    train_df = pd.read_parquet(here / "data" / "base_train_small.parquet")
    val_df = pd.read_parquet(here / "data" / "base_val_small.parquet")

    y_train = train_df["isFraud"].values.astype("int32")
    y_val = val_df["isFraud"].values.astype("int32")

    # Strip target and ID before either file's code can see them.
    X_train = train_df.drop(columns=["isFraud", "TransactionID"])
    X_val = val_df.drop(columns=["isFraud", "TransactionID"])

    # Import here so that any syntax error in features.py / model.py surfaces
    # as a real error, not a silent module-cache hit.
    sys.path.insert(0, str(here))
    from features import FeatureBuilder
    from model import train_and_evaluate

    fb = FeatureBuilder().fit(X_train, y_train)
    X_train_t = fb.transform(X_train)
    X_val_t = fb.transform(X_val)

    if X_train_t.shape[1] != X_val_t.shape[1]:
        print(
            f"Constraint violated: train and val transform produced different "
            f"feature counts ({X_train_t.shape[1]} vs {X_val_t.shape[1]})"
        )
        return 1

    auc = train_and_evaluate(X_train_t, y_train, X_val_t, y_val)
    if not (0.0 <= auc <= 1.0):
        print(f"Constraint violated: AUC out of range ({auc})")
        return 1

    print(f"auc_roc: {auc:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
