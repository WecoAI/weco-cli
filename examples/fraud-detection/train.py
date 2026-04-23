"""Baseline fraud-detection pipeline on IEEE-CIS. Weco will optimize this file.

Weco can modify anything in `build_features` and `train_and_evaluate`. The
`run_pipeline` function is the entry point called by `evaluate.py`.

Keep the final print format exactly as `auc_roc: 0.xxxxxx` so Weco can parse
the metric. Everything else is fair game to rewrite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score


def build_features(
    train_df: pd.DataFrame, val_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build features from the base data. Returns (X_train, y_train, X_val, y_val).

    This is a small starting set. Weco can replace or extend it — the case study
    found UID-based aggregations (card1 + addr1 + account-creation-day estimate),
    target encoding with out-of-fold protection, frequency encoding, and velocity
    features are the most impactful additions.
    """
    y_train = train_df["isFraud"].values.astype(np.int32)
    y_val = val_df["isFraud"].values.astype(np.int32)

    n_train = len(train_df)
    df = pd.concat([train_df, val_df], axis=0, ignore_index=True)

    # Drop the label BEFORE any cross-column aggregation to avoid target leakage.
    df = df.drop(columns=["isFraud", "TransactionID"])

    # --- Time features from TransactionDT (seconds offset from a reference date) ---
    df["hour"] = (df["TransactionDT"] // 3600) % 24
    df["day_of_week"] = (df["TransactionDT"] // 86400) % 7
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # --- Amount features ---
    df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])
    df["TransactionAmt_decimal"] = (
        df["TransactionAmt"] - df["TransactionAmt"].astype(int)
    ).round(2)
    df["TransactionAmt_is_round"] = (df["TransactionAmt_decimal"] == 0).astype(np.int8)

    # --- Simple aggregations on card1 / addr1 ---
    for key in ["card1", "addr1"]:
        grp = df.groupby(key)["TransactionAmt"]
        df[f"{key}_amt_mean"] = grp.transform("mean")
        df[f"{key}_amt_std"] = grp.transform("std").fillna(0)
        df[f"{key}_amt_count"] = grp.transform("count")

    # --- Frequency encoding for high-cardinality categoricals ---
    for col in ["card1", "card2", "card5", "addr1"]:
        if col in df.columns:
            freq = df[col].value_counts(normalize=True)
            df[f"{col}_freq"] = df[col].map(freq).fillna(0)

    df = df.drop(columns=["TransactionDT"])
    X = df.values.astype(np.float32)
    return X[:n_train], y_train, X[n_train:], y_val


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    """Train LightGBM and return AUC-ROC on the validation set.

    Reasonable-but-not-heavily-tuned hyperparameters. A fraud team would
    typically run Optuna for 50-100 trials on these — there is headroom.
    """
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "max_depth": -1,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": 1,
        "n_jobs": 4,
        "verbose": -1,
        "seed": 42,
    }

    train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, free_raw_data=False)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )

    y_pred = model.predict(X_val)
    return float(roc_auc_score(y_val, y_pred))


def run_pipeline() -> float:
    train_df = pd.read_parquet("data/base_train_small.parquet")
    val_df = pd.read_parquet("data/base_val_small.parquet")
    X_train, y_train, X_val, y_val = build_features(train_df, val_df)
    return train_and_evaluate(X_train, y_train, X_val, y_val)


if __name__ == "__main__":
    auc = run_pipeline()
    print(f"auc_roc: {auc:.6f}")
