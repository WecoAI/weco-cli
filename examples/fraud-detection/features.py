"""Feature engineering — Weco optimizes this file (and only this file) for the
Features-only scope. fit() sees only train; transform() sees only X.

Interface contract enforced by evaluate.py:
- isFraud and TransactionID are stripped before X reaches FeatureBuilder.
- val data is never visible during fit() — time leakage is impossible.
- transform() has no `y` argument — val labels can't influence val features.

What you CAN do here:
- Fit frequency, target, and group encoders on (X_train, y_train) inside fit().
- Use K-fold OOF protection if you want target encoding *within* train.
- Construct UIDs (e.g. card1+addr1+account-creation-day proxy) and aggregate.
- Stash any state in `self.*` so transform() can apply it deterministically.

What you CANNOT do:
- Concatenate train+val (val is not in scope).
- Branch on y in transform() (it's not an argument).
- Recompute encoders during transform — only look up self.* state.

Output: a numpy float32 array. transform must produce the same n_features and
the same column order on both train and val.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureBuilder:
    def __init__(self) -> None:
        # State populated by fit, read by transform.
        self.freq_: dict[str, dict] = {}
        self.amt_stats_: dict[str, dict[str, dict]] = {}
        self.train_amt_mean_: float = 0.0
        self.train_amt_std_: float = 0.0

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "FeatureBuilder":
        # Frequency encoders fit on train values only.
        for col in ("card1", "card2", "card5", "addr1"):
            if col in X_train.columns:
                self.freq_[col] = X_train[col].value_counts(normalize=True).to_dict()

        # Group amount aggregations fit on train rows only.
        for key in ("card1", "addr1"):
            if key not in X_train.columns:
                continue
            grp = X_train.groupby(key)["TransactionAmt"]
            stats = grp.agg(["mean", "std", "count"]).fillna(0)
            self.amt_stats_[key] = {
                "mean": stats["mean"].to_dict(),
                "std": stats["std"].to_dict(),
                "count": stats["count"].to_dict(),
            }

        # Train-global defaults for unseen keys at transform time.
        self.train_amt_mean_ = float(X_train["TransactionAmt"].mean())
        self.train_amt_std_ = float(X_train["TransactionAmt"].std())
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Apply self.* state to X. Called once each on X_train and X_val."""
        out = pd.DataFrame(index=X.index)

        # Per-row time features (no cross-row dependency).
        out["hour"] = (X["TransactionDT"] // 3600) % 24
        out["day_of_week"] = (X["TransactionDT"] // 86400) % 7
        out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
        out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)

        # Per-row amount features.
        out["TransactionAmt"] = X["TransactionAmt"].astype(np.float32)
        out["TransactionAmt_log"] = np.log1p(X["TransactionAmt"])
        out["TransactionAmt_decimal"] = (
            X["TransactionAmt"] - X["TransactionAmt"].astype(int)
        ).round(2)
        out["TransactionAmt_is_round"] = (out["TransactionAmt_decimal"] == 0).astype(np.int8)

        # Frequency lookups (unseen keys → 0).
        for col in ("card1", "card2", "card5", "addr1"):
            if col in X.columns and col in self.freq_:
                out[f"{col}_freq"] = X[col].map(self.freq_[col]).fillna(0)

        # Group amount aggregations (unseen keys → train-global default).
        for key in ("card1", "addr1"):
            if key in X.columns and key in self.amt_stats_:
                s = self.amt_stats_[key]
                out[f"{key}_amt_mean"] = X[key].map(s["mean"]).fillna(self.train_amt_mean_)
                out[f"{key}_amt_std"] = X[key].map(s["std"]).fillna(self.train_amt_std_)
                out[f"{key}_amt_count"] = X[key].map(s["count"]).fillna(0)

        # Pass-through every remaining numeric column.
        for col in X.columns:
            if col == "TransactionDT":
                continue
            if col in out.columns:
                continue
            if pd.api.types.is_numeric_dtype(X[col]):
                out[col] = X[col].values

        return out.values.astype(np.float32)

    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series) -> np.ndarray:
        return self.fit(X_train, y_train).transform(X_train)
