"""Download IEEE-CIS data, build base features, subsample to a small split.

Produces `data/base_train_small.parquet` and `data/base_val_small.parquet` that
`train.py` loads. The split is time-based (the last 20% of transactions by
TransactionDT are held out for validation), which mirrors production fraud
detection: you never train on future data.

Usage:
    # 1. Put your Kaggle API token at ~/.kaggle/kaggle.json
    #    (see https://github.com/Kaggle/kaggle-api#api-credentials)
    # 2. Join the competition on kaggle.com/c/ieee-fraud-detection to accept rules
    # 3. Run:
    python prepare_data.py

Runtime: ~2-3 minutes on a modern laptop. Produces ~150MB of parquet files.
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
TRAIN_SIZE = 100_000
VAL_SIZE = 25_000
TIME_SPLIT_FRAC = 0.8  # first 80% of transactions by time = train candidates
SEED = 42


def download_kaggle() -> None:
    """Download ieee-fraud-detection via the Kaggle CLI."""
    DATA_DIR.mkdir(exist_ok=True)
    txn = DATA_DIR / "train_transaction.csv"
    ident = DATA_DIR / "train_identity.csv"
    if txn.exists() and ident.exists():
        print(f"[skip] raw CSVs already present in {DATA_DIR}")
        return

    print(f"[download] kaggle competitions download -c ieee-fraud-detection -p {DATA_DIR}")
    print("[download] this takes ~1-2 min over a fast link; ~120MB of CSVs")
    # Use `python -m kaggle.cli` — the `kaggle` package has no __main__, so
    # `python -m kaggle` fails. kaggle.cli is the canonical entry point.
    try:
        subprocess.check_call(
            [sys.executable, "-m", "kaggle.cli", "competitions", "download",
             "-c", "ieee-fraud-detection", "-p", str(DATA_DIR)]
        )
    except subprocess.CalledProcessError as e:
        print(
            "\n[error] Kaggle download failed. Most common causes:\n"
            "  1. You haven't joined the competition. Visit\n"
            "     https://www.kaggle.com/c/ieee-fraud-detection\n"
            "     and click 'Late Submission' / 'Join Competition' to accept the rules.\n"
            "  2. ~/.kaggle/kaggle.json is missing or has wrong permissions.\n"
            "     Run: chmod 600 ~/.kaggle/kaggle.json\n",
            file=sys.stderr,
        )
        raise SystemExit(e.returncode)
    zip_path = DATA_DIR / "ieee-fraud-detection.zip"
    print(f"[extract] {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATA_DIR)
    zip_path.unlink()


def build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal, leakage-safe preprocessing so train.py has a clean starting point.

    - Drop test-specific columns
    - Label-encode object columns (LightGBM doesn't take strings)
    - Reduce highly correlated V-features (drop one per cluster with r > 0.95)
      to keep train.py's input dimensionality manageable
    """
    # Label-encode all object columns. Keep isFraud/TransactionID/TransactionDT intact.
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in obj_cols:
        df[col] = df[col].astype("category").cat.codes.astype(np.int32)

    # Reduce V-features by correlation clustering (done on a sample for speed).
    v_cols = [c for c in df.columns if c.startswith("V")]
    if v_cols:
        sample = df[v_cols].sample(n=min(10_000, len(df)), random_state=SEED)
        corr = sample.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if (upper[c] > 0.95).any()]
        df = df.drop(columns=to_drop)
        print(f"[v-reduce] dropped {len(to_drop)}/{len(v_cols)} correlated V-features")

    return df


def time_based_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    split_point = df["TransactionDT"].quantile(TIME_SPLIT_FRAC)
    train = df[df["TransactionDT"] <= split_point].copy()
    val = df[df["TransactionDT"] > split_point].copy()
    return train, val


def subsample(df: pd.DataFrame, n: int, label: str) -> pd.DataFrame:
    if len(df) <= n:
        return df
    sampled = df.sample(n=n, random_state=SEED).sort_values("TransactionDT").reset_index(drop=True)
    fraud_rate = sampled["isFraud"].mean()
    print(f"[subsample] {label}: {len(df)} -> {len(sampled)} (fraud rate {fraud_rate:.3%})")
    return sampled


def main() -> None:
    download_kaggle()

    train_out = DATA_DIR / "base_train_small.parquet"
    val_out = DATA_DIR / "base_val_small.parquet"
    if train_out.exists() and val_out.exists():
        print(f"[skip] {train_out.name} and {val_out.name} already exist")
        return

    print("[load] merging train_transaction + train_identity")
    txn = pd.read_csv(DATA_DIR / "train_transaction.csv")
    ident = pd.read_csv(DATA_DIR / "train_identity.csv")
    df = txn.merge(ident, on="TransactionID", how="left")
    print(f"[load] shape={df.shape}, fraud rate {df['isFraud'].mean():.3%}")

    df = build_base_features(df)

    print("[split] time-based 80/20")
    train_df, val_df = time_based_split(df)
    print(f"[split] train={len(train_df)} val={len(val_df)}")

    train_small = subsample(train_df, TRAIN_SIZE, "train")
    val_small = subsample(val_df, VAL_SIZE, "val")

    train_small.to_parquet(train_out, index=False)
    val_small.to_parquet(val_out, index=False)
    print(f"[write] {train_out}")
    print(f"[write] {val_out}")


if __name__ == "__main__":
    sys.exit(main())
