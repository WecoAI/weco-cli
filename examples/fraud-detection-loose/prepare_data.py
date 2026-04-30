"""Download IEEE-CIS data and build the fixed train/val parquets used by train.py.

Produces `data/base_train_small.parquet` (100K rows, stratified by fraud) and
`data/base_val_small.parquet` (25K rows, time-later subsample). Identical SHA-256
to the parquets used in the published case study.

Usage:
    # 1. Put your Kaggle API token at ~/.kaggle/kaggle.json
    #    (see https://github.com/Kaggle/kaggle-api#api-credentials)
    # 2. Join the competition on kaggle.com/c/ieee-fraud-detection to accept rules
    # 3. Run:
    python prepare_data.py

Runtime: ~2-3 minutes on a modern laptop. Produces ~150MB of parquet files.

Pipeline (must stay byte-identical to the originals — see SHAs in the README):
1. Merge `train_transaction.csv` + `train_identity.csv` on TransactionID.
2. Time-based 80/20 split on TransactionDT (last 20% by time = validation).
3. V-feature correlation pruning: sample 10_000 rows from the FULL merged df with
   `random_state=42`, drop V-cols whose pairwise |corr| > 0.95.
4. Label-encode all `object`/`string` columns using categories from the
   `concat(train, val)` dtype, so the same string maps to the same int in both
   splits.
5. **Stratified** subsample to 100K train via global `np.random.seed(42)` +
   `np.random.choice` over fraud/legit indices (preserves the 3.5% fraud rate
   exactly), and a uniform 25K val subsample drawn from the same RNG state.

Each of these details matters for reproducing the published baseline AUC of
0.910171. In particular:
- "object" alone misses pandas-3 string-dtype columns; include "string" too.
- pandas `df.sample()` and `np.random.seed`+`np.random.choice` give DIFFERENT
  rows even with the same seed — the original used the latter.
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
TIME_SPLIT_FRAC = 0.8
SEED = 42
V_CORR_SAMPLE = 10_000
V_CORR_THRESHOLD = 0.95


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


def time_based_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_point = df["TransactionDT"].quantile(TIME_SPLIT_FRAC)
    train = df[df["TransactionDT"] <= split_point].copy()
    val = df[df["TransactionDT"] > split_point].copy()
    return train, val


def reduce_v_features(
    df_full: pd.DataFrame, train: pd.DataFrame, val: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Drop V-cols whose pairwise |corr| > threshold, sampled from FULL merged df."""
    v_cols = [c for c in df_full.columns if c.startswith("V")]
    if not v_cols:
        return train, val, []
    sample = df_full[v_cols].sample(n=min(V_CORR_SAMPLE, len(df_full)), random_state=SEED)
    corr = sample.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if (upper[c] > V_CORR_THRESHOLD).any()]
    return train.drop(columns=to_drop), val.drop(columns=to_drop), to_drop


def label_encode_with_combined_categories(
    train: pd.DataFrame, val: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Encode all object/string cols using categories from concat(train, val).

    Important: include both "object" AND "string" — pandas 3 strings have
    StringDtype and aren't picked up by `include=["object"]` alone.
    """
    obj_cols = train.select_dtypes(include=["object", "string"]).columns
    obj_cols = [c for c in obj_cols if c not in ("TransactionID", "isFraud")]
    for col in obj_cols:
        combined = pd.concat([train[col], val[col]]).astype("category")
        cats = combined.cat.categories
        train[col] = train[col].astype("category").cat.set_categories(cats).cat.codes
        val[col] = val[col].astype("category").cat.set_categories(cats).cat.codes
    return train, val


def stratified_subsample(
    train: pd.DataFrame, val: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train subsample preserving fraud rate; uniform val subsample.

    Uses ONE global `np.random.seed(42)` then sequential `np.random.choice`
    calls — the val subsample inherits the RNG state advanced by the train
    subsample. This sequential coupling matters for reproducibility.
    """
    np.random.seed(SEED)
    fraud_idx = train[train["isFraud"] == 1].index
    legit_idx = train[train["isFraud"] == 0].index
    fraud_rate = len(fraud_idx) / len(train)
    n_fraud = int(TRAIN_SIZE * fraud_rate)
    n_legit = TRAIN_SIZE - n_fraud
    si = np.sort(
        np.concatenate([
            np.random.choice(fraud_idx, n_fraud, replace=False),
            np.random.choice(legit_idx, n_legit, replace=False),
        ])
    )
    train_small = train.loc[si].reset_index(drop=True)
    val_small = val.iloc[
        np.random.choice(len(val), VAL_SIZE, replace=False)
    ].reset_index(drop=True)
    return train_small, val_small


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

    print("[split] time-based 80/20")
    train, val = time_based_split(df)
    print(f"[split] train={len(train)} val={len(val)}")

    print("[v-reduce] correlation pruning over full merged df")
    train, val, dropped = reduce_v_features(df, train, val)
    print(f"[v-reduce] dropped {len(dropped)} V cols (threshold {V_CORR_THRESHOLD})")

    print("[encode] label-encode object/string cols using combined categories")
    train, val = label_encode_with_combined_categories(train, val)

    print("[subsample] stratified train, uniform val (np.random.seed=42)")
    train_small, val_small = stratified_subsample(train, val)
    print(f"[subsample] train={len(train_small)} (fraud {train_small['isFraud'].mean():.3%}), "
          f"val={len(val_small)} (fraud {val_small['isFraud'].mean():.3%})")

    train_small.to_parquet(train_out, index=False)
    val_small.to_parquet(val_out, index=False)
    print(f"[write] {train_out}")
    print(f"[write] {val_out}")
    print()
    print("Expected SHA-256 (matches the published case study parquets):")
    print("  train: a2d7a6740559975b8e6d89bd605f1e29791dd7d3fee8abc6449552bbc18d29ae")
    print("  val:   8b426c8bf7fa845bc234dbce304b1107fd295143fac2398bab97b78805f50753")


if __name__ == "__main__":
    sys.exit(main())
