# Fraud Detection (IEEE-CIS)

Optimize a tabular fraud-detection pipeline on the
[IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) Kaggle
dataset (real Vesta payment transactions). Weco rewrites `train.py` — both
feature engineering and the LightGBM configuration — to maximize AUC-ROC on a
held-out, time-based validation split.

This example reproduces the setup from Weco's fraud-detection case study
([blog post](https://weco.ai/blog/framing-the-problem),
[code](https://github.com/WecoAI/fraud-detection-case-study)). The example's
baseline is **AUC ≈ 0.9102** (deterministic; verifiable via the SHA-256s
in `prepare_data.py`). The case study reported 0.914, which used a slightly
leaky `build_features` (concat-then-groupby on train+val); this example's
`train.py` fits all encoders on `train_df` only — no time-leakage. With the
bundled `instructions.md` and 200 steps of `gemini-3.1-pro-preview`, expect
AUC in the **0.928–0.933** range.

## Prerequisites

1. **Kaggle API token**. Put a valid `kaggle.json` at `~/.kaggle/kaggle.json`
   (see [Kaggle API credentials](https://github.com/Kaggle/kaggle-api#api-credentials)),
   then `chmod 600 ~/.kaggle/kaggle.json` to silence the permissions warning.
2. **You must join the competition.** Visit
   <https://www.kaggle.com/c/ieee-fraud-detection> and click "Late Submission" /
   "Join Competition" to accept the rules. Without this,
   `prepare_data.py` will fail with `403 Forbidden` from the Kaggle API —
   this is the single most common first-time friction.
3. **Weco API key** (free tier is fine). See the
   [Weco docs](https://docs.weco.ai).

## Setup

```bash
cd examples/fraud-detection

# Virtualenv is strongly recommended — modern Python installs (Debian/Ubuntu,
# recent Homebrew) refuse `pip install` to the system site-packages under
# PEP 668. If you skip this step you'll hit
# `error: externally-managed-environment`.
python3 -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
# After activation, `python` resolves to the venv's interpreter.

pip install -r requirements.txt

# Downloads ~120MB of CSVs, builds a small 100K/25K parquet split.
# Time-based split: last 20% of transactions by TransactionDT = validation.
# ~2-3 minutes on a modern laptop.
python prepare_data.py
```

After this you should have:

```
data/
  train_transaction.csv, train_identity.csv, test_*.csv  # raw
  base_train_small.parquet   # 100K rows, time-ordered
  base_val_small.parquet     # 25K rows, later in time
```

## Quick sanity check

Run the baseline once to confirm everything loads:

```bash
python evaluate.py
# → auc_roc: 0.910171   (deterministic, takes ~30s)
```

If you see an AUC in the 0.90-0.92 range, you're ready.

## Run Weco

The "default" run uses the full EDA + techniques instructions (recommended —
they contain the column semantics and known-good techniques for this dataset):

```bash
weco run --source train.py \
     --eval-command "python evaluate.py" \
     --metric auc_roc \
     --goal maximize \
     --steps 50 \
     --model gemini-3.1-pro-preview \
     --additional-instructions instructions.md \
     --eval-timeout 300 \
     --log-dir .runs/fraud-detection
```

Expected trajectory:

- Steps 1–10: Weco explores — tries log-amount, simple aggregations, category
  encodings. AUC moves into 0.918-0.925.
- Steps 10–50: builds UID-style features (card1 + addr1 + account-creation
  estimate via `D1`), target encoding with out-of-fold protection, velocity
  features. AUC climbs to 0.928-0.933.
- Beyond step 50: diminishing returns; the pooled mean across 6 seeds in our
  case study was 0.9305 ± 0.0035.

## Explanation

- `--source train.py` — the file Weco rewrites. Both `build_features` and
  `train_and_evaluate` are fair game.
- `--eval-command "python evaluate.py"` — called after every proposed edit;
  reimports `train.py`, runs the pipeline, prints `auc_roc: 0.xxxxxx`. Weco
  parses the last line matching `--metric`.
- `--metric auc_roc --goal maximize` — Weco optimizes the metric printed by
  the evaluator.
- `--additional-instructions instructions.md` — injects domain context into
  every optimization step. **This is what mostly matters.** See the
  case study: EDA-level instructions (what each column means in this
  specific dataset) drive most of the gain. Kaggle-classic techniques are
  typically already in the LLM's pretraining distribution. Feed the optimizer
  what it couldn't already know — dataset-specific semantics, proprietary
  heuristics, internal constraints.
- `--eval-timeout 300` — one eval takes ~30-60s; 300s gives headroom for
  feature-heavy proposals.

## Things to try

1. **No instructions baseline**: remove `--additional-instructions` and watch
   variance across seeds balloon (std ~0.008 vs ~0.002 with instructions).
   Also watch for silently-leaky proposals (see below).
2. **EDA only**: keep only the column-meaning section of `instructions.md` —
   the case study found this accounts for most of the mean gain.
3. **Scope restriction**: point Weco at `train.py`'s `build_features` only by
   editing the file to expose just that function (or split the pipeline into
   `features.py` + `model.py`). In our case study, features-only delivered
   most of the improvement that full-pipeline did.

## Watch out for silent leakage

Two flavors both show up in IEEE-CIS optimization runs.

**Target leakage** — `isFraud` ends up encoded into features. A plausible
idea like "count how many columns are zero per row" becomes leaky if the
dataframe still contains `isFraud`, because fraud rows contribute a
different count than non-fraud rows. The baseline `build_features` drops
`isFraud` and `TransactionID` up-front; don't let proposals reintroduce
aggregations on a dataframe that still has the label. The case study walks
through a real instance where this bug reported AUC 0.9591 that dropped to
0.9154 after a one-line fix — see
<https://weco.ai/blog/framing-the-problem>.

**Time leakage** — validation-period statistics leak into train features.
This is a time-based split; at serving time you don't have the val period.
Any encoder, groupby aggregation, frequency count, or target encoding must
be **fit on `train_df` only** and then applied to both splits. The baseline
demonstrates the pattern — fit `card1_amt_mean` on train, `.join` it onto
both train and val, fill unseen val keys with a train-global default. If a
proposal does `pd.concat([train_df, val_df]).groupby(...)`, that's a leak
even if it drops `isFraud` first.

Signs a run has one of these leaks (AUC suspiciously high on this 100K/25K
subsample, e.g. > 0.95):

- Any `df.sum`/`df.mean`/`(df == x)` across all columns before the label is
  dropped.
- Target encoding without out-of-fold protection (encoder fit on full train
  then applied to train).
- Groupby / value-counts / target encoders fit on `pd.concat([train, val])`.
- Features computed using validation data at all — velocity features that
  sort train + val together and take row-wise diffs, etc.

## Citing the case study

If you use this example, the underlying numbers come from
<https://github.com/WecoAI/fraud-detection-case-study>. Setup: 200 steps,
3 seeds per condition (6 for the Full pipeline + Full-instructions condition,
pooled since the two ablations share that configuration),
`gemini-3.1-pro-preview`.
