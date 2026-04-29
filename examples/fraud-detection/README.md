# Fraud Detection (IEEE-CIS) — strict fit/transform API

Optimize a tabular fraud-detection pipeline on the
[IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)
Kaggle dataset (real Vesta payment transactions). Weco rewrites two files —
`features.py` (a `FeatureBuilder` with `fit`/`transform`) and `model.py` (a
`train_and_evaluate` function) — independently, separately, or together,
to maximize AUC-ROC on a held-out time-based validation split.

This example reproduces Weco's fraud-detection case study
([blog post](https://weco.ai/blog/framing-the-problem),
[code](https://github.com/WecoAI/fraud-detection-case-study)) with an
**API that makes train/val leakage impossible by construction** — see the
"Why this design" section below.

Baseline AUC: **0.9091** (deterministic; reproducible by running `python evaluate.py`
after `python prepare_data.py`). With the bundled `instructions.md` and 200
steps of `gemini-3.1-pro-preview`, expect AUC in the **0.928-0.933** range.

## Layout

```
features.py     ← Weco edits this for Features-only scope.
                 Defines FeatureBuilder.fit(X_train, y_train) + transform(X).
model.py        ← Weco edits this for Model-only scope.
                 Defines train_and_evaluate(X_train, y_train, X_val, y_val).
evaluate.py     ← Frozen orchestrator. Loads data, calls fit/transform/train, prints AUC.
prepare_data.py ← One-off Kaggle download + parquet build. Run once.
instructions.md ← Domain knowledge prompt for Weco (--additional-instructions).
```

## Why this design

The original case study had `build_features(train_df, val_df)` in a single
function — the agent could `pd.concat([train, val])` and silently introduce
time-leakage. We measured the inflation at 0.001-0.005 AUC, and found that
even with explicit "fit on train only" warnings in the prompt, Weco's
proposals frequently reintroduced the leak.

This API kills both leakage flavors at the interface boundary:

| Leakage path | Killed by |
|---|---|
| `isFraud` in cross-column aggregations | `evaluate.py` strips `isFraud` before X reaches `FeatureBuilder` |
| `pd.concat([train_df, val_df])` for groupby/freq | `val_df` is never visible to `fit()` |
| Val labels at predict time | `transform(X)` has no `y` argument |

Weco can't write the leaky pattern because the leaky symbols literally aren't
in scope.

## Prerequisites

1. **Kaggle API token** at `~/.kaggle/kaggle.json` and
   `chmod 600 ~/.kaggle/kaggle.json`.
2. **Join the competition** at <https://www.kaggle.com/c/ieee-fraud-detection>
   (Late Submission / Join Competition). Without this, `prepare_data.py`
   gets a 403 from Kaggle.
3. **Weco API key** — see the [Weco docs](https://docs.weco.ai).

## Setup

```bash
cd examples/fraud-detection

# Virtualenv strongly recommended (PEP 668).
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade -r requirements.txt
# --upgrade is intentional: weco-cli ships fixes regularly. 0.3.31 added
# native auto-resume that fixes a transient submit-failure race that earlier
# versions hit.

# Download Kaggle data + build a 100K/25K time-based split. ~2-3 min.
python prepare_data.py

# Sanity check — should print auc_roc: 0.909132 deterministically.
python evaluate.py
```

## Run Weco

Three scope options. Pick one:

### Full pipeline (recommended)

```bash
weco run \
    --sources features.py model.py \
    --eval-command "python evaluate.py" \
    --metric auc_roc --goal maximize \
    --steps 200 \
    --model gemini-3.1-pro-preview \
    --additional-instructions instructions.md \
    --eval-timeout 900 \
    --log-dir .runs/full
```

Weco edits both files. Best AUC across seeds: ~0.929-0.933.

### Features only

```bash
weco run \
    --sources features.py \
    --eval-command "python evaluate.py" \
    --metric auc_roc --goal maximize \
    --steps 200 \
    --model gemini-3.1-pro-preview \
    --additional-instructions instructions.md \
    --eval-timeout 900 \
    --log-dir .runs/features
```

`model.py` stays at its baseline LightGBM. Weco can only improve features.

### Model only

```bash
weco run \
    --sources model.py \
    --eval-command "python evaluate.py" \
    --metric auc_roc --goal maximize \
    --steps 200 \
    --model gemini-3.1-pro-preview \
    --additional-instructions instructions.md \
    --eval-timeout 900 \
    --log-dir .runs/model
```

Features are frozen at the baseline `FeatureBuilder`. Weco can only improve
the model. Headroom is small (~+0.008 AUC) on this task — model tuning isn't
where the wins live for tabular fraud.

## Things to try

1. **No instructions** — drop `--additional-instructions`. Watch variance
   across seeds balloon (~3-5×). Watch for proposed code that leaks
   train/val statistics inside `fit()` even though the interface tries to
   prevent it (it's harder, but still possible if `fit` calls into shared
   helpers — the API keeps Weco honest at the boundary, not deep inside).
2. **EDA-only vs Tech-only** — split `instructions.md` into two prompts.
   The case study found EDA (column meanings) drives most of the gain;
   technique listings (UID construction, target encoding, etc.) are mostly
   already in the LLM's pretraining and add little.
3. **Disable auto-resume** — pass `--no-auto-resume` to see what transient
   failures look like without 0.3.31's recovery.

## Citing the case study

Numbers come from <https://github.com/WecoAI/fraud-detection-case-study>.
Setup: 200 steps, 3 seeds per condition, `gemini-3.1-pro-preview`. Strict-API
rerun on a clean leakage-safe baseline (this example).

## See also

- `examples/fraud-detection-loose/` — earlier single-file API (`train.py` with
  `build_features(train_df, val_df)`). Kept for comparison; not recommended
  for new work because it admits time-leakage.
