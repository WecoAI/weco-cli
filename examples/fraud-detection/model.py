"""Model training and evaluation — Weco optimizes this file for the Model
scope. Features arrive pre-built; labels arrive separately.

Interface:
- train_and_evaluate(X_train, y_train, X_val, y_val) -> float (val AUC)
- X_* are float32 ndarrays of identical shape; y_* are pd.Series of int32 labels.
- Return validation AUC-ROC. Print a final `auc_roc: 0.xxxxxx` line in
  evaluate.py (this file just returns the float).

What you CAN do:
- Tune LightGBM hyperparameters, boosting strategy, num_iterations, etc.
- Switch model class (xgboost, catboost, sklearn ensemble, custom torch model).
- Build ensembles, stacking, blending.
- Modify class-imbalance handling, custom objectives.

What you CANNOT do:
- See the feature column names (already projected to ndarray).
- Re-engineer features here — features.py owns that scope.
- Peek at val labels at training time (they're a separate argument; use them
  only inside the AUC computation at the end).
"""

from __future__ import annotations

import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    """Train a model on (X_train, y_train); return AUC-ROC on (X_val, y_val).

    Reasonable-but-not-heavily-tuned LightGBM defaults. There is real headroom
    here — class imbalance, regularization, deeper trees, more rounds, ensembles.
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
