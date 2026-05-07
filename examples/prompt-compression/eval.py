"""Evaluation script Weco invokes after each prompt mutation.

Loads a held-out slice of PolyAI/banking77, runs ``optimize.classify`` on each
example in parallel, parses predicted labels, and emits three lines on stdout
that Weco can read:

    accuracy: <0..1>
    chars:    <int>
    metric:   <int>     # the value Weco minimizes

The composite ``metric`` is **quality-constrained minimize chars**:
    metric = chars                if accuracy >= ACCURACY_THRESHOLD
    metric = chars + huge penalty otherwise

This gives a clean narrative ("we kept accuracy at or above the baseline and
got X% smaller") and makes Weco's optimization target unambiguous.
"""

from __future__ import annotations
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from labels import LABELS, parse_predicted_label  # noqa: E402
import optimize  # noqa: E402  -- the file Weco mutates

# --- configuration ---------------------------------------------------------

EVAL_MODEL = os.environ.get("EVAL_MODEL", "gpt-5-mini")
N_SAMPLES = int(os.environ.get("EVAL_SAMPLES", "200"))
N_WORKERS = int(os.environ.get("EVAL_WORKERS", "20"))
SEED = int(os.environ.get("EVAL_SEED", "0"))

# Accuracy threshold the compressed prompt must hold above. Loaded from
# baseline_accuracy.json if present (written by measure_baseline.py); falls
# back to a conservative default.
_BASELINE_FILE = ROOT / "baseline_accuracy.json"
if _BASELINE_FILE.exists():
    _baseline = json.loads(_BASELINE_FILE.read_text())
    BASELINE_ACC = float(_baseline["accuracy"])
    SLACK = float(os.environ.get("EVAL_SLACK", "0.02"))
    ACCURACY_THRESHOLD = max(0.0, BASELINE_ACC - SLACK)
else:
    BASELINE_ACC = None
    ACCURACY_THRESHOLD = float(os.environ.get("EVAL_THRESHOLD", "0.75"))

PENALTY = 10_000_000  # added to chars when accuracy is below threshold

# --- main ------------------------------------------------------------------


def _load_eval_set():
    """Return a list of (text, gold_label) tuples sampled from the test split."""
    from datasets import load_dataset  # heavy import — lazy

    ds = load_dataset("PolyAI/banking77", split="test", trust_remote_code=True)
    ds = ds.shuffle(seed=SEED).select(range(min(N_SAMPLES, len(ds))))
    return [(ex["text"], LABELS[ex["label"]]) for ex in ds]


def _classify_one(item):
    text, gold = item
    raw = optimize.classify(text, model=EVAL_MODEL)
    pred = parse_predicted_label(raw)
    return text, gold, pred, raw


def main() -> int:
    t0 = time.time()
    print(f"[setup] loading {N_SAMPLES} BANKING77 test samples (seed={SEED})", file=sys.stderr)
    eval_set = _load_eval_set()
    n = len(eval_set)
    print(f"[setup] running {n} classifications via {EVAL_MODEL} ({N_WORKERS} workers)", file=sys.stderr)

    correct = 0
    parsed = 0
    log_every = max(1, n // 5)
    with concurrent.futures.ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        for i, (text, gold, pred, _raw) in enumerate(ex.map(_classify_one, eval_set), start=1):
            if pred is not None:
                parsed += 1
                if pred == gold:
                    correct += 1
            if i % log_every == 0 or i == n:
                running = correct / i
                elapsed = time.time() - t0
                print(
                    f"[progress] {i}/{n} completed, accuracy: {running:.4f}, "
                    f"parse-rate: {parsed / i:.4f}, elapsed {elapsed:.1f}s",
                    file=sys.stderr,
                )

    accuracy = correct / n
    chars = len(optimize.SYSTEM_PROMPT)
    if accuracy >= ACCURACY_THRESHOLD:
        metric = chars
    else:
        metric = chars + PENALTY

    print(f"accuracy: {accuracy:.4f}")
    print(f"chars: {chars}")
    print(f"metric: {metric}")

    msg = (
        f"[summary] accuracy={accuracy:.4f}  threshold={ACCURACY_THRESHOLD:.4f}  "
        f"chars={chars:,}  metric={metric:,}  "
        f"parse_rate={parsed / n:.4f}  elapsed={time.time() - t0:.1f}s"
    )
    print(msg, file=sys.stderr)
    if BASELINE_ACC is not None:
        print(f"[summary] baseline_accuracy={BASELINE_ACC:.4f} (from baseline_accuracy.json)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
