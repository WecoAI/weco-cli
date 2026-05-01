"""Measure the baseline accuracy of the bloated SYSTEM_PROMPT and freeze it.

Run once before kicking off Weco:

    python measure_baseline.py

Output: ``baseline_accuracy.json`` next to this script. ``eval.py`` reads
that file to set the accuracy floor that compressed prompts must respect.
"""

from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from labels import LABELS, parse_predicted_label  # noqa: E402
import optimize  # noqa: E402

EVAL_MODEL = os.environ.get("EVAL_MODEL", "gpt-5-mini")
N_SAMPLES = int(os.environ.get("EVAL_SAMPLES", "200"))
N_WORKERS = int(os.environ.get("EVAL_WORKERS", "20"))
SEED = int(os.environ.get("EVAL_SEED", "0"))


def main() -> int:
    import concurrent.futures
    from datasets import load_dataset

    t0 = time.time()
    print(f"[baseline] model={EVAL_MODEL} samples={N_SAMPLES} seed={SEED}")
    print(f"[baseline] SYSTEM_PROMPT chars={len(optimize.SYSTEM_PROMPT):,}")

    ds = load_dataset("PolyAI/banking77", split="test", trust_remote_code=True).shuffle(seed=SEED)
    ds = ds.select(range(min(N_SAMPLES, len(ds))))
    items = [(ex["text"], LABELS[ex["label"]]) for ex in ds]

    def run_one(item):
        text, gold = item
        raw = optimize.classify(text, model=EVAL_MODEL)
        return parse_predicted_label(raw), gold

    correct = 0
    parsed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        for i, (pred, gold) in enumerate(ex.map(run_one, items), start=1):
            if pred is not None:
                parsed += 1
                if pred == gold:
                    correct += 1
            if i % max(1, len(items) // 10) == 0:
                print(
                    f"[baseline] {i}/{len(items)}  acc={correct / i:.4f}  "
                    f"parse={parsed / i:.4f}  elapsed={time.time() - t0:.1f}s"
                )

    accuracy = correct / len(items)
    out = {
        "accuracy": accuracy,
        "n_samples": len(items),
        "model": EVAL_MODEL,
        "seed": SEED,
        "chars": len(optimize.SYSTEM_PROMPT),
        "parse_rate": parsed / len(items),
        "elapsed_seconds": time.time() - t0,
    }
    out_path = ROOT / "baseline_accuracy.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\n[baseline] wrote {out_path}")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
