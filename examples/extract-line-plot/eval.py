import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

from optimize import VLMExtractor


def read_index(index_csv_path: Path) -> List[Tuple[str, Path, Path]]:
    rows: List[Tuple[str, Path, Path]] = []
    with open(index_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["id"].strip(), Path(row["image"].strip()), Path(row["table"].strip())))
    return rows


def write_csv(output_dir: Path, example_id: str, csv_text: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{example_id}.csv"
    out_path.write_text(csv_text, encoding="utf-8")
    return out_path


def normalize_csv_rows(path: Path) -> List[str]:
    normalized: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            cells = [c.strip() for c in row]
            normalized.append(",".join(cells))
    return normalized


def jaccard_row_similarity(gt_rows: List[str], pred_rows: List[str]) -> float:
    gt_set = {r.lower() for r in gt_rows}
    pred_set = {r.lower() for r in pred_rows}
    if not gt_set and not pred_set:
        return 1.0
    if not gt_set or not pred_set:
        return 0.0
    inter = len(gt_set & pred_set)
    union = len(gt_set | pred_set)
    return inter / union if union > 0 else 0.0


def evaluate_predictions(gt_csv_path: Path, pred_csv_path: Path) -> float:
    gt_rows = normalize_csv_rows(gt_csv_path)
    pred_rows = normalize_csv_rows(pred_csv_path)
    if not gt_rows or not pred_rows:
        return 0.0
    header_score = 1.0 if gt_rows[0].lower() == pred_rows[0].lower() else 0.0
    content_score = jaccard_row_similarity(gt_rows[1:], pred_rows[1:]) if len(gt_rows) > 1 else 0.0
    return 0.2 * header_score + 0.8 * content_score


def process_one(
    extractor: VLMExtractor,
    base_dir: Path,
    example_id: str,
    image_rel: Path,
    gt_table_rel: Path,
    output_dir: Path,
) -> Tuple[str, float, Path]:
    image_path = base_dir / image_rel
    gt_csv_path = base_dir / gt_table_rel
    pred_csv_text = extractor.image_to_csv(image_path)
    pred_path = write_csv(output_dir, example_id, pred_csv_text)
    score = evaluate_predictions(gt_csv_path, pred_path)
    return example_id, score, pred_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VLM extraction: image -> CSV")
    parser.add_argument("--data-dir", type=str, default="subset_line_100")
    parser.add_argument("--index", type=str, default="index.csv")
    parser.add_argument("--out-dir", type=str, default="predictions")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("[error] OPENAI_API_KEY not set in environment", file=sys.stderr)
        sys.exit(1)

    base_dir = Path(args.data_dir)
    index_path = base_dir / args.index
    if not index_path.exists():
        print(f"[error] index.csv not found at {index_path}", file=sys.stderr)
        sys.exit(1)

    rows = read_index(index_path)[: args.max_samples]
    extractor = VLMExtractor()

    print(f"[setup] evaluating {len(rows)} samples using {extractor.model} …", flush=True)
    start = time.time()
    scores: List[float] = []

    with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as pool:
        futures = [
            pool.submit(
                process_one,
                extractor,
                base_dir,
                example_id,
                image_rel,
                gt_table_rel,
                Path(args.out_dir),
            )
            for (example_id, image_rel, gt_table_rel) in rows
        ]

        try:
            for idx, fut in enumerate(as_completed(futures), 1):
                try:
                    example_id, score, pred_path = fut.result()
                    scores.append(score)
                    if idx % 5 == 0 or idx == len(rows):
                        elapsed = time.time() - start
                        avg = sum(scores) / len(scores) if scores else 0.0
                        print(
                            f"[progress] {idx}/{len(rows)} done, avg score: {avg:.4f}, elapsed {elapsed:.1f}s",
                            flush=True,
                        )
                except Exception as e:
                    print(f"[error] failed on sample {idx}: {e}", file=sys.stderr)
        except KeyboardInterrupt:
            print("\n[warn] interrupted by user", file=sys.stderr)
            sys.exit(1)

    final_score = sum(scores) / len(scores) if scores else 0.0

    # Apply cost cap: accuracy is zeroed if average cost/query exceeds $0.01
    avg_cost_per_query = (
        (extractor.total_cost_usd / extractor.num_queries) if getattr(extractor, "num_queries", 0) else 0.0
    )
    if avg_cost_per_query > 0.01:
        print(
            f"[cost] avg ${avg_cost_per_query:.4f}/query exceeds $0.01 cap; accuracy set to 0.0",
            flush=True,
        )
        final_score = 0.0
    else:
        print(f"[cost] avg ${avg_cost_per_query:.4f}/query within cap", flush=True)

    print(f"accuracy: {final_score:.4f}")


if __name__ == "__main__":
    main()


