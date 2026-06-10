import argparse
import csv
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from optimize import extract_csv

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency
    plt = None

COST_ACCURACY_THRESHOLD_DEFAULT = 0.45
COST_CONSTRAINT_PENALTY = 1_000_000.0


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


def read_csv_table(path: Path) -> Tuple[List[str], List[List[str]]]:
    header: List[str] = []
    rows: List[List[str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            cleaned = [cell.strip() for cell in row]
            if not any(cleaned):
                continue
            if not header:
                header = cleaned
            else:
                rows.append(cleaned)
    return header, rows


def header_match_score(gt_header: List[str], pred_header: List[str]) -> float:
    if not gt_header or not pred_header:
        return 0.0
    normalized_gt = [cell.strip().lower() for cell in gt_header]
    normalized_pred = [cell.strip().lower() for cell in pred_header]
    return 1.0 if normalized_gt == normalized_pred else 0.0


def _to_float(cell: str) -> Optional[float]:
    cell = cell.strip()
    if not cell:
        return None
    normalized = cell.replace(",", "")
    if normalized.endswith("%"):
        normalized = normalized[:-1]
    if normalized.startswith("$"):
        normalized = normalized[1:]
    try:
        return float(normalized)
    except ValueError:
        return None


def _build_row_map(rows: List[List[str]]) -> Dict[str, List[List[str]]]:
    mapping: Dict[str, List[List[str]]] = {}
    for row in rows:
        if not row:
            continue
        key = row[0].strip().lower()
        mapping.setdefault(key, []).append(row[1:])
    return mapping


def _score_row(gt_values: List[str], pred_values: List[str]) -> float:
    if not gt_values:
        return 1.0

    per_column_scores: List[float] = []
    for idx, gt_cell in enumerate(gt_values):
        pred_cell = pred_values[idx] if idx < len(pred_values) else ""
        gt_num = _to_float(gt_cell)
        pred_num = _to_float(pred_cell)
        if gt_num is None or pred_num is None:
            per_column_scores.append(0.0)
            continue
        denom = abs(gt_num) + abs(pred_num)
        if denom == 0:
            per_column_scores.append(1.0)
            continue
        smape = 2.0 * abs(pred_num - gt_num) / denom
        per_column_scores.append(max(0.0, 1.0 - smape / 2.0))

    if not per_column_scores:
        return 0.0
    return sum(per_column_scores) / len(per_column_scores)


def visualize_difference(
    gt_csv_path: Path,
    pred_csv_path: Path,
    example_id: str,
    output_dir: Path,
    ignore_header_mismatch: bool = False,
    verbose: bool = False,
) -> Optional[Path]:
    def _viz_skip(reason: str) -> None:
        if verbose:
            print(f"[viz] skip {example_id}: {reason}", file=sys.stderr)

    if plt is None:
        _viz_skip("matplotlib not available")
        return None

    try:
        gt_header, gt_rows = read_csv_table(gt_csv_path)
        pred_header, pred_rows = read_csv_table(pred_csv_path)
    except Exception:
        _viz_skip("failed to read CSVs")
        return None

    if not gt_header or not pred_header:
        _viz_skip("missing headers")
        return None

    if header_match_score(gt_header, pred_header) < 1.0 and not ignore_header_mismatch:
        _viz_skip("header mismatch (use --visualize-allow-header-mismatch to override)")
        return None

    if len(gt_header) <= 1:
        _viz_skip("no data columns in GT header")
        return None

    columns = gt_header[1:]
    gt_series: Dict[str, List[float]] = {col: [] for col in columns}
    pred_series: Dict[str, List[float]] = {col: [] for col in columns}
    diff_series: Dict[str, List[float]] = {col: [] for col in columns}
    x_labels: List[str] = []

    pred_map = _build_row_map(pred_rows)
    pred_consumed: Dict[str, int] = {}

    for gt_row in gt_rows:
        if not gt_row:
            continue
        x_label = gt_row[0].strip()
        key = x_label.lower()
        pred_entries = pred_map.get(key, [])
        pred_idx = pred_consumed.get(key, 0)
        pred_values = pred_entries[pred_idx] if pred_idx < len(pred_entries) else []
        pred_consumed[key] = pred_idx + 1

        x_labels.append(x_label or f"row_{len(x_labels) + 1}")

        for col_idx, col_name in enumerate(columns):
            gt_val = _to_float(gt_row[col_idx + 1]) if col_idx + 1 < len(gt_row) else None
            pred_val = _to_float(pred_values[col_idx]) if col_idx < len(pred_values) else None
            gt_float = gt_val if gt_val is not None else math.nan
            pred_float = pred_val if pred_val is not None else math.nan

            if math.isnan(gt_float):
                # If GT is missing, treat as zero difference but keep nan for plotting gaps
                diff = math.nan
            elif math.isnan(pred_float):
                diff = math.nan
            else:
                diff = pred_float - gt_float

            gt_series[col_name].append(gt_float)
            pred_series[col_name].append(pred_float)
            diff_series[col_name].append(diff)

    if not x_labels:
        _viz_skip("no x labels in GT rows")
        return None

    num_series = len(columns)
    if num_series == 0:
        _viz_skip("no numeric series to plot")
        return None

    x_positions = list(range(len(x_labels)))
    fig_height = max(3.0, 3.0 * num_series)
    fig, axes = plt.subplots(num_series, 1, sharex=True, figsize=(11, fig_height))
    if num_series == 1:
        axes = [axes]

    for ax, col_name in zip(axes, columns):
        gt_values = gt_series[col_name]
        pred_values = pred_series[col_name]
        diff_values = diff_series[col_name]

        ax.plot(x_positions, gt_values, marker="o", linewidth=1.5, label="Ground Truth")
        ax.plot(x_positions, pred_values, marker="o", linewidth=1.5, label="Prediction")
        ax.set_ylabel(col_name)
        ax.grid(True, axis="y", alpha=0.3)

        has_diff = any(not math.isnan(v) for v in diff_values)
        legend_handles, legend_labels = ax.get_legend_handles_labels()

        if has_diff:
            ax2 = ax.twinx()
            ax2.plot(x_positions, diff_values, linestyle="--", color="tab:red", marker="x", linewidth=1.2, label="Pred - GT")
            ax2.axhline(0.0, color="tab:red", linewidth=0.8, alpha=0.4)
            ax2.set_ylabel("Pred - GT")
            handles2, labels2 = ax2.get_legend_handles_labels()
            legend_handles += handles2
            legend_labels += labels2

        if legend_handles:
            ax.legend(legend_handles, legend_labels, loc="upper left")

    axes[-1].set_xticks(x_positions)
    axes[-1].set_xticklabels(x_labels, rotation=45, ha="right")
    axes[-1].set_xlabel(gt_header[0])

    fig.suptitle(f"{example_id}: Ground Truth vs Prediction", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{example_id}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def evaluate_predictions(gt_csv_path: Path, pred_csv_path: Path) -> float:
    gt_header, gt_rows = read_csv_table(gt_csv_path)
    pred_header, pred_rows = read_csv_table(pred_csv_path)
    if not gt_header or not pred_header:
        return 0.0

    header_score = header_match_score(gt_header, pred_header)

    gt_map = _build_row_map(gt_rows)
    pred_map = _build_row_map(pred_rows)

    row_scores: List[float] = []
    for key, gt_entries in gt_map.items():
        pred_entries = pred_map.get(key, [])
        for idx, gt_values in enumerate(gt_entries):
            pred_values = pred_entries[idx] if idx < len(pred_entries) else []
            row_scores.append(_score_row(gt_values, pred_values))

    content_score = sum(row_scores) / len(row_scores) if row_scores else 0.0
    return 0.2 * header_score + 0.8 * content_score


def process_one(
    base_dir: Path, example_id: str, image_rel: Path, gt_table_rel: Path, output_dir: Path
) -> Tuple[str, float, Path, Path, float]:
    image_path = base_dir / image_rel
    gt_csv_path = base_dir / gt_table_rel
    pred_csv_text, cost_usd = extract_csv(image_path)
    pred_path = write_csv(output_dir, example_id, pred_csv_text)
    score = evaluate_predictions(gt_csv_path, pred_path)
    return example_id, score, pred_path, gt_csv_path, cost_usd


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VLM extraction: image -> CSV")
    parser.add_argument("--data-dir", type=str, default="subset_line_100")
    parser.add_argument("--index", type=str, default="index.csv")
    parser.add_argument("--out-dir", type=str, default="predictions")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--cost-metric",
        action="store_true",
        help=(
            "When set, also report a `cost:` metric suitable for Weco minimization. "
            "Requires final accuracy to exceed --cost-accuracy-threshold; otherwise a large penalty is reported."
        ),
    )
    parser.add_argument(
        "--cost-accuracy-threshold",
        type=float,
        default=COST_ACCURACY_THRESHOLD_DEFAULT,
        help="Minimum accuracy required when --cost-metric is set (default: 0.45).",
    )
    parser.add_argument(
        "--visualize-dir",
        type=str,
        default=None,
        help="Directory where GT vs prediction plots will be saved (requires matplotlib).",
    )
    parser.add_argument(
        "--visualize-max",
        type=int,
        default=1,
        help="Maximum number of plots to generate when --visualize-dir is set. Use 0 for no limit.",
    )
    parser.add_argument(
        "--visualize-allow-header-mismatch",
        action="store_true",
        help="If set, still plot GT vs prediction even when headers differ.",
    )
    parser.add_argument("--visualize-verbose", action="store_true", help="Print reasons when a visualization is skipped.")
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

    visualize_dir: Optional[Path] = Path(args.visualize_dir) if args.visualize_dir else None
    visualize_max = max(0, args.visualize_max)
    if visualize_dir and plt is None:
        print("[warn] matplotlib not available; skipping visualization.", file=sys.stderr)
        visualize_dir = None

    print(f"[setup] evaluating {len(rows)} samples …", flush=True)
    start = time.time()
    scores: List[float] = []
    costs: List[float] = []
    saved_visualizations = 0

    with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as pool:
        futures = [
            pool.submit(process_one, base_dir, example_id, image_rel, gt_table_rel, Path(args.out_dir))
            for (example_id, image_rel, gt_table_rel) in rows
        ]

        try:
            for idx, fut in enumerate(as_completed(futures), 1):
                try:
                    example_id, score, pred_path, gt_csv_path, cost_usd = fut.result()
                    scores.append(score)
                    costs.append(cost_usd)
                    if visualize_dir and (visualize_max == 0 or saved_visualizations < visualize_max):
                        out_path = visualize_difference(
                            gt_csv_path,
                            pred_path,
                            example_id,
                            visualize_dir,
                            ignore_header_mismatch=args.visualize_allow_header_mismatch,
                            verbose=args.visualize_verbose,
                        )
                        if out_path is not None:
                            saved_visualizations += 1
                            print(f"[viz] saved {out_path}", flush=True)
                    if idx % 5 == 0 or idx == len(rows):
                        elapsed = time.time() - start
                        avg = sum(scores) / len(scores) if scores else 0.0
                        avg_cost = sum(costs) / len(costs) if costs else 0.0
                        print(
                            f"[progress] {idx}/{len(rows)} done, avg score: {avg:.4f}, avg cost: ${avg_cost:.4f}, elapsed {elapsed:.1f}s",
                            flush=True,
                        )
                except Exception as e:
                    print(f"[error] failed on sample {idx}: {e}", file=sys.stderr)
        except KeyboardInterrupt:
            print("\n[warn] interrupted by user", file=sys.stderr)
            sys.exit(1)

    final_score = sum(scores) / len(scores) if scores else 0.0

    # Apply cost cap: accuracy is zeroed if average cost/query exceeds $0.02
    avg_cost_per_query = (sum(costs) / len(costs)) if costs else 0.0
    if avg_cost_per_query > 0.02:
        print(f"[cost] avg ${avg_cost_per_query:.4f}/query exceeds $0.02 cap; accuracy set to 0.0", flush=True)
        final_score = 0.0
    else:
        print(f"[cost] avg ${avg_cost_per_query:.4f}/query within cap", flush=True)

    print(f"accuracy: {final_score:.4f}")

    if args.cost_metric:
        if final_score > args.cost_accuracy_threshold:
            reported_cost = avg_cost_per_query
        else:
            print(
                (
                    f"[constraint] accuracy {final_score:.4f} <= "
                    f"threshold {args.cost_accuracy_threshold:.2f}; reporting penalty ${COST_CONSTRAINT_PENALTY:.1f}"
                ),
                flush=True,
            )
            reported_cost = COST_CONSTRAINT_PENALTY
        print(f"cost: {reported_cost:.6f}")


if __name__ == "__main__":
    main()
