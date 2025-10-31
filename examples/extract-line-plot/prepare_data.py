import argparse
import json
import random
import shutil
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download


def collect_line_charts(root: Path, split: str):
    ann_dir = root / "ChartQA Dataset" / split / "annotations"
    png_dir = root / "ChartQA Dataset" / split / "png"
    tbl_dir = root / "ChartQA Dataset" / split / "tables"

    items = []
    for ann_path in ann_dir.glob("*.json"):
        try:
            meta = json.loads(ann_path.read_text(encoding="utf-8"))
            if meta.get("type") != "line":
                continue
            stem = ann_path.stem
            img = png_dir / f"{stem}.png"
            csv = tbl_dir / f"{stem}.csv"
            if img.exists() and csv.exists():
                items.append((stem, img, csv))
        except Exception:
            continue
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="subset_line_100.zip")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("Downloading ChartQA (full) snapshot from Hugging Face…")
    # Repo linked from ChartQA README (“Full version with annotations”)
    local_dir = snapshot_download(repo_id="ahmed-masry/ChartQA", repo_type="dataset")
    root = Path(local_dir)

    # The dataset snapshot contains a large zip named "ChartQA Dataset.zip".
    # If it hasn't been extracted yet, extract it so that expected folders exist.
    target_dir = root / "ChartQA Dataset"
    if not target_dir.exists():
        zip_candidates = list(root.glob("**/ChartQA Dataset.zip"))
        if zip_candidates:
            zip_path = zip_candidates[0]
            print(f"Extracting '{zip_path.name}'… This may take a minute.")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(root)
        else:
            raise SystemExit(
                "Could not find 'ChartQA Dataset.zip' in the downloaded snapshot; dataset layout may have changed."
            )

    print(f"Scanning {args.split} annotations for line charts…")
    pool = collect_line_charts(root, args.split)
    if len(pool) < args.n:
        raise SystemExit(f"Only found {len(pool)} line charts in {args.split}, less than requested {args.n}")

    random.Random(args.seed).shuffle(pool)
    chosen = pool[: args.n]

    work = Path("subset_line_100")
    if work.exists():
        shutil.rmtree(work)
    (work / "images").mkdir(parents=True)
    (work / "tables").mkdir(parents=True)

    with open(work / "index.csv", "w", encoding="utf-8") as f:
        f.write("id,image,table\n")
        for stem, img, csv in chosen:
            dst_img = work / "images" / f"{stem}.png"
            dst_csv = work / "tables" / f"{stem}.csv"
            shutil.copy2(img, dst_img)
            shutil.copy2(csv, dst_csv)
            f.write(f"{stem},images/{stem}.png,tables/{stem}.csv\n")

    # Zip it
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in work.rglob("*"):
            z.write(p, p.relative_to(work.parent))

    print(f"Done: wrote {args.out}")
    print("Preview first 5 rows of index.csv:")
    print(*(open(work / "index.csv", "r", encoding="utf-8").read().splitlines()[:6]), sep="\n")


if __name__ == "__main__":
    main()
