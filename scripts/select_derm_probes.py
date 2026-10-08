"""
Select 50 stratified dermoscopy images for MedTEC-Bench probing.

Reads ISIC 2018 Task 3 ground truth CSV (HAM10000) for diagnosis labels.
Requires images from ISIC 2018 Task 1-2 Training Input (the same image set).

Output:
  data/derm_probes/images/   -- copied probe images (flat, renamed by category)
  data/derm_probes/manifest.json  -- per-image metadata + disease category

Dataset structure expected under --data-dir:
  <data-dir>/
    ISIC2018_Task3_Training_GroundTruth.csv   ← diagnosis labels
    images/
      ISIC2018_Task3_Training_Input/          ← images
        ISIC_0024306.jpg
        ...

Download:
  Task 3 CSV:   https://challenge.isic-archive.com/data/#2018  (Task 3 training labels)
  Task 1/2 imgs: same page (Task 1 training data — already present in data/isic/)

Usage:
  python scripts/select_derm_probes.py --data-dir data/isic
  python scripts/select_derm_probes.py --data-dir data/isic --n-per-category 10 --out-dir data/derm_probes
"""

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# HAM10000 / ISIC 2018 Task 3 category definitions
# ---------------------------------------------------------------------------

# Maps our internal key → CSV column name (one-hot columns in the CSV)
DERM_CATEGORIES = {
    "mel":   "MEL",    # Melanoma
    "nv":    "NV",     # Melanocytic Nevus
    "bcc":   "BCC",    # Basal Cell Carcinoma
    "akiec": "AKIEC",  # Actinic Keratosis / Intraepithelial Carcinoma
    "bkl":   "BKL",    # Benign Keratosis (seborrheic keratosis, solar lentigo, LPLK)
}

CATEGORY_DISPLAY = {
    "mel":   "Melanoma",
    "nv":    "Melanocytic Nevus",
    "bcc":   "Basal Cell Carcinoma",
    "akiec": "Actinic Keratosis / Bowen's",
    "bkl":   "Benign Keratosis",
}

# Image sub-directory name within data-dir
IMAGE_SUBDIR = "ISIC2018_Task3_Training_Input"

# CSV filename
TASK3_CSV = "ISIC2018_Task3_Training_GroundTruth.csv"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def find_image(data_dir: Path, image_name: str) -> str | None:
    """Resolve an image filename to an absolute path.

    Tries the standard Task 1/2 sub-directory first, then any sub-directory
    named 'images', then falls back to a full recursive search so alternate
    download layouts work.
    """
    # Primary: images/<subdir>/
    primary = data_dir / "images" / IMAGE_SUBDIR / image_name
    if primary.exists():
        return str(primary)

    # Secondary: direct subdir at data-dir root (no 'images' wrapper)
    secondary = data_dir / IMAGE_SUBDIR / image_name
    if secondary.exists():
        return str(secondary)

    # Fallback: search anywhere under data_dir/images/
    images_root = data_dir / "images"
    if images_root.exists():
        for candidate in images_root.rglob(image_name):
            return str(candidate)

    # Last resort: full recursive search
    for candidate in data_dir.rglob(image_name):
        return str(candidate)
    return None


def load_ham10000(data_dir: Path) -> pd.DataFrame:
    """Load ISIC 2018 Task 3 labels CSV and attach resolved image paths."""
    csv_path = data_dir / TASK3_CSV
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Task 3 labels CSV not found: {csv_path}\n"
            f"Download from https://challenge.isic-archive.com/data/#2018 "
            f"(Task 3 Training Ground Truth) and place it in {data_dir}"
        )
    df = pd.read_csv(csv_path)

    # Normalise column name: CSV may use 'image' or 'image_id'
    id_col = None
    for c in ("image", "image_id", "Image", "Image Index"):
        if c in df.columns:
            id_col = c
            break
    if id_col is None:
        raise ValueError(
            f"Cannot find image ID column in {csv_path}. "
            f"Columns found: {list(df.columns)}"
        )
    if id_col != "image":
        df = df.rename(columns={id_col: "image"})

    return df


def select_stratified(data_dir: Path, n: int, seed: int) -> list:
    """Select n images per category from HAM10000, returning metadata records."""
    df = load_ham10000(data_dir)
    records = []

    for category, col in DERM_CATEGORIES.items():
        if col not in df.columns:
            print(f"  [WARN] Column '{col}' not in CSV — skipping {category}")
            continue

        # Positive examples: label == 1.0
        subset = df[df[col] == 1.0].copy()

        if len(subset) < n:
            print(
                f"  [WARN] Only {len(subset)} positive '{col}' images — "
                f"using all of them (wanted {n})"
            )
            chosen = subset
        else:
            chosen = subset.sample(n=n, random_state=seed)

        for _, row in chosen.iterrows():
            # Try both .jpg and .png extensions
            img_name = None
            for ext in (".jpg", ".jpeg", ".png"):
                name = row["image"] + ext
                if find_image(data_dir, name):
                    img_name = name
                    break

            if img_name is None:
                print(f"  [WARN] No image file found for {row['image']} — skipping")
                continue

            abs_path = find_image(data_dir, img_name)

            # Collect all category labels for this image
            all_labels = {
                v: (float(row[v]) if v in df.columns and pd.notna(row[v]) else None)
                for v in DERM_CATEGORIES.values()
            }

            records.append({
                "image_id": row["image"],
                "original_path": abs_path,
                "disease_category": category,
                "disease_display": CATEGORY_DISPLAY[category],
                "disease_labels": all_labels,
                "source": "isic2018_task3",
            })

    return records


# ---------------------------------------------------------------------------
# Copy images + write manifest
# ---------------------------------------------------------------------------

def copy_probes(records: list, out_dir: Path, repo_root: Path) -> list:
    """Copy selected images into out_dir/images/ and embed relative paths.

    image_path stored as POSIX path relative to repo_root so that
    REPO_ROOT / record['image_path'] resolves at eval time.
    """
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    updated = []
    seen_names: dict = {}

    for rec in records:
        src = Path(rec["original_path"])
        key = f"{rec['disease_category']}_{src.stem}"

        if key in seen_names:
            seen_names[key] += 1
            dest_name = f"{key}_{seen_names[key]}{src.suffix}"
        else:
            seen_names[key] = 0
            dest_name = f"{key}{src.suffix}"

        dest = img_dir / dest_name
        shutil.copy2(src, dest)

        rec = dict(rec)
        rec["image_path"] = dest.relative_to(repo_root).as_posix()
        rec["filename"] = dest_name
        updated.append(rec)

    return updated


def write_manifest(records: list, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nManifest written -> {manifest_path}")
    return manifest_path


def print_summary(records: list):
    counts = Counter(r["disease_category"] for r in records)
    print("\n--- Selection summary ---")
    for cat, n in sorted(counts.items()):
        display = CATEGORY_DISPLAY.get(cat, cat)
        print(f"  {cat:<8}  {display:<35}: {n} images")
    print(f"  {'TOTAL':<8}  {'':35}: {len(records)} images")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Select 50 stratified dermoscopy probe images from ISIC 2018 Task 3"
    )
    parser.add_argument(
        "--data-dir", required=True,
        help=(
            "Root of ISIC 2018 data — should contain "
            f"'{TASK3_CSV}' and '{IMAGE_SUBDIR}/'"
        ),
    )
    parser.add_argument(
        "--out-dir", default="data/derm_probes",
        help="Output directory for probe images + manifest (default: data/derm_probes)",
    )
    parser.add_argument(
        "--n-per-category", type=int, default=10,
        help="Images to select per disease category (default: 10)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print selection summary without copying files",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = repo_root / data_dir
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir

    if not data_dir.exists():
        print(f"ERROR: data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Data dir  : {data_dir}")
    print(f"Output dir: {out_dir}")
    print(f"n per cat : {args.n_per_category}, seed: {args.seed}")

    records = select_stratified(data_dir, args.n_per_category, args.seed)
    print_summary(records)

    if not records:
        print("\nERROR: No records selected — check CSV path and image files.", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print("\n[dry-run] No files copied.")
        return

    records = copy_probes(records, out_dir, repo_root)
    write_manifest(records, out_dir)
    print(f"\nDone. {len(records)} probe images in {out_dir / 'images'}")


if __name__ == "__main__":
    main()
