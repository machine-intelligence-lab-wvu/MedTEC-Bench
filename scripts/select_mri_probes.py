"""
Select stratified brain MRI images for MedTEC-Bench probing.

Source: OASIS Alzheimer's MRI dataset
(Kaggle: 'ninadaithal/imagesoasis') — JPG slices organized by
dementia stage folders under <data-dir>/Data/.

Expected layout:
  <data-dir>/
    Data/
      Non Demented/        *.jpg
      Very mild Dementia/  *.jpg
      Mild Dementia/       *.jpg
      Moderate Dementia/   *.jpg

Output:
  data/mri_probes/images/       — copied probe images (flat, renamed by category)
  data/mri_probes/manifest.json — per-image metadata + disease category

Usage:
  python scripts/select_mri_probes.py --data-dir data/oasis
  python scripts/select_mri_probes.py --data-dir data/oasis --n-per-category 12 --out-dir data/mri_probes
"""

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------------------
# OASIS dementia-stage folder mapping
# ---------------------------------------------------------------------------

# internal_key -> folder_name under data-dir/Data/
MRI_CATEGORIES = {
    "non_demented":        "Non Demented",
    "very_mild_dementia":  "Very mild Dementia",
    "mild_dementia":       "Mild Dementia",
    "moderate_dementia":   "Moderate Dementia",
}

CATEGORY_DISPLAY = {
    "non_demented":        "Non Demented",
    "very_mild_dementia":  "Very Mild Dementia",
    "mild_dementia":       "Mild Dementia",
    "moderate_dementia":   "Moderate Dementia",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def list_category_images(data_dir: Path, folder_name: str) -> list:
    """List all image files in data_dir/Data/<folder_name>/."""
    cat_dir = data_dir / "Data" / folder_name
    if not cat_dir.exists():
        # Fallback: no "Data" wrapper
        cat_dir = data_dir / folder_name
    if not cat_dir.exists():
        return []
    return sorted(p for p in cat_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def select_stratified(data_dir: Path, n: int, seed: int) -> list:
    """Select n images per dementia category and return metadata records."""
    rng = random.Random(seed)
    records = []

    for category, folder in MRI_CATEGORIES.items():
        imgs = list_category_images(data_dir, folder)
        if not imgs:
            print(f"  [WARN] No images found for '{folder}' — skipping {category}")
            continue

        if len(imgs) < n:
            print(
                f"  [WARN] Only {len(imgs)} images in '{folder}' — "
                f"using all (wanted {n})"
            )
            chosen = imgs
        else:
            chosen = rng.sample(imgs, n)

        for img_path in chosen:
            records.append({
                "image_id": img_path.stem,
                "original_path": str(img_path),
                "disease_category": category,
                "disease_display": CATEGORY_DISPLAY[category],
                "source": "oasis_alzheimers",
            })

    return records


# ---------------------------------------------------------------------------
# Copy + manifest
# ---------------------------------------------------------------------------

def copy_probes(records: list, out_dir: Path, repo_root: Path) -> list:
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
        print(f"  {cat:<22}  {display:<25}: {n} images")
    print(f"  {'TOTAL':<22}  {'':25}: {len(records)} images")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Select stratified brain MRI probe images from OASIS Alzheimer's"
    )
    parser.add_argument(
        "--data-dir", required=True,
        help=("Root of OASIS data — should contain 'Data/<category>/*.jpg'"),
    )
    parser.add_argument(
        "--out-dir", default="data/mri_probes",
        help="Output directory (default: data/mri_probes)",
    )
    parser.add_argument(
        "--n-per-category", type=int, default=12,
        help="Images per dementia stage (default: 12)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print summary without copying files",
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
        print("\nERROR: No records selected — check data layout.", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print("\n[dry-run] No files copied.")
        return

    records = copy_probes(records, out_dir, repo_root)
    write_manifest(records, out_dir)
    print(f"\nDone. {len(records)} probe images in {out_dir / 'images'}")


if __name__ == "__main__":
    main()
