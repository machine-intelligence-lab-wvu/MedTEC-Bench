"""
Select 50 stratified chest X-ray images for MedTEC-Bench probing.

Supports both CheXpert and NIH ChestX-ray14 formats (auto-detected).

Output:
  data/cxr_probes/images/   -- copied probe images (flat, renamed by category)
  data/cxr_probes/manifest.json  -- per-image metadata + disease category

Usage:
  python scripts/select_cxr_probes.py --data-dir data/nih_xray
  python scripts/select_cxr_probes.py --data-dir data/chexpert --split train
  python scripts/select_cxr_probes.py --data-dir data/nih_xray --n-per-category 10 --out-dir data/cxr_probes
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Dataset format detection
# ---------------------------------------------------------------------------

def detect_format(data_dir: Path):
    """Return 'chexpert' or 'nih' based on which CSV is present."""
    if (data_dir / "train.csv").exists() or (data_dir / "valid.csv").exists():
        return "chexpert"
    if (data_dir / "Data_Entry_2017.csv").exists():
        return "nih"
    raise FileNotFoundError(
        f"No recognised CSV found in {data_dir}.\n"
        "Expected 'train.csv' / 'valid.csv' (CheXpert) or 'Data_Entry_2017.csv' (NIH)."
    )


# ---------------------------------------------------------------------------
# CheXpert loader
# ---------------------------------------------------------------------------

CHEXPERT_CATEGORIES = {
    "no_finding":      "No Finding",
    "pneumonia":       "Pneumonia",
    "cardiomegaly":    "Cardiomegaly",
    "pleural_effusion":"Pleural Effusion",
    "atelectasis":     "Atelectasis",
}

def load_chexpert(data_dir: Path, split: str = "train") -> pd.DataFrame:
    csv_path = data_dir / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CheXpert CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)

    # Keep frontal views only
    df = df[df["Frontal/Lateral"] == "Frontal"].copy()

    # Drop rows where the Path column doesn't resolve to an existing file.
    # CheXpert paths are relative to the parent of data_dir (CheXpert-v1.0 structure)
    # or relative to data_dir itself — try both.
    def resolve(p):
        for base in (data_dir, data_dir.parent):
            full = base / p
            if full.exists():
                return str(full)
        return None

    df["abs_path"] = df["Path"].apply(resolve)
    df = df[df["abs_path"].notna()].copy()
    return df


def select_chexpert(data_dir: Path, n: int, split: str, seed: int) -> list:
    df = load_chexpert(data_dir, split)
    rng = random.Random(seed)
    records = []

    for category, col in CHEXPERT_CATEGORIES.items():
        if col not in df.columns:
            print(f"  [WARN] Column '{col}' not in CheXpert CSV — skipping {category}")
            continue

        # Certain positives only (label == 1.0); NaN / -1 excluded
        mask = df[col] == 1.0
        subset = df[mask].copy()

        if len(subset) < n:
            print(f"  [WARN] Only {len(subset)} certain-positive '{col}' images — "
                  f"using all of them (wanted {n})")
            chosen = subset
        else:
            chosen = subset.sample(n=n, random_state=seed)

        for _, row in chosen.iterrows():
            # Build a dict of all disease labels for this image
            label_cols = list(CHEXPERT_CATEGORIES.values())
            labels = {c: (float(row[c]) if c in df.columns and pd.notna(row[c]) else None)
                      for c in label_cols}
            records.append({
                "image_id": Path(row["abs_path"]).stem,
                "original_path": row["abs_path"],
                "disease_category": category,
                "disease_labels": labels,
                "source": "chexpert",
                "split": split,
            })

    return records


# ---------------------------------------------------------------------------
# NIH ChestX-ray14 loader
# ---------------------------------------------------------------------------

NIH_CATEGORIES = {
    "no_finding":      "No Finding",
    "pneumonia":       "Pneumonia",
    "cardiomegaly":    "Cardiomegaly",
    "pleural_effusion":"Effusion",
    "atelectasis":     "Atelectasis",
}

def find_nih_image(data_dir: Path, filename: str) -> str | None:
    """Search common NIH sub-directory layouts for an image filename."""
    # Common layouts:
    #   data_dir/images/*.png
    #   data_dir/images_001/images/*.png  ...  images_012/images/*.png
    for candidate in data_dir.rglob(filename):
        return str(candidate)
    return None


def load_nih(data_dir: Path) -> pd.DataFrame:
    csv_path = data_dir / "Data_Entry_2017.csv"
    df = pd.read_csv(csv_path)
    # Keep frontal (PA or AP) only
    if "View Position" in df.columns:
        df = df[df["View Position"].isin(["PA", "AP"])].copy()
    return df


def select_nih(data_dir: Path, n: int, seed: int, single_label: bool = True) -> list:
    df = load_nih(data_dir)
    rng = random.Random(seed)
    records = []

    for category, nih_label in NIH_CATEGORIES.items():
        if nih_label == "No Finding":
            # No Finding is a singleton label by definition
            mask = df["Finding Labels"] == "No Finding"
        elif single_label:
            # Exact match: image has *only* this label, no co-occurring findings.
            # Avoids the L4/L5 confound where the negative prompt set names a
            # disease that the image actually has alongside the target.
            mask = df["Finding Labels"] == nih_label
        else:
            mask = df["Finding Labels"].str.contains(nih_label, na=False)

        subset = df[mask].copy()

        if len(subset) < n:
            print(f"  [WARN] Only {len(subset)} '{nih_label}' images — "
                  f"using all of them (wanted {n})")
            chosen = subset
        else:
            chosen = subset.sample(n=n, random_state=seed)

        for _, row in chosen.iterrows():
            img_file = row["Image Index"]
            abs_path = find_nih_image(data_dir, img_file)
            if abs_path is None:
                print(f"  [WARN] Image file not found: {img_file} — skipping")
                continue

            raw_labels = [l.strip() for l in row["Finding Labels"].split("|")]
            labels = {v: (1 if v in raw_labels else 0)
                      for v in NIH_CATEGORIES.values()}

            records.append({
                "image_id": Path(img_file).stem,
                "original_path": abs_path,
                "disease_category": category,
                "disease_labels": labels,
                "source": "nih",
                "split": "all",
            })

    return records


# ---------------------------------------------------------------------------
# Copy images + write manifest
# ---------------------------------------------------------------------------

def copy_probes(records: list, out_dir: Path, repo_root: Path) -> list:
    """Copy selected images to out_dir/images/ and attach final paths.

    image_path in each record is stored relative to repo_root so that
    _resolve_image(path) = repo_root / path resolves correctly at eval time.
    """
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    updated = []
    seen_names = {}
    for rec in records:
        src = Path(rec["original_path"])
        stem = src.stem
        # Disambiguate collisions by prepending category
        key = f"{rec['disease_category']}_{stem}"
        if key in seen_names:
            seen_names[key] += 1
            dest_name = f"{key}_{seen_names[key]}{src.suffix}"
        else:
            seen_names[key] = 0
            dest_name = f"{key}{src.suffix}"

        dest = img_dir / dest_name
        shutil.copy2(src, dest)
        rec = dict(rec)
        # Path relative to repo root (e.g. "data/cxr_probes/images/foo.png")
        rec["image_path"] = str(dest.relative_to(repo_root))
        rec["filename"] = dest_name
        updated.append(rec)

    return updated


def write_manifest(records: list, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nManifest written -> {manifest_path}")
    return manifest_path


def print_summary(records: list):
    from collections import Counter
    counts = Counter(r["disease_category"] for r in records)
    print("\n--- Selection summary ---")
    for cat, n in sorted(counts.items()):
        print(f"  {cat:20s}: {n} images")
    print(f"  {'TOTAL':20s}: {len(records)} images")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Select 50 stratified CXR probe images")
    parser.add_argument("--data-dir", required=True,
                        help="Root of CheXpert or NIH ChestX-ray14 dataset")
    parser.add_argument("--out-dir", default="data/cxr_probes",
                        help="Output directory for probe images + manifest")
    parser.add_argument("--n-per-category", type=int, default=10,
                        help="Images to select per disease category (default: 10)")
    parser.add_argument("--split", default="train",
                        help="CheXpert split to use: train or valid (ignored for NIH)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--single-label", action="store_true", default=True,
                        help="Filter to images with a single label only (NIH only). "
                             "Avoids L4/L5 multi-label confound. Default: True.")
    parser.add_argument("--allow-multi-label", dest="single_label",
                        action="store_false",
                        help="Disable single-label filter (legacy behavior).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print selection summary without copying files")
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

    fmt = detect_format(data_dir)
    print(f"Detected dataset format: {fmt.upper()}")
    print(f"Data dir  : {data_dir}")
    print(f"Output dir: {out_dir}")
    print(f"n per cat : {args.n_per_category}, seed: {args.seed}")

    if fmt == "chexpert":
        records = select_chexpert(data_dir, args.n_per_category, args.split, args.seed)
    else:
        records = select_nih(data_dir, args.n_per_category, args.seed,
                             single_label=args.single_label)
        print(f"Single-label filter: {'ON' if args.single_label else 'OFF'}")

    print_summary(records)

    if args.dry_run:
        print("\n[dry-run] No files copied.")
        return

    records = copy_probes(records, out_dir, repo_root)
    write_manifest(records, out_dir)
    print(f"\nDone. {len(records)} probe images in {out_dir / 'images'}")


if __name__ == "__main__":
    main()
