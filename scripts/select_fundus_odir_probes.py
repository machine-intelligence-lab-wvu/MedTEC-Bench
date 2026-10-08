"""
Select stratified fundus images from ODIR-5K for MedTEC-Bench probing.

Source: ODIR-5K (Ocular Disease Intelligent Recognition).
Layout expected (Kaggle download):
  <data-dir>/
    archive/ODIR-5K/ODIR-5K/
      data.xlsx                   ← per-patient annotations (3,500 rows)
      Training Images/            ← 7,000 images (3,500 patients × 2 eyes)
      Testing Images/             ← 1,000 test images

Filtering — per eye, single-label only:
  Each eye has its own keyword list (Left- or Right-Diagnostic Keywords).
  An eye is kept iff:
    (1) at least one keyword maps to the target class, AND
    (2) no keyword maps to any *other* of the 7 target classes, AND
    (3) every other keyword on that eye is in the EXCLUDED-DISEASE-FREE
        whitelist (image-quality artefacts only — lens dust, low image
        quality, refractive media opacity).
  Eyes with co-occurring out-of-scope diseases (vein occlusion, papilledema,
  retinitis pigmentosa, etc.) are dropped to keep probe pools clean.

Output:
  data/fundus_probes_n1000/images/       ← copied PNG/JPG slices (flat)
  data/fundus_probes_n1000/manifest.json ← per-image metadata + class

Usage:
  python scripts/select_fundus_odir_probes.py --data-dir data/ODIR-5K
  python scripts/select_fundus_odir_probes.py --data-dir data/ODIR-5K --n-per-category 150
"""

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Class mapping — keyword → (class_key | "exclude" | "artefact")
# ---------------------------------------------------------------------------

# Disease keywords mapped to one of the 7 target classes (drop O bucket).
KEYWORD_TO_CLASS = {
    # N — Normal
    "normal fundus": "normal",

    # D — Diabetic retinopathy (NPDR/PDR collapsed)
    "mild nonproliferative retinopathy": "diabetic_retinopathy",
    "moderate non proliferative retinopathy": "diabetic_retinopathy",
    "severe nonproliferative retinopathy": "diabetic_retinopathy",
    "diabetic retinopathy": "diabetic_retinopathy",
    "proliferative diabetic retinopathy": "diabetic_retinopathy",
    "severe proliferative diabetic retinopathy": "diabetic_retinopathy",

    # G — Glaucoma
    "glaucoma": "glaucoma",
    "suspected glaucoma": "glaucoma",

    # C — Cataract
    "cataract": "cataract",

    # A — AMD (dry/wet collapsed)
    "dry age-related macular degeneration": "amd",
    "wet age-related macular degeneration": "amd",
    "age-related macular degeneration": "amd",

    # H — Hypertensive retinopathy
    "hypertensive retinopathy": "hypertensive_retinopathy",

    # M — Pathological myopia
    "pathological myopia": "pathological_myopia",
    "myopia retinopathy": "pathological_myopia",
    "myopic retinopathy": "pathological_myopia",
    "myopic maculopathy": "pathological_myopia",
}

# Image-quality / non-disease artefacts — allowed to co-occur on any class.
ARTEFACT_KEYWORDS = {
    "lens dust",
    "low image quality",
    "image offset",
    "optic disk photographically invisible",
    "anterior segment image",
    "no fundus image",
}

CLASS_DISPLAY = {
    "normal":                   "Normal Retina",
    "diabetic_retinopathy":     "Diabetic Retinopathy",
    "glaucoma":                 "Glaucoma",
    "cataract":                 "Cataract",
    "amd":                      "Age-related Macular Degeneration",
    "hypertensive_retinopathy": "Hypertensive Retinopathy",
    "pathological_myopia":      "Pathological Myopia",
}

ALL_CLASSES = list(CLASS_DISPLAY.keys())


# ---------------------------------------------------------------------------
# Data location
# ---------------------------------------------------------------------------

# data-dir candidates → (xlsx path, [image dirs in priority order])
def discover_layout(data_dir: Path):
    candidates = [
        # Kaggle nested
        data_dir / "archive" / "ODIR-5K" / "ODIR-5K",
        # Already-unpacked
        data_dir / "ODIR-5K",
        # Direct
        data_dir,
    ]
    for root in candidates:
        xlsx = root / "data.xlsx"
        if xlsx.exists():
            train_dir = root / "Training Images"
            test_dir = root / "Testing Images"
            img_dirs = [d for d in (train_dir, test_dir) if d.exists()]
            if img_dirs:
                return xlsx, img_dirs
    raise FileNotFoundError(
        f"Could not find data.xlsx + Training Images/ under {data_dir}.\n"
        f"Tried: {[str(c) for c in candidates]}"
    )


def parse_keywords(s) -> list:
    """Split an ODIR keyword field on Latin or Chinese commas."""
    if pd.isna(s) or s is None:
        return []
    text = str(s).replace("，", ",")  # Chinese comma → ASCII comma
    return [k.strip() for k in text.split(",") if k.strip()]


def classify_eye(keywords: list) -> str | None:
    """
    Return the target class if this eye's keyword list is single-label
    against our 7 classes, else None.

    Rules:
      - At least one keyword maps to a target class.
      - No two distinct target classes appear.
      - All non-target keywords are in ARTEFACT_KEYWORDS (else exclude).
    """
    target_classes = set()
    for kw in keywords:
        kw_low = kw.lower()
        if kw_low in KEYWORD_TO_CLASS:
            target_classes.add(KEYWORD_TO_CLASS[kw_low])
        elif kw_low in ARTEFACT_KEYWORDS:
            continue
        else:
            # Unrecognised keyword = some other disease → exclude
            return None
    if len(target_classes) == 1:
        return next(iter(target_classes))
    return None


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_eyes(xlsx_path: Path) -> pd.DataFrame:
    """Return one row per eye with columns: image_file, side, keywords, class."""
    df = pd.read_excel(xlsx_path)
    rows = []
    for _, r in df.iterrows():
        for side, fname_col, kw_col in (
            ("left",  "Left-Fundus",  "Left-Diagnostic Keywords"),
            ("right", "Right-Fundus", "Right-Diagnostic Keywords"),
        ):
            fname = r.get(fname_col)
            if pd.isna(fname) or not fname:
                continue
            kws = parse_keywords(r.get(kw_col))
            cls = classify_eye(kws)
            rows.append({
                "patient_id": int(r["ID"]),
                "side": side,
                "image_file": str(fname),
                "keywords": kws,
                "disease_category": cls,
                "patient_age": r.get("Patient Age"),
                "patient_sex": r.get("Patient Sex"),
            })
    return pd.DataFrame(rows)


def find_image(image_file: str, img_dirs: list) -> Path | None:
    for d in img_dirs:
        p = d / image_file
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def select_stratified(eyes_df: pd.DataFrame, img_dirs: list, n_per_category: int, seed: int) -> list:
    print("\n--- Single-label eye counts (before sampling) ---")
    counts = eyes_df["disease_category"].value_counts(dropna=False)
    print(counts.to_string())

    records = []
    for cls in ALL_CLASSES:
        subset = eyes_df[eyes_df["disease_category"] == cls].copy()
        # Filter to eyes whose images actually exist on disk
        subset["abs_path"] = subset["image_file"].apply(
            lambda f: find_image(f, img_dirs)
        )
        subset = subset[subset["abs_path"].notna()].copy()

        if len(subset) == 0:
            print(f"  [WARN] no usable eyes for class '{cls}'")
            continue

        if len(subset) <= n_per_category:
            chosen = subset
            print(f"  [INFO] class '{cls}': only {len(subset)} eyes available; using all")
        else:
            chosen = subset.sample(n=n_per_category, random_state=seed)

        for _, row in chosen.iterrows():
            records.append({
                "image_id": Path(row["image_file"]).stem,
                "original_path": str(row["abs_path"]),
                "disease_category": cls,
                "disease_display": CLASS_DISPLAY[cls],
                "side": row["side"],
                "patient_id": row["patient_id"],
                "keywords": row["keywords"],
                "patient_age": (None if pd.isna(row["patient_age"]) else int(row["patient_age"])),
                "patient_sex": (None if pd.isna(row["patient_sex"]) else str(row["patient_sex"])),
                "source": "odir_5k",
            })

    return records


# ---------------------------------------------------------------------------
# Copy + manifest
# ---------------------------------------------------------------------------

def copy_probes(records: list, out_dir: Path, repo_root: Path) -> list:
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    seen = {}
    updated = []
    for rec in records:
        src = Path(rec["original_path"])
        key = f"{rec['disease_category']}_{src.stem}"
        if key in seen:
            seen[key] += 1
            dest_name = f"{key}_{seen[key]}{src.suffix}"
        else:
            seen[key] = 0
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
    for cat in ALL_CLASSES:
        n = counts.get(cat, 0)
        print(f"  {cat:<26} {CLASS_DISPLAY[cat]:<35}: {n} eyes")
    print(f"  {'TOTAL':<26} {'':35}: {len(records)} eyes")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Select ODIR-5K fundus probes")
    parser.add_argument("--data-dir", required=True,
        help="Root of ODIR-5K (will look for archive/ODIR-5K/ODIR-5K/)")
    parser.add_argument("--out-dir", default="data/fundus_probes_n1000")
    parser.add_argument("--n-per-category", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
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

    xlsx, img_dirs = discover_layout(data_dir)
    print(f"Annotations: {xlsx}")
    print(f"Image dirs : {[str(d) for d in img_dirs]}")
    print(f"Output dir : {out_dir}")
    print(f"n per cat  : {args.n_per_category}, seed: {args.seed}")

    eyes_df = load_eyes(xlsx)
    print(f"Total eyes loaded: {len(eyes_df)}")

    records = select_stratified(eyes_df, img_dirs, args.n_per_category, args.seed)
    print_summary(records)

    if not records:
        print("\nERROR: No records selected.", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print("\n[dry-run] No images copied.")
        return

    records = copy_probes(records, out_dir, repo_root)
    write_manifest(records, out_dir)
    print(f"\nDone. {len(records)} probe images in {out_dir / 'images'}")


if __name__ == "__main__":
    main()
