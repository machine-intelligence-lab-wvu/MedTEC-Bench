"""
Select stratified histopathology tiles for MedTEC-Bench probing.

Source: NCT-CRC-HE-100K (9-class colorectal H&E tiles, 224x224).
Each class has its own folder of .tif tiles.

Expected layout:
  <data-dir>/
    NCT-CRC-HE-100K/        (or NCT-CRC-HE-7K as a smaller validation set)
      ADI/   *.tif
      BACK/  *.tif
      DEB/   *.tif
      LYM/   *.tif
      MUC/   *.tif
      MUS/   *.tif
      NORM/  *.tif
      STR/   *.tif
      TUM/   *.tif

The script searches recursively under --data-dir for the 9 class folder
names so alternate download layouts work.

Output:
  data/histopath_probes/images/       — copied tiles (flat, renamed by class)
  data/histopath_probes/manifest.json — per-tile metadata + tissue class

Usage:
  python scripts/select_histopath_probes.py --data-dir data/histopath
  python scripts/select_histopath_probes.py --data-dir data/histopath --n-per-category 6
"""

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------------------
# NCT-CRC-HE-100K tissue classes
# internal_key -> (display_name, folder_name)
# ---------------------------------------------------------------------------

HISTOPATH_CATEGORIES = {
    "tum":  ("Colorectal Adenocarcinoma Epithelium", "TUM"),
    "norm": ("Normal Colon Mucosa",                  "NORM"),
    "str":  ("Cancer-Associated Stroma",             "STR"),
    "lym":  ("Lymphocytes",                          "LYM"),
    "mus":  ("Smooth Muscle",                        "MUS"),
    "adi":  ("Adipose Tissue",                       "ADI"),
    "muc":  ("Mucus",                                "MUC"),
    "deb":  ("Debris",                               "DEB"),
    "back": ("Background",                           "BACK"),
}

IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


# ---------------------------------------------------------------------------
# Folder discovery (recursive)
# ---------------------------------------------------------------------------

def find_class_dir(data_dir: Path, folder_name: str) -> Path | None:
    """Search under data_dir for the first directory matching folder_name."""
    for candidate in data_dir.rglob(folder_name):
        if candidate.is_dir():
            return candidate
    return None


def list_tiles(class_dir: Path) -> list:
    return sorted(p for p in class_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def select_stratified(data_dir: Path, n: int, seed: int) -> list:
    rng = random.Random(seed)
    records = []

    for cat_key, (display, folder) in HISTOPATH_CATEGORIES.items():
        cls_dir = find_class_dir(data_dir, folder)
        if cls_dir is None:
            print(f"  [WARN] class folder '{folder}' not found under {data_dir} — skipping {cat_key}")
            continue

        tiles = list_tiles(cls_dir)
        if not tiles:
            print(f"  [WARN] no tiles in {cls_dir} — skipping {cat_key}")
            continue

        if len(tiles) < n:
            print(
                f"  [WARN] only {len(tiles)} tiles in {folder} — "
                f"using all (wanted {n})"
            )
            chosen = tiles
        else:
            chosen = rng.sample(tiles, n)

        for tile in chosen:
            records.append({
                "image_id": tile.stem,
                "original_path": str(tile),
                "disease_category": cat_key,
                "disease_display": display,
                "nct_label": folder,
                "source": "nct_crc_he_100k",
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
        display = HISTOPATH_CATEGORIES.get(cat, (cat, ""))[0]
        print(f"  {cat:<6}  {display:<40}: {n} tiles")
    print(f"  {'TOTAL':<6}  {'':40}: {len(records)} tiles")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Select stratified histopathology probe tiles from NCT-CRC-HE-100K"
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Root of NCT-CRC-HE data (must contain 9 class subfolders: TUM, NORM, ...)",
    )
    parser.add_argument(
        "--out-dir", default="data/histopath_probes",
        help="Output directory (default: data/histopath_probes)",
    )
    parser.add_argument(
        "--n-per-category", type=int, default=6,
        help="Tiles per tissue class (default: 6 → 54 total)",
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

    # Sanity check: does data_dir actually have any class folder? If not, the
    # dataset probably has not been unzipped yet — abort early with a clear msg.
    any_class = any(
        find_class_dir(data_dir, folder) is not None
        for _, folder in HISTOPATH_CATEGORIES.values()
    )
    if not any_class:
        print(
            f"ERROR: none of the 9 NCT-CRC-HE class folders (TUM, NORM, STR, LYM, "
            f"MUS, ADI, MUC, DEB, BACK) were found under {data_dir}.\n"
            f"If the dataset has not been downloaded/unzipped, fetch NCT-CRC-HE-100K "
            f"(Zenodo DOI 10.5281/zenodo.1214456) and re-run.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Data dir  : {data_dir}")
    print(f"Output dir: {out_dir}")
    print(f"n per cat : {args.n_per_category}, seed: {args.seed}")

    records = select_stratified(data_dir, args.n_per_category, args.seed)
    print_summary(records)

    if not records:
        print("\nERROR: No records selected.", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print("\n[dry-run] No files copied.")
        return

    records = copy_probes(records, out_dir, repo_root)
    write_manifest(records, out_dir)
    print(f"\nDone. {len(records)} probe tiles in {out_dir / 'images'}")


if __name__ == "__main__":
    main()
