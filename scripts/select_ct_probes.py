"""
Select stratified abdominal CT 2D slices for MedTEC-Bench probing.

Source: AMOS 2022 (Abdominal Multi-Organ Segmentation).
For each target organ label, pick the axial slice with the *largest*
mask area for that label; then render the CT slice to PNG with a
standard soft-tissue window.

Expected layout (.nii.gz volumes + label volumes):
  <data-dir>/
    imagesTr/ *.nii.gz      ← training CT volumes
    labelsTr/ *.nii.gz      ← matching label volumes (same filename)
    imagesVa/ *.nii.gz      (optional — validation)
    labelsVa/ *.nii.gz

Output:
  data/ct_probes/images/       — rendered PNG slices (flat)
  data/ct_probes/manifest.json — per-slice metadata + organ category

Requires: nibabel, numpy, pillow (optional — falls back to skimage if needed).

Usage:
  python scripts/select_ct_probes.py --data-dir data/amos
  python scripts/select_ct_probes.py --data-dir data/amos --n-per-category 8
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# AMOS label ids (CT task) — see AMOS 2022 challenge documentation.
# internal_key -> (display_name, amos_label_id)
# ---------------------------------------------------------------------------

CT_CATEGORIES = {
    "spleen":        ("Spleen",        1),
    "kidney_right":  ("Right Kidney",  2),
    "kidney_left":   ("Left Kidney",   3),
    "gallbladder":   ("Gallbladder",   4),
    "liver":         ("Liver",         6),
    "pancreas":      ("Pancreas",      10),
}

VOL_SUBDIRS = [("imagesTr", "labelsTr"), ("imagesVa", "labelsVa")]
VOL_EXTS = {".nii", ".nii.gz"}


def _has_nii_ext(path: Path) -> bool:
    s = path.name.lower()
    return s.endswith(".nii") or s.endswith(".nii.gz")


# ---------------------------------------------------------------------------
# Slice rendering
# ---------------------------------------------------------------------------

def _window_to_uint8(slice2d, wl=40.0, ww=400.0):
    """Apply soft-tissue window and scale to uint8."""
    import numpy as np
    lo, hi = wl - ww / 2.0, wl + ww / 2.0
    s = np.clip(slice2d, lo, hi)
    s = (s - lo) / max(hi - lo, 1e-6)
    return (s * 255.0).astype("uint8")


def _save_png(arr2d, dest_path: Path):
    try:
        from PIL import Image  # type: ignore
        Image.fromarray(arr2d).save(dest_path)
    except ImportError:
        import imageio.v2 as imageio  # type: ignore
        imageio.imwrite(str(dest_path), arr2d)


# ---------------------------------------------------------------------------
# Volume pair discovery
# ---------------------------------------------------------------------------

def discover_volume_pairs(data_dir: Path) -> list:
    """Yield (image_path, label_path) pairs that both exist."""
    pairs = []
    for img_sub, lbl_sub in VOL_SUBDIRS:
        img_dir = data_dir / img_sub
        lbl_dir = data_dir / lbl_sub
        if not img_dir.exists() or not lbl_dir.exists():
            continue
        for img_path in sorted(img_dir.iterdir()):
            if not img_path.is_file() or not _has_nii_ext(img_path):
                continue
            lbl_path = lbl_dir / img_path.name
            if lbl_path.exists():
                pairs.append((img_path, lbl_path))
    return pairs


# ---------------------------------------------------------------------------
# Slice selection per organ
# ---------------------------------------------------------------------------

def select_slices(pairs: list, n_per_category: int, seed: int) -> list:
    """
    For each organ, rank volumes by max 2D label area for that organ,
    pick the top n_per_category volumes, and record the best-area slice.
    """
    import numpy as np
    # nibabel 5.4.0 silently crashes on AMOS uint16 labels in this env;
    # use the bundled raw-gzip parser instead.
    from _amos_loader import load_volume  # type: ignore

    rng = random.Random(seed)

    # org_key -> list of (max_area, vol_idx, slice_idx)
    per_organ = defaultdict(list)

    print(f"Scanning {len(pairs)} CT volumes for organ slices...")
    for vi, (img_p, lbl_p) in enumerate(pairs):
        try:
            lbl = load_volume(lbl_p)
        except Exception as e:
            print(f"  [WARN] failed to load {lbl_p}: {e}")
            continue

        # Assume last axis = axial (standard for AMOS); compute per-slice area.
        for org_key, (_, lbl_id) in CT_CATEGORIES.items():
            mask = (lbl == lbl_id)
            if not mask.any():
                continue
            # Determine axial axis: choose axis whose length looks like "slices"
            # AMOS CT conventionally stores axial in axis 2 (shape ~ H,W,Z).
            axis = 2 if lbl.ndim == 3 else lbl.ndim - 1
            areas = mask.sum(axis=tuple(i for i in range(lbl.ndim) if i != axis))
            best_slice = int(areas.argmax())
            best_area = int(areas[best_slice])
            if best_area > 0:
                per_organ[org_key].append((best_area, vi, best_slice))

    # Pick top-N by area per organ; if ties, break randomly.
    records = []
    for org_key, (display, lbl_id) in CT_CATEGORIES.items():
        candidates = per_organ.get(org_key, [])
        if not candidates:
            print(f"  [WARN] no volumes contain label {lbl_id} ({org_key})")
            continue
        candidates.sort(key=lambda t: (-t[0], rng.random()))
        chosen = candidates[:n_per_category]
        if len(chosen) < n_per_category:
            print(
                f"  [WARN] only {len(chosen)} volumes have {org_key}; "
                f"wanted {n_per_category}"
            )

        for best_area, vi, slice_idx in chosen:
            img_p, lbl_p = pairs[vi]
            records.append({
                "volume_path": str(img_p),
                "label_path": str(lbl_p),
                "slice_idx": slice_idx,
                "label_area_px": best_area,
                "disease_category": org_key,
                "disease_display": display,
                "amos_label_id": lbl_id,
                "source": "amos_2022",
            })

    return records


# ---------------------------------------------------------------------------
# Render slices + manifest
# ---------------------------------------------------------------------------

def render_probes(records: list, out_dir: Path, repo_root: Path) -> list:
    import numpy as np
    # nibabel 5.4.0 silently crashes on AMOS uint16 labels in this env;
    # use the bundled raw-gzip parser instead.
    from _amos_loader import load_volume  # type: ignore

    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    updated = []
    seen_names: dict = {}

    # Cache volume loads to avoid re-reading the same volume many times.
    vol_cache: dict = {}

    for rec in records:
        vol_path = rec["volume_path"]
        if vol_path not in vol_cache:
            vol_cache[vol_path] = load_volume(vol_path)
        vol = vol_cache[vol_path]

        # Axial axis handling (match select_slices logic)
        axis = 2 if vol.ndim == 3 else vol.ndim - 1
        slc = vol.take(rec["slice_idx"], axis=axis)
        # AMOS HU values — use soft-tissue window
        slc_u8 = _window_to_uint8(slc, wl=40.0, ww=400.0)

        vol_stem = Path(vol_path).name.replace(".nii.gz", "").replace(".nii", "")
        key = f"{rec['disease_category']}_{vol_stem}_s{rec['slice_idx']}"
        if key in seen_names:
            seen_names[key] += 1
            dest_name = f"{key}_{seen_names[key]}.png"
        else:
            seen_names[key] = 0
            dest_name = f"{key}.png"

        dest = img_dir / dest_name
        _save_png(slc_u8, dest)

        rec = dict(rec)
        rec["image_path"] = dest.relative_to(repo_root).as_posix()
        rec["filename"] = dest_name
        rec["image_id"] = dest.stem
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
        display = CT_CATEGORIES.get(cat, (cat, -1))[0]
        print(f"  {cat:<15}  {display:<20}: {n} slices")
    print(f"  {'TOTAL':<15}  {'':20}: {len(records)} slices")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Select stratified abdominal CT probe slices from AMOS 2022"
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Root of AMOS data — must contain imagesTr/ and labelsTr/",
    )
    parser.add_argument(
        "--out-dir", default="data/ct_probes",
        help="Output directory (default: data/ct_probes)",
    )
    parser.add_argument(
        "--n-per-category", type=int, default=8,
        help="Slices per organ category (default: 8)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print summary without rendering PNGs",
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

    try:
        import nibabel  # noqa: F401
        import numpy    # noqa: F401
    except ImportError as e:
        print(
            f"ERROR: missing dependency ({e}). Install with: "
            f"pip install nibabel numpy pillow",
            file=sys.stderr,
        )
        sys.exit(1)

    pairs = discover_volume_pairs(data_dir)
    if not pairs:
        print(
            f"ERROR: no (image, label) volume pairs found under {data_dir}.\n"
            f"Expected imagesTr/*.nii.gz + labelsTr/*.nii.gz (same filenames).\n"
            f"If AMOS is not yet downloaded, fetch it first and re-run.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Data dir   : {data_dir}")
    print(f"Output dir : {out_dir}")
    print(f"n per cat  : {args.n_per_category}, seed: {args.seed}")
    print(f"Volume pairs found: {len(pairs)}")

    records = select_slices(pairs, args.n_per_category, args.seed)
    print_summary(records)

    if not records:
        print("\nERROR: No slices selected.", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print("\n[dry-run] No PNGs rendered.")
        return

    records = render_probes(records, out_dir, repo_root)
    write_manifest(records, out_dir)
    print(f"\nDone. {len(records)} probe slices in {out_dir / 'images'}")


if __name__ == "__main__":
    main()
