"""
select_openi_full_probes.py — Build a MedTEC probe manifest from the full
OpenI release using the existing CheXpert-5 labels (atelectasis, cardiomegaly,
consolidation, edema, effusion).

Differences vs. the earlier 50-image OpenI manifest:
  * uses every OpenI image with a clean single-label assignment instead of
    the 10-per-category stratified subset (3,318 images vs 50)
  * supports OpenI's two extra disease categories (consolidation, edema) —
    requires the extended cxr_prompts.json that adds disease_prompts for
    these two categories at L4/L5
  * maps OpenI's "effusion" label onto the existing "pleural_effusion" key
    so the existing CXR prompt config covers it directly

Single-label means: of the five labelled findings, exactly one is positive,
or all five are zero (the latter becomes disease_category="no_finding").
Multi-label images (~371 of 3,689) are dropped — the L4/L5 design assumes a
single primary finding per image, and choosing "the" primary one introduces
arbitrary tie-breaking. They can be added back with a more careful selector
once we decide the rule.

Usage:
    python scripts/select_openi_full_probes.py
    python scripts/select_openi_full_probes.py --no-finding-cap 500   # cap "normal" images
    python scripts/select_openi_full_probes.py --out-dir data/openi_probes_full

Output:
    data/openi_probes_full/manifest.json
    data/openi_probes_full/selection_summary.json
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

REPO = Path(__file__).parent.parent

# Map OpenI label vocabulary -> CXR prompt config disease_category keys.
# OpenI uses "effusion"; cxr_prompts.json uses "pleural_effusion" for the
# same finding, so we normalize here so the prompt lookup works directly.
OPENI_TO_CXR_CATEGORY = {
    "atelectasis":   "atelectasis",
    "cardiomegaly":  "cardiomegaly",
    "consolidation": "consolidation",
    "edema":         "edema",
    "effusion":      "pleural_effusion",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/openi/labels_chexpert5.json",
                    help="OpenI CheXpert-5 labels JSON.")
    ap.add_argument("--image-dir", default="data/openi/images",
                    help="Directory holding OpenI image PNGs.")
    ap.add_argument("--out-dir", default="data/openi_probes_full",
                    help="Output directory for the new manifest.")
    ap.add_argument("--no-finding-cap", type=int, default=None,
                    help="Optional cap on the no_finding category (random "
                         "sample with --seed). Default: keep all.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    labels_path = REPO / args.labels
    img_dir = REPO / args.image_dir
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {labels_path}")
    labels = json.load(open(labels_path, encoding="utf-8"))["labels"]
    print(f"  {len(labels)} labelled images")

    # Bucket images by single-label disease (or "no_finding" for 0 positives).
    bucket = {k: [] for k in OPENI_TO_CXR_CATEGORY.keys()}
    bucket["no_finding"] = []
    n_multi = 0
    n_missing_image = 0

    for img_id, lab_dict in labels.items():
        positives = [k for k, v in lab_dict.items() if v == 1]
        # locate image file (id is either bare or already prefixed)
        img_path = img_dir / f"{img_id}.png"
        if not img_path.exists():
            # try common variations
            alt = img_dir / f"openi_{img_id}.png" if not img_id.startswith("openi_") else img_dir / f"{img_id}.png"
            if alt.exists():
                img_path = alt
            else:
                n_missing_image += 1
                continue

        if len(positives) == 0:
            bucket["no_finding"].append((img_id, img_path, lab_dict))
        elif len(positives) == 1:
            bucket[positives[0]].append((img_id, img_path, lab_dict))
        else:
            n_multi += 1

    print(f"\nSingle-label bucket sizes:")
    for k in ["no_finding"] + list(OPENI_TO_CXR_CATEGORY.keys()):
        print(f"  {k:<15}: {len(bucket[k])}")
    print(f"  multi-label (dropped): {n_multi}")
    print(f"  image file missing (dropped): {n_missing_image}")

    # Optional cap on no_finding (heavily over-represented at 2244)
    rng = random.Random(args.seed)
    if args.no_finding_cap is not None and len(bucket["no_finding"]) > args.no_finding_cap:
        n_before = len(bucket["no_finding"])
        bucket["no_finding"] = rng.sample(bucket["no_finding"], args.no_finding_cap)
        print(f"\nCapped no_finding from {n_before} -> {len(bucket['no_finding'])} (seed={args.seed})")

    # Build the manifest entries
    records = []
    for openi_cat, items in bucket.items():
        # canonical category key used by cxr_prompts.json
        if openi_cat == "no_finding":
            cxr_cat = "no_finding"
            disease_labels = {"No Finding": 1, "Pneumonia": 0, "Cardiomegaly": 0,
                              "Effusion": 0, "Atelectasis": 0, "Consolidation": 0, "Edema": 0}
        else:
            cxr_cat = OPENI_TO_CXR_CATEGORY[openi_cat]
            # Build NIH-style disease_labels dict for compatibility with the
            # existing per-image probe lookup machinery.
            disease_labels = {"No Finding": 0, "Pneumonia": 0, "Cardiomegaly": 0,
                              "Effusion": 0, "Atelectasis": 0, "Consolidation": 0, "Edema": 0}
            label_key = {"atelectasis": "Atelectasis", "cardiomegaly": "Cardiomegaly",
                         "consolidation": "Consolidation", "edema": "Edema",
                         "effusion": "Effusion"}[openi_cat]
            disease_labels[label_key] = 1

        for img_id, img_path, full_labels in items:
            records.append({
                "image_id": img_id,
                "original_path": str(img_path),
                "disease_category": cxr_cat,
                "disease_labels": disease_labels,
                "openi_label_dict": full_labels,
                "source": "openi",
                "split": "all",
                "image_path": str(img_path),
                "filename": img_path.name,
            })

    # Persist
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"\nWrote {manifest_path}: {len(records)} entries")

    summary = {
        "n_total": len(records),
        "by_category": dict(Counter(r["disease_category"] for r in records)),
        "n_multi_label_dropped": n_multi,
        "n_missing_image_dropped": n_missing_image,
        "no_finding_cap": args.no_finding_cap,
        "seed": args.seed,
    }
    with open(out_dir / "selection_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n--- Summary ---\n{json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
