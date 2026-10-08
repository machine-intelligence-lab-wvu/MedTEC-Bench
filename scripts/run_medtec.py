"""
Run MedTEC-Bench: Text Embedding Collapse evaluation across all models and levels.

Run the modality's probe-selector script first
(select_{cxr,derm,mri,ct,fundus,histopath}_probes.py) to build
data/<modality>_probes/manifest.json + images/. Then pass the manifest to
this runner so L4/L5 use per-image disease-specific probes.

Usage:
    # CXR
    python scripts/run_medtec.py --modality cxr --manifest data/cxr_probes/manifest.json

    # Dermoscopy
    python scripts/run_medtec.py --modality dermoscopy --manifest data/derm_probes/manifest.json

    # Brain MRI
    python scripts/run_medtec.py --modality brain_mri --manifest data/mri_probes/manifest.json

    # Abdomen CT
    python scripts/run_medtec.py --modality abdomen_ct --manifest data/ct_probes/manifest.json

    # Retinal fundus
    python scripts/run_medtec.py --modality fundus --manifest data/fundus_probes/manifest.json

    # Histopathology
    python scripts/run_medtec.py --modality histopath --manifest data/histopath_probes/manifest.json

    # Specific models / levels
    python scripts/run_medtec.py --modality cxr --manifest data/cxr_probes/manifest.json \\
        --models biomedclip openclip_vitb32 --levels 4 5
"""

import os
# Windows + conda + torch/numpy can link two OpenMP runtimes. Set before any
# numpy/torch import to avoid the OMP Error #15 crash.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from models.model_registry import load_model, list_models, MODEL_REGISTRY
from benchmarks.medtec_bench import (
    get_probes_by_level,
    has_image_probes,
    LEVEL_DEFINITIONS,
    SUPPORTED_MODALITIES,
)
from metrics.collapse_metrics import evaluate_probe


def _resolve_image(path_str: str) -> Path:
    """Resolve an image path that may be absolute or relative to repo root."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def _encode_image_cache(model, image_paths, batch_size=32) -> dict:
    """Encode every image once and return a {path -> embedding} dict.

    The original implementation re-opened and re-encoded each image inside
    nested (probe x image) loops, which is fine at 50 images and crippling
    at 3,000+. This pre-encodes everything in batches so subsequent shared-
    probe and per-image level evaluations are pure vector arithmetic.
    """
    cache = {}
    if not image_paths:
        return cache
    # Deduplicate while preserving order
    unique = list(dict.fromkeys(str(p) for p in image_paths))
    for i in tqdm(range(0, len(unique), batch_size),
                  desc="  encode imgs", leave=False, ncols=80):
        batch_paths = unique[i:i + batch_size]
        batch_imgs = []
        valid_paths = []
        for p in batch_paths:
            try:
                batch_imgs.append(Image.open(p).convert("RGB"))
                valid_paths.append(p)
            except Exception as e:
                print(f"  Warning: could not open {p}: {e}")
        if not batch_imgs:
            continue
        embs = model.encode_image(batch_imgs).cpu().numpy()
        for p, e in zip(valid_paths, embs):
            cache[p] = np.asarray(e, dtype=np.float32).squeeze()
    return cache


def run_level(model, level, modality, image_paths, manifest_path, repo_root,
              image_emb_cache: dict = None, prompts_path: str = None):
    """
    Run one level for one model. Returns list of per-probe metric dicts.

    image_emb_cache: optional {path -> np.ndarray} of pre-encoded image
    embeddings. When supplied, image encoding is skipped entirely and the
    level reduces to text encoding + dot products. Build the cache once per
    model with `_encode_image_cache(model, all_image_paths)`.

    prompts_path: optional override prompt file (variant experiments). Swaps
    the text source while keeping modality/manifest/model-filtering unchanged.
    """
    use_per_image = (
        manifest_path is not None
        and has_image_probes(modality, level, prompts_path)
    )

    probes = get_probes_by_level(
        level, modality,
        manifest_path=manifest_path if use_per_image else None,
        prompts_path=prompts_path,
    )
    if not probes:
        return []

    cache = image_emb_cache if image_emb_cache is not None else {}
    results = []

    def _img_emb(path) -> np.ndarray:
        key = str(path)
        if key in cache:
            return cache[key]
        # Fallback (single-shot encode) — only hit if path isn't in the cache
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"  Warning: could not open {path}: {e}")
            return None
        e = model.encode_image([img]).cpu().numpy().squeeze()
        cache[key] = np.asarray(e, dtype=np.float32)
        return cache[key]

    if use_per_image:
        # One probe per image; evaluate each on its specific image
        for probe in tqdm(probes, desc=f"  L{level} (per-image)", leave=False):
            if not probe.image_path:
                continue
            img_path = _resolve_image(probe.image_path)
            img_emb = _img_emb(img_path)
            if img_emb is None:
                continue
            pos_emb = model.encode_text(probe.positive_texts).cpu().numpy()
            neg_emb = model.encode_text(probe.negative_texts).cpu().numpy()
            m = evaluate_probe(img_emb, pos_emb, neg_emb)
            m["probe_id"] = probe.probe_id
            m["image_path"] = str(img_path)
            m["disease_category"] = probe.metadata.get("disease_category", "")
            results.append(m)
    else:
        # Shared probes × all images. Pre-encode probe texts once per probe
        # rather than per (probe, image) pair to avoid wasted text encodes.
        for probe in tqdm(probes, desc=f"  L{level} probes", leave=False):
            pos_emb = model.encode_text(probe.positive_texts).cpu().numpy()
            neg_emb = model.encode_text(probe.negative_texts).cpu().numpy()
            for img_path in image_paths:
                img_emb = _img_emb(img_path)
                if img_emb is None:
                    continue
                m = evaluate_probe(img_emb, pos_emb, neg_emb)
                m["probe_id"] = probe.probe_id
                m["image_path"] = str(img_path)
                m["disease_category"] = probe.metadata.get("disease_category", "")
                results.append(m)

    return results


def _per_category_summary(level_results: list) -> dict:
    """Aggregate DS per disease_category for L4/L5 breakdowns."""
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in level_results:
        cat = r.get("disease_category") or "unknown"
        buckets[cat].append(r["discrimination_score"])
    return {cat: {"avg_DS": float(np.mean(vals)), "n": len(vals)}
            for cat, vals in sorted(buckets.items())}


def run_medtec(
    model_keys: list,
    levels: list,
    image_paths: list,
    modality: str = "cxr",
    manifest_path: str = None,
    device: str = "cuda",
    output_path: str = None,
    prompts_path: str = None,
):
    results = {}

    for model_key in model_keys:
        supported = MODEL_REGISTRY[model_key].get("modalities", [])
        if supported and modality not in supported:
            print(f"\nSkipping {MODEL_REGISTRY[model_key]['display_name']} "
                  f"(not designed for {modality})")
            continue

        print(f"\n{'='*60}")
        print(f"  {MODEL_REGISTRY[model_key]['display_name']}  [{modality}]")
        print(f"{'='*60}")

        try:
            model = load_model(model_key, device=device)
        except Exception as e:
            print(f"  Failed to load: {e}")
            continue

        results[model_key] = {"display_name": MODEL_REGISTRY[model_key]["display_name"]}

        # Pre-encode every image this model will see across all levels. Build a
        # superset over (a) the shared-probe image list and (b) any per-image
        # probe paths from the manifest, so per-level lookups never miss.
        all_paths = list(image_paths)
        if manifest_path:
            with open(manifest_path, encoding="utf-8") as f:
                _manifest = json.load(f)
            for entry in _manifest:
                p = entry.get("image_path") or entry.get("original_path")
                if p:
                    all_paths.append(str(_resolve_image(p)))
        if all_paths:
            print(f"  Encoding {len(set(all_paths))} unique images once "
                  f"(reused across L1-L6) …")
            image_emb_cache = _encode_image_cache(model, all_paths)
            print(f"  -> cached embeddings for {len(image_emb_cache)} images")
        else:
            image_emb_cache = {}

        for level in levels:
            level_results = run_level(
                model, level, modality, image_paths, manifest_path, REPO_ROOT,
                image_emb_cache=image_emb_cache, prompts_path=prompts_path,
            )
            if not level_results:
                continue

            avg_ds = float(np.mean([r["discrimination_score"] for r in level_results]))
            avg_cr = float(np.mean([r["collapse_ratio"] for r in level_results]))

            level_entry = {
                "avg_discrimination_score": avg_ds,
                "avg_collapse_ratio": avg_cr,
                "n_probes": len(level_results),
                "per_probe": level_results,
            }
            # Per-category breakdown for L4/L5
            if level in (4, 5):
                level_entry["by_category"] = _per_category_summary(level_results)

            results[model_key][f"L{level}"] = level_entry

            status = "COLLAPSE" if avg_ds <= 0 else ("Weak" if avg_ds < 0.05 else "OK")
            level_name = LEVEL_DEFINITIONS[level]["name"]
            print(f"  L{level} ({level_name:15s}): DS={avg_ds:+.4f}  CR={avg_cr:.3f}  "
                  f"n={len(level_results):>3}  [{status}]")

            if level in (4, 5) and "by_category" in level_entry:
                for cat, stats in level_entry["by_category"].items():
                    print(f"         {cat:<22}: DS={stats['avg_DS']:+.4f}  n={stats['n']}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if output_path:
        # Merge into existing file if present so subsequent per-model runs
        # against the same output don't clobber prior model entries. Caller
        # can still force a fresh write by deleting the file first.
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        merged = {}
        if Path(output_path).exists():
            try:
                with open(output_path, encoding="utf-8") as f:
                    merged = json.load(f)
            except Exception as e:
                print(f"  [warn] could not read existing {output_path}: {e}")
                merged = {}
        merged.update(results)
        with open(output_path, "w") as f:
            json.dump(merged, f, indent=2)
        print(f"\nResults saved to {output_path} (merged with {len(merged)} model entries)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run MedTEC-Bench")
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--modality", default="cxr", choices=SUPPORTED_MODALITIES)
    parser.add_argument("--levels", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    parser.add_argument(
        "--manifest", default=None,
        help="Path to data/<modality>_probes/manifest.json for per-image L4/L5 probes"
    )
    parser.add_argument(
        "--images", nargs="+", default=None,
        help="Image paths for shared-probe levels (L1-L3, L6). "
             "Defaults to the images/ subdirectory next to --manifest when provided."
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--prompts-file", default=None,
        help="Override prompt config (variant experiments, e.g. "
             "configs/prompts/variants/cxr_report.json). Keeps --modality for "
             "model filtering and the manifest; only swaps the text source."
    )
    args = parser.parse_args()

    # Resolve model list
    model_keys = list_models() if "all" in args.models else args.models
    model_keys = [k for k in model_keys if MODEL_REGISTRY[k].get("modalities") != []]

    # Resolve manifest
    manifest_path = None
    if args.manifest:
        manifest_path = str(REPO_ROOT / args.manifest
                            if not Path(args.manifest).is_absolute()
                            else args.manifest)
        if not Path(manifest_path).exists():
            print(f"ERROR: manifest not found: {manifest_path}")
            print(f"Run scripts/select_{args.modality}_probes.py first "
                  f"(or the closest equivalent selector script for this modality).")
            raise SystemExit(1)

    # Resolve image paths for shared-probe levels
    if args.images:
        image_paths = [str(p) for p in args.images]
    elif manifest_path:
        # Preferred: use the probe images directory next to the manifest.
        # Fallback: read image paths from the manifest entries themselves.
        # This supports manifests that reference images outside the probe
        # directory (e.g. data/openi_probes_full/manifest.json which points
        # to data/openi/images/...) without forcing a copy.
        probe_img_dir = Path(manifest_path).parent / "images"
        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        if probe_img_dir.exists():
            image_paths = sorted(str(p) for p in probe_img_dir.iterdir()
                                 if p.suffix.lower() in exts)
            print(f"Auto-discovered {len(image_paths)} images from {probe_img_dir}")
        else:
            with open(manifest_path, encoding="utf-8") as f:
                _manifest_entries = json.load(f)
            seen = set()
            image_paths = []
            for entry in _manifest_entries:
                p = entry.get("image_path") or entry.get("original_path")
                if not p or p in seen:
                    continue
                seen.add(p)
                image_paths.append(p)
            print(f"No images/ subdir at {probe_img_dir}; "
                  f"using {len(image_paths)} image paths from manifest entries.")
    else:
        print("ERROR: provide --images or --manifest")
        raise SystemExit(1)

    output = args.output or f"experiments/results/medtec_{args.modality}.json"

    print(f"Modality : {args.modality}")
    print(f"Models   : {model_keys}")
    print(f"Levels   : {args.levels}")
    print(f"Images   : {len(image_paths)} (shared probe levels)")
    print(f"Manifest : {manifest_path or 'none (fallback probes for L4/L5)'}")
    print(f"Prompts  : {args.prompts_file or 'default (per-modality config)'}")
    print(f"Output   : {output}")

    run_medtec(
        model_keys, args.levels, image_paths, args.modality,
        manifest_path, args.device, output, prompts_path=args.prompts_file,
    )


if __name__ == "__main__":
    main()
