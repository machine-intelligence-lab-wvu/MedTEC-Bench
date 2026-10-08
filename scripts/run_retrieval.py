"""
run_retrieval.py — Image-text retrieval benchmark on ROCO (or any image-caption dataset).

Computes R@1/5/10/50/100, medPct, median/mean rank for each model. These numbers
are the x-axis of the "retrieval vs. collapse" scatter plot that is the paper's
central figure (and the input to Phase 1 Task 1.5 bootstrap correlations).

Usage:
    python scripts/run_retrieval.py                     # all models, N=2000 (default)
    python scripts/run_retrieval.py --n-samples 1000    # faster, N=1000
    python scripts/run_retrieval.py --all               # full 8176 pairs
    python scripts/run_retrieval.py --models biomedclip openclip_vitb32
    python scripts/run_retrieval.py --dataset roco      # only one supported so far

Outputs:
    experiments/results/retrieval_{dataset}.json                              # scalar metrics
    experiments/results/retrieval_cache/{dataset}_seed{S}_n{N}/pool.json      # pool definition
    experiments/results/retrieval_cache/{dataset}_seed{S}_n{N}/{model}/sim.npy  # (N,N) float32
"""
import os
# Windows conda + torch/numpy + umap-learn often link against two OpenMP runtimes
# (libomp + libiomp5md). Known benign on Intel CPUs; set before numpy/torch import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "src"))
from models.model_registry import load_model, MODEL_REGISTRY  # noqa: E402

DATA_DIRS = {
    "roco":     REPO / "data" / "roco",
    "roco_cxr": REPO / "data" / "roco_cxr",
    "openi":    REPO / "data" / "openi",
}
RESULTS_DIR = REPO / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_ROOT = RESULTS_DIR / "retrieval_cache"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)


# ── Dataset loader ────────────────────────────────────────────────────────────

def load_dataset(name: str, n_samples: int | None, seed: int = 42):
    """Return list of (image_path: Path, caption: str, id: str)."""
    root = DATA_DIRS[name]
    captions_file = root / "captions.json"
    if not captions_file.exists():
        raise FileNotFoundError(f"Missing {captions_file}. Run scripts/download_{name}.py first.")
    with open(captions_file, encoding="utf-8") as f:
        items = json.load(f)

    # Filter: keep only entries with non-empty caption AND existing image
    filtered = []
    for x in items:
        cap = (x.get("caption") or "").strip()
        if not cap:
            continue
        img_path = root / x["image"]
        if not img_path.exists():
            continue
        filtered.append((img_path, cap, x["id"]))

    if n_samples is not None and n_samples < len(filtered):
        rng = random.Random(seed)
        filtered = rng.sample(filtered, n_samples)

    return filtered


# ── Batched encoding ──────────────────────────────────────────────────────────

def encode_images_batched(model, image_paths, batch_size=32, desc="imgs"):
    """Return L2-normalized (N, D) float32 numpy array."""
    embs = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc=desc, ncols=80):
        batch_paths = image_paths[i : i + batch_size]
        batch = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                batch.append(img)
            except Exception:
                # Use a black image as a silent stand-in; rare.
                batch.append(Image.new("RGB", (224, 224)))
        with torch.no_grad():
            emb = model.encode_image(batch).float().cpu().numpy()
        embs.append(emb)
    E = np.concatenate(embs, axis=0)
    # Ensure L2 norm (most wrappers already do this, but be safe)
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    E = E / np.clip(norms, 1e-8, None)
    return E


def encode_texts_batched(model, texts, batch_size=64, desc="txts"):
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc=desc, ncols=80):
        batch = texts[i : i + batch_size]
        with torch.no_grad():
            emb = model.encode_text(batch).float().cpu().numpy()
        embs.append(emb)
    E = np.concatenate(embs, axis=0)
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    E = E / np.clip(norms, 1e-8, None)
    return E


# ── Retrieval metrics ─────────────────────────────────────────────────────────

def compute_retrieval_metrics(img_emb: np.ndarray, txt_emb: np.ndarray):
    """
    Given matched pairs (i-th image ↔ i-th caption), compute both directions:
      i2t (image retrieves caption) — standard for ROCO
      t2i (caption retrieves image)
    Returns a dict including the full (N,N) similarity matrix under "sim" so
    callers can persist it for downstream bootstrap analyses (Task 1.5).
    """
    # Use torch for the matmul: numpy's BLAS hard-crashes on this Windows
    # build when torch+numpy load duplicate OpenMP runtimes. KMP_DUPLICATE_LIB_OK
    # silences the warning but the crash on large matmuls remains.
    sim_t = torch.from_numpy(img_emb) @ torch.from_numpy(txt_emb).T
    sim = sim_t.numpy().astype(np.float32, copy=False)   # (N, N), float32
    N = sim.shape[0]

    # image → text
    # rank of the correct caption for each image: how many captions score strictly higher?
    diag = np.diag(sim)
    ranks_i2t = (sim > diag[:, None]).sum(axis=1)   # 0 = best
    r_at = lambda ranks, k: float(np.mean(ranks < k))
    K_VALUES = [1, 5, 10, 50, 100]
    i2t = {f"R@{k}": r_at(ranks_i2t, k) for k in K_VALUES if k <= N}
    i2t["median_rank"] = float(np.median(ranks_i2t)) + 1.0
    i2t["mean_rank"]   = float(np.mean(ranks_i2t))   + 1.0
    # Median percentile rank (higher is better; dataset-size invariant)
    i2t["medPct"] = float(1.0 - i2t["median_rank"] / N)

    # text → image
    ranks_t2i = (sim.T > diag[:, None]).sum(axis=1)
    t2i = {f"R@{k}": r_at(ranks_t2i, k) for k in K_VALUES if k <= N}
    t2i["median_rank"] = float(np.median(ranks_t2i)) + 1.0
    t2i["mean_rank"]   = float(np.mean(ranks_t2i))   + 1.0
    t2i["medPct"] = float(1.0 - t2i["median_rank"] / N)

    return {"i2t": i2t, "t2i": t2i, "n": N, "sim": sim}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="roco", choices=list(DATA_DIRS.keys()))
    ap.add_argument("--n-samples", type=int, default=2000,
                    help="Random subset for speed+variance. Default 2000. Use --all to disable.")
    ap.add_argument("--all", action="store_true", help="Use the full dataset (overrides --n-samples).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", nargs="+", default=["all"],
                    help="Model keys (from MODEL_REGISTRY), or 'all'.")
    ap.add_argument("--batch-size-img", type=int, default=32)
    ap.add_argument("--batch-size-txt", type=int, default=64)
    ap.add_argument("--out", default=None, help="Override output path.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-cache-sim", action="store_true",
                    help="Skip writing the (N,N) similarity matrix per model.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_samples = None if args.all else args.n_samples

    # Resolve models
    if "all" in args.models:
        model_keys = [k for k in MODEL_REGISTRY if MODEL_REGISTRY[k].get("modalities") != []]
    else:
        model_keys = args.models

    # Load data once (same split for all models — essential for fair comparison)
    items = load_dataset(args.dataset, n_samples, args.seed)
    image_paths = [x[0] for x in items]
    captions   = [x[1] for x in items]
    ids        = [x[2] for x in items]
    print(f"\nRetrieval on {args.dataset!r}: {len(items)} image-caption pairs (seed={args.seed})")

    # Cache directory for similarity matrices and the pinned pool definition.
    cache_dir = CACHE_ROOT / f"{args.dataset}_seed{args.seed}_n{len(items)}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    pool_meta_path = cache_dir / "pool.json"
    if not pool_meta_path.exists() or args.overwrite:
        pool_meta = {
            "dataset": args.dataset,
            "seed": args.seed,
            "n_samples": len(items),
            "items": [
                {"id": ids[i],
                 "image_path": str(image_paths[i].relative_to(REPO)).replace("\\", "/"),
                 "caption": captions[i]}
                for i in range(len(items))
            ],
        }
        with open(pool_meta_path, "w", encoding="utf-8") as f:
            json.dump(pool_meta, f, ensure_ascii=False, indent=2)
        print(f"  Pool definition pinned: {pool_meta_path}")

    out_path = Path(args.out) if args.out else RESULTS_DIR / f"retrieval_{args.dataset}.json"
    if out_path.exists() and not args.overwrite:
        with open(out_path, encoding="utf-8") as f:
            all_results = json.load(f)
        print(f"Appending to existing {out_path.name} (use --overwrite to start fresh).")
    else:
        all_results = {
            "dataset": args.dataset,
            "n_samples": len(items),
            "seed": args.seed,
            "sample_ids": ids,
            "models": {},
        }

    for mkey in model_keys:
        cfg = MODEL_REGISTRY.get(mkey)
        if cfg is None:
            print(f"\n  [SKIP] unknown model key: {mkey}")
            continue
        if mkey in all_results["models"] and not args.overwrite:
            print(f"\n  [SKIP] {mkey} already done.")
            continue

        print(f"\n{'='*60}\n  MODEL: {cfg.get('display_name', mkey)}\n{'='*60}")
        t0 = time.time()
        try:
            model = load_model(mkey, device=device)
        except Exception as e:
            print(f"  [LOAD FAIL] {e}")
            all_results["models"][mkey] = {"error": f"load: {e}"}
            continue

        try:
            img_emb = encode_images_batched(model, image_paths, args.batch_size_img, "  img")
            txt_emb = encode_texts_batched(model, captions,     args.batch_size_txt, "  txt")
            metrics = compute_retrieval_metrics(img_emb, txt_emb)
        except Exception as e:
            import traceback; traceback.print_exc()
            all_results["models"][mkey] = {"error": f"encode: {e}"}
            # Free GPU
            del model; torch.cuda.empty_cache()
            continue

        dt = time.time() - t0
        i2t_m = metrics["i2t"]
        print(f"  [OK] i2t R@1={i2t_m['R@1']:.3f}  R@5={i2t_m['R@5']:.3f}  "
              f"R@10={i2t_m['R@10']:.3f}  R@50={i2t_m.get('R@50', float('nan')):.3f}  "
              f"R@100={i2t_m.get('R@100', float('nan')):.3f}  "
              f"medPct={i2t_m['medPct']:.3f}   ({dt:.1f}s)")

        # Cache the similarity matrix for downstream bootstrap (Task 1.5).
        sim_cached_path = None
        if not args.no_cache_sim:
            model_cache_dir = cache_dir / mkey
            model_cache_dir.mkdir(parents=True, exist_ok=True)
            sim_cached_path = model_cache_dir / "sim.npy"
            np.save(sim_cached_path, metrics["sim"])
            print(f"  [CACHE] sim -> {sim_cached_path.relative_to(REPO)} "
                  f"({metrics['sim'].nbytes / 1e6:.1f} MB)")

        all_results["models"][mkey] = {
            "display_name": cfg.get("display_name", mkey),
            "n": metrics["n"],
            "i2t": {k: v for k, v in metrics["i2t"].items()},
            "t2i": {k: v for k, v in metrics["t2i"].items()},
            "time_sec": dt,
            "sim_cache": (str(sim_cached_path.relative_to(REPO)).replace("\\", "/")
                          if sim_cached_path else None),
        }

        # Save after every model so partial results survive a crash.
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        # Free memory before next model.
        del model, metrics
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*80)
    print(f"  RETRIEVAL SUMMARY — {args.dataset}  (n={len(items)})")
    print("="*80)
    print(f"  {'Model':<22}  {'R@1':>6}  {'R@5':>6}  {'R@10':>6}  "
          f"{'R@50':>6}  {'R@100':>6}  {'medPct':>6}  {'medR':>6}")
    print("  " + "-"*84)
    for mkey, r in all_results["models"].items():
        if "error" in r:
            print(f"  {r.get('display_name', mkey):<22}   ERROR: {r['error'][:40]}")
            continue
        m = r["i2t"]
        print(f"  {r['display_name']:<22}  {m['R@1']:.3f}   {m['R@5']:.3f}   "
              f"{m['R@10']:.3f}   {m.get('R@50', float('nan')):.3f}   "
              f"{m.get('R@100', float('nan')):.3f}   {m['medPct']:.3f}   "
              f"{m['median_rank']:>5.1f}")
    print("="*80)
    print(f"  Saved: {out_path}")
    print(f"  Sim cache: {cache_dir.relative_to(REPO) if not args.no_cache_sim else '(disabled)'}")


if __name__ == "__main__":
    main()
