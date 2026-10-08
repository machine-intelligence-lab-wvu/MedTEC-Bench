"""
Phase 3 robustness — text-encoder geometry on OpenI radiology reports.

Companion to scripts/_phase3_text_geometry.py: same recipe, but encodes the
2,000 OpenI image-report pairs (sampled deterministically from
data/openi/captions.json with seed=42) instead of the ROCO pool. Lets us
compare D_eff / isotropy / hubness across two very different caption
distributions (ROCO mixed-modality short captions vs OpenI long structured
chest-radiograph reports). The expectation: encoder geometry should be
reasonably stable across pools if it is a property of the encoder, and
diverge meaningfully if it is a property of the input text distribution.

Outputs:
    experiments/results/retrieval_cache/openi_seed42_n2000/pool.json
    experiments/results/retrieval_cache/openi_seed42_n2000/{model}/text_emb.npy
    experiments/results/phase3_openi_text_geometry.csv
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "src"))
from models.model_registry import load_model, MODEL_REGISTRY  # noqa: E402
from metrics.collapse_metrics import hubness_score  # noqa: E402

CACHE_ROOT = REPO / "experiments" / "results" / "retrieval_cache"
RESULTS = REPO / "experiments" / "results"


def _torch_eigvals_centered(X: np.ndarray) -> np.ndarray:
    """Eigenvalues of cov(X.T) via torch SVD on the centered matrix.
    See _phase3_text_geometry.py for why we route through torch."""
    Xc = X - X.mean(axis=0, keepdims=True)
    Xt = torch.from_numpy(Xc.astype(np.float64))
    s = torch.linalg.svdvals(Xt)
    eig = (s ** 2 / max(X.shape[0] - 1, 1)).cpu().numpy()
    return eig[eig > 1e-10]


def embedding_isotropy_safe(emb: np.ndarray) -> float:
    eig = _torch_eigvals_centered(emb)
    if len(eig) < 2:
        return 0.0
    return float(eig.min() / eig.max())


def effective_dimensionality_safe(emb: np.ndarray) -> float:
    eig = _torch_eigvals_centered(emb)
    if len(eig) == 0:
        return 0.0
    s1 = float(eig.sum()); s2 = float((eig ** 2).sum())
    return (s1 * s1) / s2 if s2 > 1e-20 else 0.0


def encode_texts_batched(model, texts, batch_size=64):
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="  txt", ncols=80):
        batch = texts[i : i + batch_size]
        with torch.no_grad():
            emb = model.encode_text(batch).float().cpu().numpy()
        embs.append(emb)
    E = np.concatenate(embs, axis=0)
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    return E / np.clip(norms, 1e-8, None)


def load_openi_pool(n_samples: int, seed: int):
    """Sample n_samples (id, caption) pairs from data/openi/captions.json."""
    src = REPO / "data" / "openi" / "captions.json"
    with open(src, encoding="utf-8") as f:
        items = json.load(f)
    filtered = [(x["id"], (x.get("caption") or "").strip())
                for x in items if (x.get("caption") or "").strip()]
    if n_samples and n_samples < len(filtered):
        rng = random.Random(seed)
        filtered = rng.sample(filtered, n_samples)
    return filtered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["all"])
    ap.add_argument("--n-samples", type=int, default=2000,
                    help="OpenI sample size. Default 2000 to match the ROCO Phase 3 cache for fair comparison.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    items = load_openi_pool(args.n_samples, args.seed)
    captions = [c for _, c in items]
    ids = [i for i, _ in items]
    print(f"OpenI pool: n={len(items)} (seed={args.seed})")

    cache_dir = CACHE_ROOT / f"openi_seed{args.seed}_n{len(items)}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    pool_path = cache_dir / "pool.json"
    if not pool_path.exists() or args.overwrite:
        with open(pool_path, "w", encoding="utf-8") as f:
            json.dump({
                "dataset": "openi", "seed": args.seed, "n_samples": len(items),
                "items": [{"id": i, "caption": c} for i, c in items],
            }, f, ensure_ascii=False, indent=2)
        print(f"  Pool pinned: {pool_path}")

    if "all" in args.models:
        model_keys = [k for k in MODEL_REGISTRY if MODEL_REGISTRY[k].get("modalities") != []]
    else:
        model_keys = args.models

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for mkey in model_keys:
        cfg = MODEL_REGISTRY.get(mkey)
        if cfg is None:
            print(f"\n  [SKIP] unknown {mkey}"); continue
        model_dir = cache_dir / mkey
        model_dir.mkdir(parents=True, exist_ok=True)
        emb_path = model_dir / "text_emb.npy"

        print(f"\n{'='*60}\n  {cfg.get('display_name', mkey)}\n{'='*60}")
        if emb_path.exists() and not args.overwrite:
            te = np.load(emb_path)
            print(f"  [CACHED] {emb_path.relative_to(REPO)}  shape={te.shape}")
        else:
            t0 = time.time()
            try:
                model = load_model(mkey, device=device)
            except Exception as e:
                print(f"  [LOAD FAIL] {e}"); continue
            try:
                te = encode_texts_batched(model, captions, args.batch_size)
            except Exception as e:
                import traceback; traceback.print_exc()
                del model; torch.cuda.empty_cache(); continue
            np.save(emb_path, te.astype(np.float32))
            print(f"  [ENC] shape={te.shape}  saved -> {emb_path.relative_to(REPO)}  ({time.time()-t0:.1f}s)")
            del model; torch.cuda.empty_cache()

        try:
            iso = embedding_isotropy_safe(te)
            d_eff = effective_dimensionality_safe(te)
        except Exception as e:
            print(f"  [GEO FAIL] {e}")
            iso = float("nan"); d_eff = float("nan")

        # Hubness on OpenI sim is NOT directly available — we don't have a (image, text)
        # OpenI sim cached for all 11 models in the Phase-3 cache. Compute hubness from
        # the cached retrieval_openi.json sim if available, else skip. Actually
        # retrieval_openi.json doesn't store the sim matrix. Compute hubness directly
        # from the text embeddings as text-text k-occurrence skewness (a different but
        # related notion: which texts get over-retrieved by other texts).
        # Route the (N,N) text-text matmul through torch — numpy's BLAS path
        # hard-crashes on this Windows env (same root cause as run_retrieval.py).
        _t = torch.from_numpy(te.astype(np.float32))
        text_text_sim = (_t @ _t.T).numpy()
        np.fill_diagonal(text_text_sim, -np.inf)
        try:
            hub = hubness_score(text_text_sim, k=5)
        except Exception as e:
            print(f"  [HUB FAIL] {e}")
            hub = float("nan")

        D = te.shape[1]
        rows.append({
            "model": mkey,
            "display_name": cfg.get("display_name", mkey),
            "embedding_dim": D,
            "n_captions": te.shape[0],
            "isotropy_min_max": iso,
            "D_eff": d_eff,
            "D_eff_over_D": d_eff / D if D else float("nan"),
            "hubness_skew_k5_text_text": hub,
        })
        print(f"  D={D}  D_eff={d_eff:.2f}  ({100*d_eff/D:.1f}% of D)  "
              f"isotropy={iso:.3e}  hub_skew(t-t)={hub:.3f}")

    out_csv = RESULTS / "phase3_openi_text_geometry.csv"
    if rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
