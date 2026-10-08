"""
Phase 3 prerequisite — text-encoder geometry per model.

For each model: re-encode the same 2000 ROCO captions used in Task 1.3 (read
from the pinned pool.json), cache the (N, D) text embeddings to disk, then
compute D_eff (participation-ratio effective dimensionality), isotropy ratio,
and hubness skewness. These feed the LIMIT-bound gap table in Phase 3.2.

Why ROCO captions: the LIMIT bound is about whether the encoder geometry has
room to discriminate fine-grained medical concepts. Measuring D_eff on the
same medical caption distribution that the L4 probes interrogate is the
right reference; general-corpus D_eff would be higher and irrelevant to the
medical-text failure mode.

Outputs:
    experiments/results/retrieval_cache/roco_seed42_n2000/{model}/text_emb.npy
    experiments/results/phase3_text_geometry.csv
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "src"))
from models.model_registry import load_model, MODEL_REGISTRY  # noqa: E402
from metrics.collapse_metrics import hubness_score  # noqa: E402  # numpy argsort-based; safe


def _torch_eigvals_centered(X: np.ndarray) -> np.ndarray:
    """Eigenvalues of cov(X.T) via torch SVD on the centered matrix.
    numpy.linalg.eigvalsh + numpy.cov both call BLAS routines that hard-crash
    in this env (same root cause as the numpy matmul + matplotlib savefig
    issues — see env_matplotlib_savefig memory). Torch's eigendecomposition
    uses its own LAPACK and is unaffected. For an (N, D) embedding matrix
    with N > D, the nonzero eigenvalues of cov(X.T) equal the squared
    singular values of (X - mean) divided by (N-1)."""
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

CACHE_DIR = REPO / "experiments" / "results" / "retrieval_cache" / "roco_seed42_n2000"
RESULTS = REPO / "experiments" / "results"


def encode_texts_batched(model, texts, batch_size=64, desc="txt"):
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc=desc, ncols=80):
        batch = texts[i : i + batch_size]
        with torch.no_grad():
            emb = model.encode_text(batch).float().cpu().numpy()
        embs.append(emb)
    E = np.concatenate(embs, axis=0)
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    return E / np.clip(norms, 1e-8, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["all"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-encode even if text_emb.npy already exists.")
    args = ap.parse_args()

    pool = json.load(open(CACHE_DIR / "pool.json", encoding="utf-8"))
    captions = [it["caption"] for it in pool["items"]]
    print(f"Pool: {pool['dataset']}, n={pool['n_samples']}, seed={pool['seed']}")

    if "all" in args.models:
        model_keys = [k for k in MODEL_REGISTRY if MODEL_REGISTRY[k].get("modalities") != []]
    else:
        model_keys = args.models

    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = []
    for mkey in model_keys:
        cfg = MODEL_REGISTRY.get(mkey)
        if cfg is None:
            print(f"\n  [SKIP] unknown {mkey}")
            continue
        model_dir = CACHE_DIR / mkey
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
                print(f"  [LOAD FAIL] {e}")
                continue
            try:
                te = encode_texts_batched(model, captions, args.batch_size)
            except Exception as e:
                import traceback; traceback.print_exc()
                del model; torch.cuda.empty_cache()
                continue
            np.save(emb_path, te.astype(np.float32))
            print(f"  [ENC] shape={te.shape}  saved -> {emb_path.relative_to(REPO)}  ({time.time()-t0:.1f}s)")
            del model
            torch.cuda.empty_cache()

        # Geometry stats. embedding_isotropy and effective_dimensionality use
        # numpy.linalg.eigvalsh internally — that's a small matrix (D up to 768)
        # and on this machine it does not trigger the BLAS+OpenMP crash that
        # plagues the larger numpy `@` matmuls. Verify per model.
        try:
            iso = embedding_isotropy_safe(te)
            d_eff = effective_dimensionality_safe(te)
        except Exception as e:
            print(f"  [GEO FAIL] {e}")
            iso = float("nan"); d_eff = float("nan")

        # Hubness needs an image embedding; reuse the cached sim matrix
        # (i.e., k-occurrences of texts among image NN searches).
        sim_path = model_dir / "sim.npy"
        if sim_path.exists():
            sim = np.load(sim_path)
            try:
                hub = hubness_score(sim, k=5)
            except Exception as e:
                print(f"  [HUB FAIL] {e}")
                hub = float("nan")
        else:
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
            "hubness_skew_k5": hub,
        })
        print(f"  D={D}  D_eff={d_eff:.2f}  ({100*d_eff/D:.1f}% of D)  "
              f"isotropy={iso:.3e}  hub_skew={hub:.3f}")

    # Persist
    out_csv = RESULTS / "phase3_text_geometry.csv"
    if rows:
        import csv
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
