"""
run_geometry.py — Text-embedding geometry analysis (E12).

For each model, on a corpus of {medical, non-medical} texts, compute:
  - Isotropy I          (min/max eigenvalue ratio of the covariance)
  - Effective dim D_eff (participation ratio of eigenvalues)
  - Hubness skewness    (k-occurrence of top-k neighbors over the corpus)
  - 2-D UMAP and t-SNE coords colored by {medical, non-medical}

These numbers provide the mechanistic explanation for MedTEC text collapse:
if medical text embeddings live in a narrow cone of the space (low I, small D_eff),
no amount of prompt design can make them discriminable.

Usage:
    python scripts/run_geometry.py                  # all models, default corpus
    python scripts/run_geometry.py --models biomedclip openclip_vitb32
    python scripts/run_geometry.py --skip-viz      # metrics only, no UMAP/t-SNE

Output:
    experiments/results/geometry_text.json      (metrics table)
    experiments/results/geometry_embeddings/    (2D coords + figures per model)
"""
import os
# Windows conda + torch/numpy + umap-learn/numba often link two OpenMP runtimes.
# Benign on x86_64 CPUs; must be set before numpy/torch import.
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
from metrics.collapse_metrics import (  # noqa: E402
    embedding_isotropy, effective_dimensionality, hubness_score,
)

RESULTS_DIR = REPO / "experiments" / "results"
EMB_DIR = RESULTS_DIR / "geometry_embeddings"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EMB_DIR.mkdir(parents=True, exist_ok=True)


# ── Text corpus ───────────────────────────────────────────────────────────────

def build_corpus(n_medical: int = 500, n_nonmedical: int = 500, seed: int = 42):
    """
    Medical half: pulls prompts from the existing MedTEC prompt configs
    (CXR / Derm / Brain MRI / Fundus / Abd-CT / Histopath — across all
    severity / location / negation / attribute variants).

    Non-medical half: generic English captions across {nature, cityscape,
    household, sport, animal, food, abstract concept}. No download required;
    built from templates so the corpus is fully reproducible.
    """
    import random
    rng = random.Random(seed)

    # ── Medical: scrape all prompt strings from configs/prompts/*.json ─────
    prompt_dir = REPO / "configs" / "prompts"
    medical = []
    if prompt_dir.exists():
        for f in sorted(prompt_dir.glob("*_prompts.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            # Recursively pull any string values from the prompt JSON
            def walk(x):
                if isinstance(x, str):
                    s = x.strip()
                    if 4 < len(s) < 300:
                        medical.append(s)
                elif isinstance(x, dict):
                    for v in x.values():
                        walk(v)
                elif isinstance(x, list):
                    for v in x:
                        walk(v)
            walk(data)
    medical = list(dict.fromkeys(medical))  # dedupe, preserve order
    if len(medical) > n_medical:
        medical = rng.sample(medical, n_medical)

    # ── Non-medical: generic captions built from templates ────────────────
    nm_templates = [
        "a photograph of a {} in a park",
        "a close-up of {}",
        "an illustration showing {}",
        "a digital drawing of {}",
        "the view from a {} on a sunny day",
        "a child holding a {}",
        "a table with {} on it",
        "a man and a woman walking near a {}",
        "a busy street with many {}",
        "a cat sitting on a {}",
        "a bird flying above a {}",
        "an old photograph of a {}",
        "a black-and-white sketch of {}",
        "a wooden {} by the window",
        "a painting of {}",
        "a crowd gathered around a {}",
    ]
    nm_nouns = [
        "apple", "bicycle", "tree", "mountain", "beach", "car", "dog", "cat",
        "house", "river", "bridge", "book", "guitar", "cup", "lantern",
        "boat", "flower", "train", "statue", "garden", "window", "table",
        "child", "woman", "man", "sunset", "cloud", "hat", "balloon",
        "pizza", "cake", "bowl", "fence", "tent", "dancer", "violin",
        "kite", "sandcastle", "candle", "clock", "umbrella", "kayak",
        "scooter", "hammock", "lighthouse", "windmill", "canyon", "waterfall",
    ]
    non_medical = []
    for tpl in nm_templates:
        for n in nm_nouns:
            non_medical.append(tpl.format(n))
    non_medical = list(dict.fromkeys(non_medical))
    if len(non_medical) > n_nonmedical:
        non_medical = rng.sample(non_medical, n_nonmedical)

    # Equalize sizes
    n = min(len(medical), len(non_medical))
    medical     = medical[:n]
    non_medical = non_medical[:n]

    texts  = medical + non_medical
    labels = ["medical"] * len(medical) + ["non_medical"] * len(non_medical)
    return texts, labels


# ── Batched text encoding (tolerant of wrappers) ──────────────────────────────

def encode_texts(model, texts, batch_size=64):
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="  txt", ncols=80):
        batch = texts[i : i + batch_size]
        with torch.no_grad():
            emb = model.encode_text(batch).float().cpu().numpy()
        embs.append(emb)
    E = np.concatenate(embs, axis=0)
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    E = E / np.clip(norms, 1e-8, None)
    return E


# ── Low-dim projection ────────────────────────────────────────────────────────

def project_2d(E, method="umap", seed=42):
    """Return (N, 2) array."""
    try:
        if method == "umap":
            import umap
            reducer = umap.UMAP(n_components=2, n_neighbors=20,
                                min_dist=0.1, metric="cosine", random_state=seed)
            return reducer.fit_transform(E)
        elif method == "tsne":
            from sklearn.manifold import TSNE
            return TSNE(n_components=2, metric="cosine",
                        perplexity=30, init="pca", random_state=seed).fit_transform(E)
    except Exception as e:
        print(f"    [{method} SKIP] {e}")
        return None


def save_figure(coords, labels, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    colors = {"medical": "#d62728", "non_medical": "#1f77b4"}
    for lab in ("non_medical", "medical"):  # medical on top
        mask = np.array([l == lab for l in labels])
        if mask.sum() == 0:
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   s=6, alpha=0.55, c=colors[lab], label=lab, edgecolors="none")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="best", frameon=True, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["all"])
    ap.add_argument("--n-medical", type=int, default=500)
    ap.add_argument("--n-nonmedical", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-viz", action="store_true",
                    help="Skip UMAP/t-SNE (metrics only).")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve models
    if "all" in args.models:
        model_keys = [k for k in MODEL_REGISTRY if MODEL_REGISTRY[k].get("modalities") != []]
    else:
        model_keys = args.models

    # Build corpus ONCE — every model sees the same texts
    texts, labels = build_corpus(args.n_medical, args.n_nonmedical, args.seed)
    n_med = labels.count("medical")
    n_nm  = labels.count("non_medical")
    print(f"\nCorpus: {n_med} medical + {n_nm} non-medical = {len(texts)} texts")

    out_json = RESULTS_DIR / "geometry_text.json"
    if out_json.exists() and not args.overwrite:
        with open(out_json, encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = {"n_medical": n_med, "n_nonmedical": n_nm, "seed": args.seed, "models": {}}

    for mkey in model_keys:
        cfg = MODEL_REGISTRY.get(mkey)
        if cfg is None:
            continue
        if mkey in results["models"] and not args.overwrite:
            print(f"  [SKIP] {mkey} already done.")
            continue

        print(f"\n{'='*60}\n  MODEL: {cfg.get('display_name', mkey)}\n{'='*60}")
        t0 = time.time()
        try:
            model = load_model(mkey, device=device)
        except Exception as e:
            print(f"  [LOAD FAIL] {e}")
            results["models"][mkey] = {"error": f"load: {e}"}
            continue

        try:
            E = encode_texts(model, texts)
        except Exception as e:
            print(f"  [ENCODE FAIL] {e}")
            results["models"][mkey] = {"error": f"encode: {e}"}
            del model; torch.cuda.empty_cache()
            continue

        E_med = E[np.array(labels) == "medical"]
        E_nm  = E[np.array(labels) == "non_medical"]

        # ── Geometry metrics (per subset + overall) ──────────────────────
        rec = {
            "display_name": cfg.get("display_name", mkey),
            "embedding_dim": int(E.shape[1]),
            "isotropy_all":    embedding_isotropy(E),
            "isotropy_med":    embedding_isotropy(E_med),
            "isotropy_nm":     embedding_isotropy(E_nm),
            "deff_all":        effective_dimensionality(E),
            "deff_med":        effective_dimensionality(E_med),
            "deff_nm":         effective_dimensionality(E_nm),
        }
        # hubness: k-occurrence across the full corpus
        sim = E @ E.T
        np.fill_diagonal(sim, -np.inf)
        rec["hubness_all"] = hubness_score(sim, k=5)
        rec["hubness_med"] = hubness_score(E_med @ E_med.T - np.eye(len(E_med))*1e9, k=5)
        rec["hubness_nm"]  = hubness_score(E_nm  @ E_nm.T  - np.eye(len(E_nm)) *1e9, k=5)

        # ── 2-D projections (optional) ───────────────────────────────────
        if not args.skip_viz:
            coords_u = project_2d(E, "umap", args.seed)
            coords_t = project_2d(E, "tsne", args.seed)
            model_dir = EMB_DIR / mkey
            model_dir.mkdir(exist_ok=True)
            if coords_u is not None:
                np.save(model_dir / "umap.npy", coords_u)
                save_figure(coords_u, labels, model_dir / "umap.png",
                            f"{cfg.get('display_name', mkey)} — UMAP (text)")
                rec["umap_path"] = str((model_dir / "umap.png").relative_to(REPO))
            if coords_t is not None:
                np.save(model_dir / "tsne.npy", coords_t)
                save_figure(coords_t, labels, model_dir / "tsne.png",
                            f"{cfg.get('display_name', mkey)} — t-SNE (text)")
                rec["tsne_path"] = str((model_dir / "tsne.png").relative_to(REPO))

        rec["time_sec"] = round(time.time() - t0, 1)
        results["models"][mkey] = rec

        print(f"  I(all)={rec['isotropy_all']:.4f}  I(med)={rec['isotropy_med']:.4f}  "
              f"I(nm)={rec['isotropy_nm']:.4f}")
        print(f"  D_eff(all)={rec['deff_all']:.1f}  D_eff(med)={rec['deff_med']:.1f}  "
              f"D_eff(nm)={rec['deff_nm']:.1f}")
        print(f"  hubness(all)={rec['hubness_all']:+.2f}   ({rec['time_sec']}s)")

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        del model; torch.cuda.empty_cache()

    # Summary table
    print("\n" + "="*90)
    print(f"  GEOMETRY SUMMARY  ({n_med} medical / {n_nm} non-medical texts)")
    print("="*90)
    hdr = f"  {'Model':<22}  {'I(med)':>8}  {'I(nm)':>8}  {'D_eff(med)':>11}  {'D_eff(nm)':>11}  {'hub':>6}"
    print(hdr)
    print("  " + "-"*(len(hdr)-2))
    for mkey, r in results["models"].items():
        if "error" in r:
            print(f"  {r.get('display_name', mkey):<22}   ERROR")
            continue
        print(f"  {r['display_name']:<22}  {r['isotropy_med']:>8.4f}  {r['isotropy_nm']:>8.4f}  "
              f"{r['deff_med']:>11.1f}  {r['deff_nm']:>11.1f}  {r['hubness_all']:>+6.2f}")
    print("="*90)
    print(f"  Saved: {out_json}")
    if not args.skip_viz:
        print(f"  Figures: {EMB_DIR}/<model>/umap.png , tsne.png")


if __name__ == "__main__":
    main()
