"""
run_openi_zeroshot.py — Zero-shot multi-label classification on OpenI / IU-CXR.

Mirrors run_nih_zeroshot.py but uses the OpenI image set with text-mined
CheXpert-5 labels (data/openi/labels_chexpert5.json). Same prompt ensemble
(3-pos / 3-neg) so AUROC is comparable to the NIH run.

Output:
    experiments/results/zeroshot_openi.json
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
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "src"))
from models.model_registry import load_model, MODEL_REGISTRY  # noqa: E402

OPENI_ROOT = REPO / "data" / "openi"
LABELS_PATH = OPENI_ROOT / "labels_chexpert5.json"
RESULTS_DIR = REPO / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Match NIH-runner ordering so comparable AUROC tables align.
FINDINGS = ["Atelectasis", "Cardiomegaly", "Effusion", "Consolidation", "Edema"]
LABEL_KEY = {
    "Atelectasis":   "atelectasis",
    "Cardiomegaly":  "cardiomegaly",
    "Effusion":      "effusion",
    "Consolidation": "consolidation",
    "Edema":         "edema",
}
PROMPT_TERMS = {
    "Atelectasis":   "atelectasis",
    "Cardiomegaly":  "cardiomegaly",
    "Effusion":      "pleural effusion",
    "Consolidation": "consolidation",
    "Edema":         "pulmonary edema",
}
POS_TEMPLATES = [
    "a chest x-ray showing {}",
    "findings consistent with {}",
    "radiograph demonstrating {}",
]
NEG_TEMPLATES = [
    "a chest x-ray without {}",
    "no evidence of {}",
    "{} is absent",
]


def load_openi_subset(n_samples: int | None, seed: int = 42):
    with open(LABELS_PATH, encoding="utf-8") as f:
        d = json.load(f)
    label_map = d["labels"]

    rows = []
    for uid, lab in label_map.items():
        img_path = OPENI_ROOT / "images" / f"{uid}.png"
        if not img_path.exists():
            continue
        y = [int(lab.get(LABEL_KEY[f], 0)) for f in FINDINGS]
        rows.append((img_path, y, uid))

    if n_samples is not None and n_samples < len(rows):
        import random
        rng = random.Random(seed)
        rows = rng.sample(rows, n_samples)

    image_paths = [r[0] for r in rows]
    labels = np.array([r[1] for r in rows], dtype=np.int64)
    ids = [r[2] for r in rows]
    return image_paths, labels, ids


def encode_images(model, image_paths, batch_size=32):
    embs = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="  img", ncols=80):
        batch_paths = image_paths[i : i + batch_size]
        batch = []
        for p in batch_paths:
            try:
                batch.append(Image.open(p).convert("RGB"))
            except Exception:
                batch.append(Image.new("RGB", (224, 224)))
        with torch.no_grad():
            emb = model.encode_image(batch).float().cpu().numpy()
        embs.append(emb)
    E = np.concatenate(embs, axis=0)
    E = E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-8, None)
    return E


def encode_texts(model, texts):
    with torch.no_grad():
        emb = model.encode_text(texts).float().cpu().numpy()
    emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8, None)
    return emb


def evaluate_model(model, image_paths, labels, batch_size=32):
    img_emb = encode_images(model, image_paths, batch_size)
    per_finding = {}
    aurocs = []
    for k, f in enumerate(FINDINGS):
        term = PROMPT_TERMS[f]
        pos_emb = encode_texts(model, [t.format(term) for t in POS_TEMPLATES])
        neg_emb = encode_texts(model, [t.format(term) for t in NEG_TEMPLATES])
        s_pos = (img_emb @ pos_emb.T).mean(axis=1)
        s_neg = (img_emb @ neg_emb.T).mean(axis=1)
        scores = s_pos - s_neg
        y = labels[:, k]
        n_pos = int(y.sum())
        prev = float(n_pos / len(y))
        if n_pos < 5 or n_pos > len(y) - 5:
            auroc = float("nan")
        else:
            auroc = float(roc_auc_score(y, scores))
        per_finding[f] = {"auroc": auroc, "n_pos": n_pos, "prevalence": prev}
        if not np.isnan(auroc):
            aurocs.append(auroc)
    return {
        "per_finding": per_finding,
        "macro_auroc": float(np.mean(aurocs)) if aurocs else float("nan"),
        "n_images": int(len(image_paths)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=None,
                    help="Subsample size. Default: use all labelled OpenI frontals.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", nargs="+", default=["all"])
    ap.add_argument("--batch-size-img", type=int, default=32)
    ap.add_argument("--out", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if "all" in args.models:
        model_keys = [k for k, v in MODEL_REGISTRY.items() if v.get("modalities") != []]
    else:
        model_keys = args.models

    image_paths, labels, ids = load_openi_subset(args.n_samples, args.seed)
    print(f"\nOpenI zero-shot: {len(image_paths)} images, findings={FINDINGS}")
    print("  per-finding positive counts: " +
          ", ".join(f"{f}={int(labels[:, k].sum())}" for k, f in enumerate(FINDINGS)))

    out_path = Path(args.out) if args.out else RESULTS_DIR / "zeroshot_openi.json"
    if out_path.exists() and not args.overwrite:
        with open(out_path, encoding="utf-8") as f:
            all_results = json.load(f)
        print(f"Appending to existing {out_path.name} (use --overwrite to start fresh).")
    else:
        all_results = {
            "dataset": "openi_iucxr",
            "n_samples": len(image_paths),
            "seed": args.seed,
            "findings": FINDINGS,
            "sample_ids": ids,
            "models": {},
        }

    for mkey in model_keys:
        cfg = MODEL_REGISTRY.get(mkey)
        if cfg is None:
            print(f"\n  [SKIP] unknown model: {mkey}")
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
            r = evaluate_model(model, image_paths, labels, args.batch_size_img)
        except Exception as e:
            import traceback; traceback.print_exc()
            all_results["models"][mkey] = {"error": f"encode: {e}"}
            del model; torch.cuda.empty_cache()
            continue

        dt = time.time() - t0
        r["display_name"] = cfg.get("display_name", mkey)
        r["time_sec"] = dt
        all_results["models"][mkey] = r
        print(f"  [OK] macro AUROC = {r['macro_auroc']:.3f}   ({dt:.1f}s)")
        for f in FINDINGS:
            pf = r["per_finding"][f]
            auroc_s = f"{pf['auroc']:.3f}" if not np.isnan(pf["auroc"]) else "   nan"
            print(f"      {f:<15} AUROC={auroc_s}  (n_pos={pf['n_pos']}, prev={pf['prevalence']:.3f})")

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        del model; torch.cuda.empty_cache()

    print("\n" + "="*80)
    print(f"  OPENI ZERO-SHOT SUMMARY  (n={len(image_paths)})")
    print("="*80)
    header = "  " + f"{'Model':<22}  " + "  ".join(f"{f[:6]:>7}" for f in FINDINGS) + f"  {'macro':>7}"
    print(header)
    print("  " + "-"*(len(header) - 2))
    for mkey, r in all_results["models"].items():
        if "error" in r:
            print(f"  {r.get('display_name', mkey):<22}   ERROR: {r['error'][:40]}")
            continue
        row = f"  {r['display_name']:<22}"
        for f in FINDINGS:
            a = r["per_finding"][f]["auroc"]
            row += f"  {a:>7.3f}" if not np.isnan(a) else f"  {'nan':>7}"
        row += f"  {r['macro_auroc']:>7.3f}"
        print(row)
    print("="*80)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
