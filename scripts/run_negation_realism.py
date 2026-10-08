"""
Exp C — Negation realism (rebuttal). Dedicated runner for the certainty-ladder
analysis that does NOT fit run_medtec's per-image `disease_prompts[own-category]`
path, because it scores a finding-ABSENT image (e.g. a normal CXR) against a
finding's full presence->absence ladder.

For each finding F and each anchor image I:
  * per-rung mean image-text similarity s_r  (rungs ordered present..absent)
  * (B) monotonicity: Spearman(s_r, asserted-presence ordinal) per image
        - present anchors (F truly present)  -> ideal rho = +1
        - absent  anchors (F truly absent)   -> ideal rho = -1
        - collapsed model -> rho ~ 0 on BOTH
  * (B) dynamic range: delta = s_present - s_absent
  * (A) per-subtype DS on PRESENT anchors (continuation of paper L5):
        DS_sub = max_sim(present paraphrases) - max_sim(subtype paraphrases)
        subtypes: definite / hedged / ruled_out / historical (subtype_map in cfg)

Ladder config: configs/prompts/variants/<modality>_negation_ladder.json
Manifest     : data/<modality>_probes_n1000/manifest.json (list of entries with
               disease_category, image_path/original_path, image_id).

Usage:
    python scripts/run_negation_realism.py --modality cxr \
        --ladder configs/prompts/variants/lit/cxr_negation_ladder_lit.json \
        --manifest data/cxr_probes_n1000/manifest.json \
        --output experiments/results/negrealism/cxr_negation_realism.json
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from models.model_registry import load_model, list_models, MODEL_REGISTRY

# Rungs pooled into the single 'absent' point for the B ladder.
_ABSENT_RUNGS = ("absent_plain", "absent_evidential")
# The 5-point ladder used for monotonicity (present..absent), with ordinals.
_LADDER_POINTS = [("present", 2), ("probable", 1), ("possible", 0),
                  ("unlikely", -1), ("absent", -2)]


def _resolve_image(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else REPO_ROOT / p


def _encode_image_cache(model, image_paths, batch_size=32) -> dict:
    """Encode every unique image once -> {path -> L2-normalized embedding}."""
    cache = {}
    unique = list(dict.fromkeys(str(p) for p in image_paths))
    for i in tqdm(range(0, len(unique), batch_size),
                  desc="  encode imgs", leave=False, ncols=80):
        batch_paths = unique[i:i + batch_size]
        imgs, valid = [], []
        for p in batch_paths:
            try:
                imgs.append(Image.open(p).convert("RGB"))
                valid.append(p)
            except Exception as e:
                print(f"  Warning: could not open {p}: {e}")
        if not imgs:
            continue
        embs = model.encode_image(imgs).cpu().numpy()
        for p, e in zip(valid, embs):
            cache[p] = np.asarray(e, dtype=np.float32).squeeze()
    return cache


def _spearman(a, b) -> float:
    """Spearman rho = Pearson of ranks. Manual (avoids the scipy/BLAS segfault
    documented in the README). Vectors are short (5 points) and tie-free."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    def _rank(x):
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), dtype=np.float64)
        r[order] = np.arange(len(x), dtype=np.float64)
        # average ranks for ties
        _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
        sums = np.zeros(len(counts)); np.add.at(sums, inv, r)
        return (sums / counts)[inv]

    ra, rb = _rank(a), _rank(b)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / denom) if denom > 1e-12 else 0.0


def _cos_all(v, text_emb) -> np.ndarray:
    """Cosine sims of image embedding v against a (k x d) text-emb matrix."""
    return v @ text_emb.T


def run_finding(model, v_by_path, present_entries, absent_entries,
                rung_emb, hist_emb, subtype_map):
    """Return (present_stats, absent_stats, subtype_DS_present) for one finding.

    rung_emb: {rung_key -> (k x d) text-emb ndarray} for all 6 raw rung keys.
    hist_emb: (k x d) ndarray for historical phrasings.
    """
    # Precompute per-rung MEAN sim and MAX sim per image, plus ladder vectors.
    def _per_image(entries):
        rhos, deltas = [], []
        rung_sim_accum = {name: [] for name, _ in _LADDER_POINTS}
        per_image = []
        for e in entries:
            path = str(_resolve_image(e.get("image_path") or e.get("original_path")))
            v = v_by_path.get(path)
            if v is None:
                continue
            mean_sim = {rk: float(np.mean(_cos_all(v, emb)))
                        for rk, emb in rung_emb.items()}
            s_absent = float(np.mean([mean_sim[r] for r in _ABSENT_RUNGS]))
            sim_vec, ord_vec = [], []
            for name, ordinal in _LADDER_POINTS:
                s = s_absent if name == "absent" else mean_sim[name]
                sim_vec.append(s); ord_vec.append(ordinal)
                rung_sim_accum[name].append(s)
            rho = _spearman(sim_vec, ord_vec)
            delta = mean_sim["present"] - s_absent
            rhos.append(rho); deltas.append(delta)
            per_image.append({"image_id": e.get("image_id", ""),
                              "rho": rho, "delta": delta})
        stats = {
            "n": len(rhos),
            "mean_rho": float(np.mean(rhos)) if rhos else None,
            "mean_delta": float(np.mean(deltas)) if deltas else None,
            "mean_rung_sims": {k: (float(np.mean(v)) if v else None)
                               for k, v in rung_sim_accum.items()},
            "rho_values": rhos,      # kept for bootstrap CIs in the aggregator
        }
        return stats, per_image

    present_stats, _ = _per_image(present_entries)
    absent_stats, _ = _per_image(absent_entries)

    # (A) per-subtype DS on PRESENT anchors (max over paraphrases, paper-style).
    pres_emb = rung_emb["present"]
    subtype_pool = {}
    for sub, keys in subtype_map.items():
        mats = []
        for k in keys:
            mats.append(hist_emb if k == "historical" else rung_emb[k])
        subtype_pool[sub] = np.concatenate(mats, axis=0)

    ds_lists = {sub: [] for sub in subtype_map}
    for e in present_entries:
        path = str(_resolve_image(e.get("image_path") or e.get("original_path")))
        v = v_by_path.get(path)
        if v is None:
            continue
        best_pos = float(np.max(_cos_all(v, pres_emb)))
        for sub, pool in subtype_pool.items():
            ds_lists[sub].append(best_pos - float(np.max(_cos_all(v, pool))))
    # keep the per-anchor DS values so bootstrap CIs can be computed downstream
    # (rebuttal T7-T11); the median alone is not enough to resample.
    subtype_DS = {sub: {"median_DS": (float(np.median(vals)) if vals else None),
                        "n": len(vals),
                        "values": [float(v) for v in vals]}
                  for sub, vals in ds_lists.items()}

    return present_stats, absent_stats, subtype_DS


def main():
    ap = argparse.ArgumentParser(description="Exp C — negation realism (ladder)")
    ap.add_argument("--models", nargs="+", default=["all"])
    ap.add_argument("--modality", default="cxr")
    ap.add_argument("--ladder", required=True, help="…_negation_ladder.json")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", default=None)
    ap.add_argument("--max-per-group", type=int, default=None,
                    help="Cap images per (finding, polarity) for a fast smoke run.")
    args = ap.parse_args()

    ladder = json.load(open(REPO_ROOT / args.ladder if not Path(args.ladder).is_absolute()
                            else args.ladder, encoding="utf-8"))
    findings = ladder["findings"]
    subtype_map = ladder["subtype_map"]
    absent_cat = ladder["absent_anchor_category"]

    manifest_path = (REPO_ROOT / args.manifest if not Path(args.manifest).is_absolute()
                     else Path(args.manifest))
    manifest = json.load(open(manifest_path, encoding="utf-8"))

    # Group manifest entries by disease_category.
    by_cat = {}
    for e in manifest:
        by_cat.setdefault(e.get("disease_category", "unknown"), []).append(e)
    absent_entries = by_cat.get(absent_cat, [])
    if args.max_per_group:
        absent_entries = absent_entries[:args.max_per_group]

    model_keys = list_models() if "all" in args.models else args.models
    model_keys = [k for k in model_keys
                  if MODEL_REGISTRY[k].get("modalities") != []
                  and (not MODEL_REGISTRY[k].get("modalities")
                       or args.modality in MODEL_REGISTRY[k]["modalities"])]

    output = args.output or f"experiments/results/negrealism/{args.modality}_negation_realism.json"
    out_path = REPO_ROOT / output if not Path(output).is_absolute() else Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _save(entry_key, entry):
        """Merge one model's entry into the output file immediately, so a hard
        crash on a later model (e.g. the known BioViL timm/DLL crash) cannot
        lose already-computed models."""
        merged = {}
        if out_path.exists():
            try:
                merged = json.load(open(out_path, encoding="utf-8"))
            except Exception:
                merged = {}
        merged[entry_key] = entry
        json.dump(merged, open(out_path, "w"), indent=2)
        return len(merged)

    print(f"Modality       : {args.modality}")
    print(f"Ladder findings: {list(findings)}")
    print(f"Absent anchor  : '{absent_cat}' (n={len(absent_entries)})")
    print(f"Models         : {model_keys}")
    print(f"Output         : {output}")

    results = {}
    for mk in model_keys:
        print(f"\n{'='*56}\n  {MODEL_REGISTRY[mk]['display_name']}  [{args.modality}]\n{'='*56}")
        try:
            model = load_model(mk, device=args.device)
        except Exception as ex:
            print(f"  Failed to load: {ex}")
            continue

        # Encode every anchor image once (present anchors for all findings + absent).
        need_paths = []
        for f in findings:
            for e in by_cat.get(f, [])[:args.max_per_group] if args.max_per_group else by_cat.get(f, []):
                need_paths.append(str(_resolve_image(e.get("image_path") or e.get("original_path"))))
        for e in absent_entries:
            need_paths.append(str(_resolve_image(e.get("image_path") or e.get("original_path"))))
        print(f"  Encoding {len(set(need_paths))} unique anchor images …")
        v_by_path = _encode_image_cache(model, need_paths)

        model_out = {"display_name": MODEL_REGISTRY[mk]["display_name"], "findings": {}}
        for f, finfo in findings.items():
            present_entries = by_cat.get(f, [])
            if args.max_per_group:
                present_entries = present_entries[:args.max_per_group]
            if not present_entries:
                print(f"  [skip] no present anchors for '{f}'")
                continue
            rung_emb = {rk: model.encode_text(txts).cpu().numpy()
                        for rk, txts in finfo["rungs"].items()}
            hist_emb = model.encode_text(finfo["historical"]).cpu().numpy()
            pres, absn, ds = run_finding(
                model, v_by_path, present_entries, absent_entries,
                rung_emb, hist_emb, subtype_map)
            model_out["findings"][f] = {
                "present": pres, "absent": absn, "subtype_DS_present": ds}
            pr = pres["mean_rho"]; ar = absn["mean_rho"]
            print(f"  {f:22s} present rho={pr:+.3f} (n={pres['n']})  "
                  f"absent rho={ar:+.3f} (n={absn['n']})  "
                  f"ds={pres['mean_delta']:+.4f}")

        # Model-level macro summary over findings.
        def _macro(key, sub=None):
            vals = []
            for f in model_out["findings"].values():
                v = f[key]["mean_rho"] if sub is None else None
                if v is not None:
                    vals.append(v)
            return float(np.mean(vals)) if vals else None
        model_out["summary"] = {
            "present_macro_rho": _macro("present"),
            "absent_macro_rho": _macro("absent"),
        }
        results[mk] = model_out
        nfile = _save(mk, model_out)   # incremental save after every model
        print(f"  [saved] {mk} -> {out_path.name} (file now holds {nfile})")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nDone. {len(results)} model entries this run -> {out_path}")


if __name__ == "__main__":
    main()
