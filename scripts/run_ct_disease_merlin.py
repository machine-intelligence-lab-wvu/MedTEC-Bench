"""
Compute MedTEC discrimination scores for the Merlin 3D CT specialist on the
disease-labeled abdominal-CT probe set (Exp F).

Uses the cached per-anchor Merlin image embeddings (data/ct_disease_probes/merlin_emb/,
512-d, already L2-normalized) + Merlin's text encoder on the L1-L5 prompts from
configs/prompts/abdomen_ct_disease_prompts.json. DS = max cos(img,pos) - max cos(img,neg),
matching src/metrics/collapse_metrics.evaluate_probe. Output mirrors run_medtec.py's
JSON so downstream aggregation is uniform.

Run in the separate Merlin environment (see README).
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="configs/prompts/abdomen_ct_disease_prompts.json")
    ap.add_argument("--manifest", default="data/ct_disease_probes/manifest.json")
    ap.add_argument("--out", default="experiments/results/medtec_ct_disease_merlin.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--levels", nargs="+", type=int, default=[1, 2, 3, 4, 5],
                    help="Levels to score (use '4 5' for prompt-robustness variants).")
    args = ap.parse_args()

    cfg = json.load(open(args.prompts, encoding="utf-8"))
    manifest = json.load(open(args.manifest, encoding="utf-8"))
    dev = args.device if torch.cuda.is_available() else "cpu"

    # ---- collect every unique prompt text across requested levels ----
    texts = set()
    for L in [str(l) for l in args.levels if l in (1, 2, 3)]:
        for pr in cfg["levels"][L]["probes"]:
            texts.update(pr["positives"]); texts.update(pr["negatives"])
    for L in [str(l) for l in args.levels if l in (4, 5)]:
        for cat, d in cfg["levels"][L]["disease_prompts"].items():
            texts.update(d["positives"]); texts.update(d["negatives"])
    texts = sorted(texts)

    # ---- encode texts with Merlin's text encoder (normalized) ----
    from merlin import Merlin
    print(f"device={dev}; loading Merlin text encoder ...", flush=True)
    model = Merlin(); model.eval().to(dev)
    emb = {}
    B = 32
    with torch.no_grad():
        for i in range(0, len(texts), B):
            batch = texts[i:i + B]
            t = model.model.encode_text(batch)
            t = t / t.norm(dim=-1, keepdim=True)
            for s, v in zip(batch, t.float().cpu().numpy()):
                emb[s] = v
    print(f"encoded {len(texts)} unique prompt texts", flush=True)

    def ds(img, pos, neg):
        cp = np.max([img @ emb[p] for p in pos])
        cn = np.max([img @ emb[n] for n in neg])
        return float(cp - cn)

    # ---- load per-anchor image embeddings ----
    imgs = {}
    for e in manifest:
        v = np.load(e["merlin_emb"]).astype(np.float32)
        v = v / (np.linalg.norm(v) + 1e-8)
        imgs[e["image_id"]] = (v, e["disease_category"])

    out = {"merlin": {"display_name": "Merlin (3D CT)"}}
    # L1-L3 shared probes
    for L in [l for l in [1, 2, 3] if l in args.levels]:
        probes = cfg["levels"][str(L)]["probes"]
        scores = []
        for iid, (img, cat) in imgs.items():
            for pr in probes:
                scores.append(ds(img, pr["positives"], pr["negatives"]))
        out["merlin"][f"L{L}"] = {"avg_discrimination_score": float(np.mean(scores)),
                                   "n_probes": len(scores),
                                   "per_probe": [{"discrimination_score": s} for s in scores]}
    # L4-L5 per-image (by disease_category)
    for L in [l for l in [4, 5] if l in args.levels]:
        dp = cfg["levels"][str(L)]["disease_prompts"]
        scores = []; by_cat = defaultdict(list); per_probe = []
        for iid, (img, cat) in imgs.items():
            if cat not in dp:
                continue
            s = ds(img, dp[cat]["positives"], dp[cat]["negatives"])
            scores.append(s); by_cat[cat].append(s)
            per_probe.append({"discrimination_score": s, "disease_category": cat})
        out["merlin"][f"L{L}"] = {
            "avg_discrimination_score": float(np.mean(scores)),
            "n_probes": len(scores),
            "per_probe": per_probe,
            "by_category": {c: {"avg_DS": float(np.mean(v)), "n": len(v)}
                            for c, v in sorted(by_cat.items())},
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print("\n== Merlin CT-disease DS ==", flush=True)
    for L in [l for l in range(1, 6) if f"L{l}" in out["merlin"]]:
        e = out["merlin"][f"L{L}"]
        tag = "COLLAPSE" if e["avg_discrimination_score"] <= 0 else "OK"
        print(f"  L{L}: DS={e['avg_discrimination_score']:+.4f}  n={e['n_probes']}  [{tag}]", flush=True)
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
