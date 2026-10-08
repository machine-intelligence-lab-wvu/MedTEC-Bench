"""Exp C (negation realism) for the Merlin 3D CT specialist on disease-labeled
abdomen CT. Reuses run_negation_realism.run_finding (image embeddings come from
v_by_path, text from rung_emb — the model object itself is not used inside), with
Merlin's cached 512-d image embeddings + Merlin's text encoder.

Run in the separate Merlin environment (see README).
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import run_negation_realism as nr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", default="configs/prompts/variants/lit/abdomen_ct_disease_negation_ladder_lit.json")
    ap.add_argument("--manifest", default="data/ct_disease_probes/manifest.json")
    ap.add_argument("--out", default="experiments/results/negrealism/abdomen_ct_disease_merlin_negation_realism.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ladder = json.load(open(args.ladder, encoding="utf-8"))
    findings = ladder["findings"]; subtype_map = ladder["subtype_map"]
    absent_cat = ladder["absent_anchor_category"]
    manifest = json.load(open(args.manifest, encoding="utf-8"))

    by_cat = {}
    for e in manifest:
        by_cat.setdefault(e.get("disease_category", "unknown"), []).append(e)
    absent_entries = by_cat.get(absent_cat, [])

    # image embeddings keyed the same way run_finding resolves them
    v_by_path = {}
    for e in manifest:
        v = np.load(e["merlin_emb"]).astype(np.float32)
        v = v / (np.linalg.norm(v) + 1e-8)
        v_by_path[str(nr._resolve_image(e["image_path"]))] = v

    dev = args.device if torch.cuda.is_available() else "cpu"
    from merlin import Merlin
    print(f"device={dev}; loading Merlin ...", flush=True)
    model = Merlin(); model.eval().to(dev)

    def enc(txts):
        with torch.no_grad():
            t = model.model.encode_text(list(txts))
            t = t / t.norm(dim=-1, keepdim=True)
        return t.float().cpu().numpy()

    out = {"display_name": "Merlin (3D CT)", "findings": {}}
    for f, finfo in findings.items():
        present_entries = by_cat.get(f, [])
        if not present_entries:
            print(f"  [skip] no present anchors for {f}"); continue
        rung_emb = {rk: enc(txts) for rk, txts in finfo["rungs"].items()}
        hist_emb = enc(finfo["historical"])
        pres, absn, ds = nr.run_finding(None, v_by_path, present_entries, absent_entries,
                                        rung_emb, hist_emb, subtype_map)
        out["findings"][f] = {"present": pres, "absent": absn, "subtype_DS_present": ds}
        print(f"  {f:20s} present rho={pres['mean_rho']:+.3f} (n={pres['n']})  "
              f"absent rho={absn['mean_rho']:+.3f}  delta={pres['mean_delta']:+.4f}", flush=True)
        # per-subtype DS median
        for sub, d in ds.items():
            print(f"       {sub:11s} DS={d['median_DS']:+.4f} (n={d['n']})", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"merlin": out}, open(args.out, "w"), indent=2)
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
