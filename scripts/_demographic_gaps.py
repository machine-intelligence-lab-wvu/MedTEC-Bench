"""
Demographic subgroup analysis of MedTEC discrimination scores (dermoscopy).

Joins per-probe DS values from a full MedTEC result JSON to the attributes in
the probe manifest, and reports the mean DS per subgroup with a percentile
bootstrap 95% CI over probes (B = 10,000, seed 42), plus the max-min gap.

  * skin tone: data/isic_skintone/manifest.json  (Fitzpatrick I-II / III-IV / V-VI; L1-L3)
  * sex, age:  data/isic_sexage/manifest.json    (male / female; age <=50 / >50; L1-L5)

Usage:
    python scripts/_demographic_gaps.py --attribute fitzpatrick \
        --input experiments/results/full_per_probe/medtec_skintone_413.json \
        --manifest data/isic_skintone/manifest.json
    python scripts/_demographic_gaps.py --attribute sex \
        --input experiments/results/full_per_probe/medtec_sexage_isic2019.json \
        --manifest data/isic_sexage/manifest.json
    python scripts/_demographic_gaps.py --attribute age ...   (same files as sex)
"""
import argparse, json, os
from collections import defaultdict
from pathlib import Path

import numpy as np

FITZ = {"I": "I-II", "II": "I-II", "III": "III-IV", "IV": "III-IV", "V": "V-VI", "VI": "V-VI"}


def group_of(row, attribute):
    if attribute == "fitzpatrick":
        return FITZ.get(row.get("fitzpatrick_skin_type"))
    if attribute == "sex":
        return row.get("sex")
    if attribute == "age":
        a = row.get("age_approx")
        return None if a is None else ("<=50" if a <= 50 else ">50")
    raise ValueError(attribute)


def boot_ci(x, B, seed):
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(B, len(x)))].mean(axis=1)
    return [float(v) for v in np.percentile(m, [2.5, 97.5])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attribute", choices=["fitzpatrick", "sex", "age"], required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    manifest = {os.path.basename(r["image_path"]): r
                for r in json.loads(Path(args.manifest).read_text(encoding="utf-8"))}
    res = json.loads(Path(args.input).read_text(encoding="utf-8"))

    out = {}
    for model, md in res.items():
        if not isinstance(md, dict):
            continue
        for lk in sorted(k for k in md if k.startswith("L")):
            groups = defaultdict(list)
            for p in md[lk].get("per_probe", []):
                row = manifest.get(os.path.basename(p["image_path"]))
                g = group_of(row, args.attribute) if row else None
                if g is not None:
                    groups[g].append(p["discrimination_score"])
            if not groups:
                continue
            cell = {g: {"ds": float(np.mean(v)), "ci": boot_ci(np.array(v), args.n_boot, args.seed), "n": len(v)}
                    for g, v in sorted(groups.items())}
            means = [c["ds"] for c in cell.values()]
            out.setdefault(model, {})[lk] = {"groups": cell, "gap": float(max(means) - min(means))}
            print(f"{model:<16} {lk}  " + "  ".join(f"{g} {c['ds']:+.3f}" for g, c in cell.items())
                  + f"   gap {out[model][lk]['gap']:.3f}")

    output = args.output or f"experiments/results/skintone/demographic_gaps_{args.attribute}.json"
    Path(output).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
