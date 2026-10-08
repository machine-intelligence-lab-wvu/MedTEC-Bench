"""Bootstrap 95% CIs for the disease-labeled abdomen-CT probe set.
Matches the paper protocol (_compute_medtec_bootstrap_n1000.py): B=10,000, seed=42,
resample per-probe discrimination scores with replacement, 2.5/97.5 percentiles.
A cell is COLLAPSE-CERTAIN when its 95% CI lies strictly below 0.
Reads the 2D-panel and Merlin result JSONs; prints a combined CI table."""
import json
import numpy as np

B = 10_000
SEED = 42
FILES = {
    "experiments/results/medtec_ct_disease_2d.json": None,
    "experiments/results/medtec_ct_disease_merlin.json": None,
}
ORDER = ["biomedclip", "pmc_clip", "pubmedclip", "openclip_vitb32",
         "openclip_vitl14", "siglip", "medclip", "merlin"]
FAM = {"biomedclip": "broad", "pmc_clip": "broad", "pubmedclip": "broad",
       "openclip_vitb32": "general", "openclip_vitl14": "general",
       "siglip": "general", "medclip": "CXR-spec", "merlin": "CT-spec(3D)"}


def ci(vals):
    vals = np.asarray([v for v in vals if v is not None], dtype=float)
    if len(vals) == 0:
        return None
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(vals), size=(B, len(vals)))
    means = vals[idx].mean(axis=1)
    return float(vals.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def flag(lo, hi):
    if hi < 0:
        return "COLLAPSE*"      # CI strictly below 0
    if lo > 0:
        return "pass"
    return "near-chance"


data = {}
for f in FILES:
    data.update(json.load(open(f)))

print(f"Disease-labeled Abdomen CT — bootstrap 95% CIs (B={B}, seed={SEED})")
print(f"{'model':16s} {'fam':11s} " + " ".join(f"L{l}".center(22) for l in range(1, 6)))
n_cc = 0
for m in ORDER:
    if m not in data:
        continue
    cells = []
    for l in range(1, 6):
        e = data[m].get(f"L{l}", {})
        vals = [p.get("discrimination_score") for p in e.get("per_probe", [])]
        r = ci(vals)
        if r is None:
            cells.append(" " * 22); continue
        mean, lo, hi = r
        fl = flag(lo, hi)
        if fl == "COLLAPSE*":
            n_cc += 1
        cells.append(f"{mean:+.3f}[{lo:+.3f},{hi:+.3f}]"[:22].ljust(22))
    print(f"{m:16s} {FAM[m]:11s} " + " ".join(cells))
print(f"\n* COLLAPSE-CERTAIN (95% CI strictly < 0): {n_cc} cells")
print("Legend: mean[lo,hi]; COLLAPSE* when hi<0, pass when lo>0, else near-chance.")
