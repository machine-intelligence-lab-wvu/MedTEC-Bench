"""Three-way per-cell verdicts for the camera-ready (reviewer eTVi request).

collapse-certain : 95% CI strictly below 0
discriminating   : 95% CI strictly above the 0.05 discrimination boundary
near-chance      : everything else (CI crosses 0, or CI > 0 but below 0.05)

Five original modalities come from medtec_bootstrap_ci_n1000.json (paper protocol);
disease-labeled abdomen CT (2D panel + MERLIN 3D) is bootstrapped here with the
same protocol (B=10,000, seed=42, percentile CI over per-probe DS).
Writes experiments/results/camera_ready_verdicts.json and prints a summary.
"""
import json
import numpy as np

B, SEED, BOUND = 10_000, 42, 0.05
R = "experiments/results/"


def verdict(lo, hi):
    if hi < 0:
        return "collapse-certain"
    if lo > BOUND:
        return "discriminating"
    return "near-chance"


def boot(vals):
    vals = np.asarray([v for v in vals if v is not None], dtype=float)
    rng = np.random.default_rng(SEED)
    means = np.empty(B)
    for b in range(B):  # loop keeps memory flat for n=2700
        means[b] = vals[rng.integers(0, len(vals), len(vals))].mean()
    return float(vals.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


out = {}
for mod, models in json.load(open(R + "medtec_bootstrap_ci_n1000.json")).items():
    out[mod] = {}
    for m, levels in models.items():
        out[mod][m] = {}
        for L, c in levels.items():
            if not isinstance(c, dict) or "ci_lo" not in c:
                continue
            out[mod][m][L] = dict(mean=c["mean"], lo=c["ci_lo"], hi=c["ci_hi"],
                                  verdict=verdict(c["ci_lo"], c["ci_hi"]))

ct = {}
for f in ["medtec_ct_disease_2d.json", "medtec_ct_disease_merlin.json"]:
    ct.update(json.load(open(R + f)))
# The released result JSONs are summaries; re-bootstrapping needs the per-probe outputs.
if not any(d.get("L1", {}).get("per_probe") for d in ct.values()):
    raise SystemExit("Per-probe scores missing. Regenerate them with scripts/run_ct_disease_merlin.py and the 2D "
                     "CT run, and scripts/run_medtec.py for OpenI, before re-deriving the verdicts. The released "
                     "experiments/results/camera_ready_verdicts.json already holds the paper's verdicts.")
out["abdomen_ct_disease"] = {}
for m, d in ct.items():
    out["abdomen_ct_disease"][m] = {}
    for l in range(1, 6):
        pp = d.get(f"L{l}", {}).get("per_probe")
        if not pp:
            continue
        mean, lo, hi = boot([p.get("discrimination_score") for p in pp])
        out["abdomen_ct_disease"][m][f"L{l}"] = dict(mean=mean, lo=lo, hi=hi, verdict=verdict(lo, hi))

# OpenI CXR probe set (Table 7) — same protocol, reported separately, not in the totals.
openi = {}
for m, d in json.load(open(R + "medtec_openi_cxr.json")).items():
    openi[m] = {}
    for l in range(1, 6):
        pp = d.get(f"L{l}", {}).get("per_probe")
        if pp:
            mean, lo, hi = boot([p.get("discrimination_score") for p in pp])
            openi[m][f"L{l}"] = dict(mean=mean, lo=lo, hi=hi, verdict=verdict(lo, hi))

json.dump(dict(out, openi_cxr=openi), open(R + "camera_ready_verdicts.json", "w"), indent=1)

# ---- summary ----
print("openi_cxr", {m: "".join({"collapse-certain": "X", "near-chance": ".", "discriminating": "D"}[openi[m][L]["verdict"]]
                                for L in sorted(openi[m])) for m in openi})
tot = {}
for mod, models in out.items():
    cnt = {}
    for m, levels in models.items():
        for L, c in levels.items():
            cnt.setdefault(L, {}).setdefault(c["verdict"], 0)
            cnt[L][c["verdict"]] += 1
            t = tot.setdefault(L, {})
            t[c["verdict"]] = t.get(c["verdict"], 0) + 1
    print(mod, {L: cnt[L] for L in sorted(cnt)})
print("ALL", {L: tot[L] for L in sorted(tot)})
for L in ["L4", "L5"]:
    best = max(((c[L]["mean"], mod, m) for mod, ms in out.items() for m, c in ms.items() if L in c))
    print(f"max {L} DS: {best[0]:+.4f} ({best[1]}, {best[2]})")
