"""
Bootstrap Spearman rho with 95% CI for OpenI retrieval x CXR MedTEC discrimination.

Companion to scripts/_phase1_correlation_bootstrap.py — same recipe but on the
OpenI CXR-only retrieval pool (n=3666 image-report pairs) instead of ROCO. The
existing experiments/results/openi_cxr_unified.json already has Spearman rho
point estimates with scipy p-values, but no bootstrap CIs. n=11 here (vs n=7
for the cross-modality ROCO analysis) since OpenI is CXR-only and every model
with CXR support can be ranked, including CONCH and PLIP whose absence from
the cross-modality stratification was a missing-data exclusion.

Outputs:
    experiments/results/phase1_openi_correlation_bootstrap.csv
    experiments/results/phase1_openi_correlation_bootstrap.json
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

REPO = Path(__file__).parent.parent
RESULTS = REPO / "experiments" / "results"

RETRIEVAL_METRICS = ["R@1", "R@5", "R@10", "R@50", "R@100", "medPct"]
MEDTEC_LEVELS = ["L1", "L2", "L3", "L4", "L5"]    # L6 not run on OpenI


def spearmanr_safe(x: np.ndarray, y: np.ndarray) -> float:
    rx = rankdata(x); ry = rankdata(y)
    sx = rx - rx.mean(); sy = ry - ry.mean()
    den_x = float(np.sqrt((sx * sx).sum()))
    den_y = float(np.sqrt((sy * sy).sum()))
    if den_x == 0.0 or den_y == 0.0:
        return float("nan")
    return float((sx * sy).sum() / (den_x * den_y))


def spearman_bootstrap(x, y, B, seed):
    n = len(x)
    rho_obs = spearmanr_safe(x, y)
    rng = np.random.default_rng(seed)
    rhos = np.full(B, np.nan)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(x[idx])) < 2 or len(np.unique(y[idx])) < 2:
            continue
        rhos[b] = spearmanr_safe(x[idx], y[idx])
    rhos = rhos[~np.isnan(rhos)]
    if len(rhos) == 0:
        return {"rho": float(rho_obs), "ci_low": np.nan, "ci_high": np.nan,
                "p_two_sided": np.nan, "n_models": n, "B_valid": 0}
    lo, hi = np.percentile(rhos, [2.5, 97.5])
    centered = rhos - rhos.mean()
    p = float(np.mean(np.abs(centered) >= abs(rho_obs)))
    return {"rho": float(rho_obs), "ci_low": float(lo), "ci_high": float(hi),
            "p_two_sided": p, "n_models": n, "B_valid": int(len(rhos))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-B", "--bootstrap-iters", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = json.load(open(RESULTS / "openi_cxr_unified.json", encoding="utf-8"))
    rows = src["rows"]
    print(f"Loaded {len(rows)} models from openi_cxr_unified.json")

    # Build per-metric/per-level vectors across models
    out_rows = []
    for k in RETRIEVAL_METRICS:
        for lvl in MEDTEC_LEVELS:
            xs, ys = [], []
            kept = []
            for r in rows:
                xv = r.get("ret", {}).get(k)
                yv = r.get("ds", {}).get(lvl)
                if xv is None or yv is None:
                    continue
                if isinstance(xv, float) and np.isnan(xv): continue
                if isinstance(yv, float) and np.isnan(yv): continue
                xs.append(float(xv)); ys.append(float(yv))
                kept.append(r["model"])
            if len(xs) < 3:
                continue
            res = spearman_bootstrap(np.array(xs), np.array(ys),
                                     B=args.bootstrap_iters, seed=args.seed)
            res.update({"stratification": "OpenI_CXR",
                        "retrieval_metric": k, "medtec_level": lvl,
                        "models_used": ",".join(kept)})
            out_rows.append(res)

    # AUROC vs DS too — interesting bonus axis the existing JSON already has
    for lvl in MEDTEC_LEVELS:
        xs, ys, kept = [], [], []
        for r in rows:
            xv = r.get("auroc"); yv = r.get("ds", {}).get(lvl)
            if xv is None or yv is None:
                continue
            if isinstance(xv, float) and np.isnan(xv): continue
            if isinstance(yv, float) and np.isnan(yv): continue
            xs.append(float(xv)); ys.append(float(yv)); kept.append(r["model"])
        if len(xs) < 3:
            continue
        res = spearman_bootstrap(np.array(xs), np.array(ys),
                                 B=args.bootstrap_iters, seed=args.seed)
        res.update({"stratification": "OpenI_CXR",
                    "retrieval_metric": "AUROC", "medtec_level": lvl,
                    "models_used": ",".join(kept)})
        out_rows.append(res)

    cols = ["stratification", "retrieval_metric", "medtec_level",
            "n_models", "rho", "ci_low", "ci_high", "p_two_sided",
            "B_valid", "models_used"]
    out_csv = RESULTS / "phase1_openi_correlation_bootstrap.csv"
    out_json = RESULTS / "phase1_openi_correlation_bootstrap.json"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in out_rows:
            w.writerow({c: r.get(c) for c in cols})
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"B": args.bootstrap_iters, "seed": args.seed,
                   "results": out_rows}, f, indent=2)

    # Headline cells (L4 + L5)
    print()
    print("=" * 92)
    print(f"OpenI CXR — bootstrap Spearman rho (B={args.bootstrap_iters}, seed={args.seed}, n=11)")
    print("=" * 92)
    print(f"  {'metric':<8}  {'level':<5}  {'rho':>7}  {'95% CI':<22}  {'p':>7}  {'flag'}")
    print("  " + "-" * 76)
    for r in out_rows:
        if r["medtec_level"] in ["L4", "L5"]:
            flag = "CI EXCLUDES ZERO" if (r["ci_low"] > 0 or r["ci_high"] < 0) else "ZERO IN CI"
            ci = f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
            print(f"  {r['retrieval_metric']:<8}  {r['medtec_level']:<5}  "
                  f"{r['rho']:>+7.3f}  {ci:<22}  {r['p_two_sided']:>7.4f}  {flag}")
    print()
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
