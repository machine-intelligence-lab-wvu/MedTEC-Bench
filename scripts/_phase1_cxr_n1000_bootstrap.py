"""
Bootstrap Spearman rho with 95% CI for the n=1,000 single-label NIH-CXR
MedTEC L1-L5 DS values against three retrieval pools:

  * ROCO (full mixed-modality, n=2,000)         — multi-modality retrieval signal
  * ROCO-CXR (filtered, n=354)                  — CXR-domain ROCO retrieval
  * OpenI (n=2,000)                             — within-CXR retrieval

For each (retrieval_metric in {R@1, R@5, R@10, R@50, R@100, medPct}) x
(MedTEC level in {L1..L5}) x (pool in the three above) we report rho, 95%
percentile CI, two-sided bootstrap p-value, and the model list used.

Outputs:
    experiments/results/phase1_cxr_n1000_correlation_bootstrap.csv
    experiments/results/phase1_cxr_n1000_correlation_bootstrap.json
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
LEVELS = ["L1", "L2", "L3", "L4", "L5"]


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


def load_retrieval(json_path: Path, models_present: set) -> dict:
    """{ model: {R@1, R@5, ...} } from a retrieval result JSON."""
    d = json.load(open(json_path, encoding="utf-8"))
    out = {}
    for mk, m in d.get("models", {}).items():
        if mk not in models_present:
            continue
        if "error" in m or "i2t" not in m:
            continue
        i = m["i2t"]
        out[mk] = {
            "R@1": float(i.get("R@1", float("nan"))),
            "R@5": float(i.get("R@5", float("nan"))),
            "R@10": float(i.get("R@10", float("nan"))),
            "R@50": float(i.get("R@50", float("nan"))),
            "R@100": float(i.get("R@100", float("nan"))),
            "medPct": float(i.get("medPct", float("nan"))),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-B", "--bootstrap-iters", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    medtec = json.load(open(RESULTS / "medtec_cxr_n1000.json", encoding="utf-8"))
    cxr_models = list(medtec.keys())
    print(f"NIH n=1,000 MedTEC has {len(cxr_models)} models: {cxr_models}")

    POOLS = [
        ("ROCO_full",   RESULTS / "retrieval_roco.json"),
        ("ROCO_CXR",    RESULTS / "retrieval_roco_cxr_v2.json"),
        ("OpenI",       RESULTS / "retrieval_openi_n2000.json"),
    ]

    out_rows = []
    for pool_name, pool_path in POOLS:
        if not pool_path.exists():
            print(f"  [skip] {pool_name}: {pool_path} missing")
            continue
        retr = load_retrieval(pool_path, set(cxr_models))
        for k in RETRIEVAL_METRICS:
            for lvl in LEVELS:
                xs, ys, kept = [], [], []
                for mk in cxr_models:
                    rv = retr.get(mk, {}).get(k)
                    yv = medtec[mk].get(lvl, {}).get("avg_discrimination_score")
                    if rv is None or yv is None: continue
                    if isinstance(rv, float) and np.isnan(rv): continue
                    if isinstance(yv, float) and np.isnan(yv): continue
                    xs.append(float(rv)); ys.append(float(yv)); kept.append(mk)
                if len(xs) < 3:
                    continue
                res = spearman_bootstrap(np.array(xs), np.array(ys),
                                         B=args.bootstrap_iters, seed=args.seed)
                res.update({
                    "stratification": pool_name,
                    "retrieval_metric": k, "medtec_level": lvl,
                    "models_used": ",".join(kept),
                })
                out_rows.append(res)

    cols = ["stratification", "retrieval_metric", "medtec_level",
            "n_models", "rho", "ci_low", "ci_high", "p_two_sided",
            "B_valid", "models_used"]
    out_csv = RESULTS / "phase1_cxr_n1000_correlation_bootstrap.csv"
    out_json = RESULTS / "phase1_cxr_n1000_correlation_bootstrap.json"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in out_rows:
            w.writerow({c: r.get(c) for c in cols})
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"B": args.bootstrap_iters, "seed": args.seed,
                   "results": out_rows}, f, indent=2)

    print()
    print("=" * 100)
    print(f"Bootstrap Spearman rho — NIH n=1,000 MedTEC L4/L5 (n_models={len(cxr_models)})")
    print("=" * 100)
    print(f"  {'pool':<12}  {'metric':<8}  {'level':<5}  {'rho':>7}  {'95% CI':<22}  {'p':>7}  {'flag'}")
    print("  " + "-" * 80)
    for r in out_rows:
        if r["medtec_level"] not in ["L4", "L5"]:
            continue
        excludes_zero = (r["ci_low"] > 0) or (r["ci_high"] < 0)
        flag = "CI excludes 0" if excludes_zero else "zero in CI"
        ci = f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
        print(f"  {r['stratification']:<12}  {r['retrieval_metric']:<8}  "
              f"{r['medtec_level']:<5}  {r['rho']:>+7.3f}  {ci:<22}  "
              f"{r['p_two_sided']:>7.4f}  {flag}")
    print()
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
