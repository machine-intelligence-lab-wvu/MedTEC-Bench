"""
Phase 1, Tasks 1.4 + 1.5 — joint analysis of retrieval R@k vs MedTEC L1-L6 DS.

Reads:
    experiments/results/retrieval_roco.json          # Task 1.3 output
    experiments/results/medtec_crossmodality.json    # tidy {model}{modality}{level} -> {ds, class}

Writes:
    experiments/results/phase1_per_model_table.csv          # Task 1.4 deliverable
    experiments/results/phase1_correlation_bootstrap.csv    # Task 1.5 deliverable
    experiments/results/phase1_correlation_bootstrap.json   # same data, structured

Design:
    Two stratifications because the missing-data pattern is structural, not random:
      - CXR-only (n=9): every model that has a CXR DS score, ranked by retrieval and by CXR DS.
      - Cross-modality-mean (n=7): the 7 multi-modality models, ranked by retrieval and by
        per-level mean DS over {CXR, Derm, Brain MRI, Fundus, Abd CT, Histopath}.
    Specialists (CheXzero, BioViL, CONCH, PLIP) have no cross-modality mean and are excluded
    from the n=7 analysis explicitly — never silently dropped.

    Bootstrap: B=10,000 model-level resamples (with replacement), seed=42.
    Two-sided p-value: fraction of resampled |rho| >= |observed rho| under the
    centered-resample null (recommended for bootstrap p-values; permutation would be
    cleaner with this n but TASKS_PHASE1.md asks for a bootstrap p-value).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def spearmanr_safe(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho computed as Pearson correlation of ranks, avoiding
    scipy.stats.spearmanr because its internal Pearson code path crashes
    on this Windows/numpy/torch build via a BLAS+OpenMP runtime conflict.
    Returns NaN if either rank vector has zero variance."""
    rx = rankdata(x); ry = rankdata(y)
    sx = rx - rx.mean(); sy = ry - ry.mean()
    den_x = float(np.sqrt((sx * sx).sum()))
    den_y = float(np.sqrt((sy * sy).sum()))
    if den_x == 0.0 or den_y == 0.0:
        return float("nan")
    return float((sx * sy).sum() / (den_x * den_y))

REPO = Path(__file__).parent.parent
RESULTS = REPO / "experiments" / "results"

RETRIEVAL_METRICS = ["R@1", "R@5", "R@10", "R@50", "R@100", "medPct"]
MEDTEC_LEVELS = ["L1", "L2", "L3", "L4", "L5", "L6"]
MULTI_MODALITY_MODELS = [
    "biomedclip", "medclip", "pubmedclip", "pmc_clip",
    "openclip_vitb32", "openclip_vitl14", "siglip",
]


def load_inputs():
    retr = json.load(open(RESULTS / "retrieval_roco.json", encoding="utf-8"))
    med  = json.load(open(RESULTS / "medtec_crossmodality.json", encoding="utf-8"))
    return retr, med


def build_per_model_table(retr: dict, med: dict) -> pd.DataFrame:
    """One row per model, columns for each retrieval metric and each (modality, level) DS,
    plus per-level cross-modality mean for the 7 multi-modality models."""
    rows = []
    for mkey, m_retr in retr["models"].items():
        if "error" in m_retr:
            continue
        row = {"model": mkey, "display_name": m_retr["display_name"]}
        for k in RETRIEVAL_METRICS:
            row[f"i2t_{k}"] = m_retr["i2t"].get(k)
            row[f"t2i_{k}"] = m_retr["t2i"].get(k)

        m_med = med.get(mkey, {})
        modalities_present = []
        for mod, levels in m_med.items():
            mod_clean = mod.replace("\\", "").strip()
            has_any = any(levels[l]["ds"] is not None for l in MEDTEC_LEVELS)
            if has_any:
                modalities_present.append(mod_clean)
            for lvl in MEDTEC_LEVELS:
                row[f"DS_{mod_clean}_{lvl}"] = levels[lvl]["ds"]

        # Cross-modality per-level mean over modalities where this model has data.
        # For the 7 multi-modality models this averages over all 6.
        for lvl in MEDTEC_LEVELS:
            vals = [row[f"DS_{mod}_{lvl}"]
                    for mod in modalities_present
                    if row.get(f"DS_{mod}_{lvl}") is not None]
            row[f"DS_meanXmod_{lvl}"] = float(np.mean(vals)) if vals else None
        row["n_modalities_with_data"] = len(modalities_present)
        rows.append(row)
    return pd.DataFrame(rows)


def spearman_bootstrap(x: np.ndarray, y: np.ndarray, B: int, seed: int):
    """Model-level bootstrap of Spearman rho.
    Returns dict with point estimate, percentile CI (95%), and a 2-sided
    bootstrap p-value via the centered-resample null."""
    n = len(x)
    rho_obs = spearmanr_safe(x, y)
    rng = np.random.default_rng(seed)
    rhos = np.full(B, np.nan)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(x[idx])) < 2 or len(np.unique(y[idx])) < 2:
            continue
        rhos[b] = spearmanr_safe(x[idx], y[idx])
    rhos_valid = rhos[~np.isnan(rhos)]
    if len(rhos_valid) == 0:
        return {"rho": float(rho_obs), "ci_low": np.nan, "ci_high": np.nan,
                "p_two_sided": np.nan, "n_models": n, "B_valid": 0}
    ci_low, ci_high = np.percentile(rhos_valid, [2.5, 97.5])
    # Centered-resample p-value: shift bootstrap distribution to be centered on 0,
    # then ask how often |centered rho| >= |observed rho|.
    centered = rhos_valid - np.mean(rhos_valid)
    p = float(np.mean(np.abs(centered) >= abs(rho_obs)))
    return {"rho": float(rho_obs),
            "ci_low": float(ci_low), "ci_high": float(ci_high),
            "p_two_sided": p, "n_models": n, "B_valid": int(len(rhos_valid))}


def run_correlations(per_model: pd.DataFrame, B: int, seed: int) -> pd.DataFrame:
    """Two stratifications: CXR-only (n up to 9), cross-modality-mean (n=7)."""
    out_rows = []

    # Stratification 1: CXR-only — use DS_CXR_<L>
    cxr_models = per_model[per_model["DS_CXR_L1"].notna()]["model"].tolist()
    for k in RETRIEVAL_METRICS:
        for lvl in MEDTEC_LEVELS:
            sub = per_model[per_model["model"].isin(cxr_models)].dropna(
                subset=[f"i2t_{k}", f"DS_CXR_{lvl}"])
            if len(sub) < 3:
                continue
            x = sub[f"i2t_{k}"].to_numpy(dtype=float)
            y = sub[f"DS_CXR_{lvl}"].to_numpy(dtype=float)
            res = spearman_bootstrap(x, y, B=B, seed=seed)
            res.update({"stratification": "CXR_only",
                        "retrieval_metric": k, "medtec_level": lvl,
                        "models_used": ",".join(sub["model"].tolist())})
            out_rows.append(res)

    # Stratification 2: cross-modality mean — multi-modality models only.
    for k in RETRIEVAL_METRICS:
        for lvl in MEDTEC_LEVELS:
            sub = per_model[per_model["model"].isin(MULTI_MODALITY_MODELS)].dropna(
                subset=[f"i2t_{k}", f"DS_meanXmod_{lvl}"])
            if len(sub) < 3:
                continue
            x = sub[f"i2t_{k}"].to_numpy(dtype=float)
            y = sub[f"DS_meanXmod_{lvl}"].to_numpy(dtype=float)
            res = spearman_bootstrap(x, y, B=B, seed=seed)
            res.update({"stratification": "crossmod_mean",
                        "retrieval_metric": k, "medtec_level": lvl,
                        "models_used": ",".join(sub["model"].tolist())})
            out_rows.append(res)

    cols = ["stratification", "retrieval_metric", "medtec_level",
            "n_models", "rho", "ci_low", "ci_high", "p_two_sided",
            "B_valid", "models_used"]
    return pd.DataFrame(out_rows)[cols]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-B", "--bootstrap-iters", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Phase 1 correlation bootstrap — B={args.bootstrap_iters}, seed={args.seed}")
    retr, med = load_inputs()
    per_model = build_per_model_table(retr, med)
    out_per_model = RESULTS / "phase1_per_model_table.csv"
    per_model.to_csv(out_per_model, index=False)
    print(f"Wrote per-model table: {out_per_model}  ({len(per_model)} models, {len(per_model.columns)} cols)")

    boot = run_correlations(per_model, B=args.bootstrap_iters, seed=args.seed)
    out_boot_csv = RESULTS / "phase1_correlation_bootstrap.csv"
    out_boot_json = RESULTS / "phase1_correlation_bootstrap.json"
    boot.to_csv(out_boot_csv, index=False)
    with open(out_boot_json, "w") as f:
        json.dump({"B": args.bootstrap_iters, "seed": args.seed,
                   "results": boot.to_dict(orient="records")}, f, indent=2)
    print(f"Wrote bootstrap table: {out_boot_csv}  ({len(boot)} cells)")
    print(f"Wrote JSON:            {out_boot_json}")

    # Pretty-print the headline cells the user flagged in TASKS_PHASE1.md
    print()
    print("=" * 92)
    print("HEADLINE CELLS (from TASKS_PHASE1.md decision gate)")
    print("=" * 92)
    print(f"  {'stratification':<14}  {'metric':<7}  {'level':<5}  {'n':>2}  "
          f"{'rho':>7}  {'95% CI':<20}  {'p':>7}  {'flag':<20}")
    print("  " + "-" * 88)
    for _, r in boot.iterrows():
        sig = "ZERO IN CI" if (r["ci_low"] <= 0 <= r["ci_high"]) else "CI EXCLUDES ZERO"
        sign = "neg" if r["rho"] < 0 else "pos"
        flag = f"{sign}, {sig}"
        print(f"  {r['stratification']:<14}  {r['retrieval_metric']:<7}  "
              f"{r['medtec_level']:<5}  {r['n_models']:>2}  "
              f"{r['rho']:>7.3f}  [{r['ci_low']:>+6.3f}, {r['ci_high']:>+6.3f}]  "
              f"{r['p_two_sided']:>7.4f}  {flag:<20}")
    print("=" * 92)
    print()
    print("Decision gate: at least one (R@k, L4/L5) cell with negative rho AND CI excluding 0?")
    gate_cells = boot[(boot["medtec_level"].isin(["L4", "L5"]))
                      & (boot["rho"] < 0)
                      & (boot["ci_high"] < 0)]
    if len(gate_cells) > 0:
        print(f"  YES — {len(gate_cells)} qualifying cell(s):")
        for _, r in gate_cells.iterrows():
            print(f"    {r['stratification']:<14}  {r['retrieval_metric']:<7}  "
                  f"{r['medtec_level']:<5}  rho={r['rho']:+.3f}  "
                  f"CI=[{r['ci_low']:+.3f},{r['ci_high']:+.3f}]")
        print("  --> headline reframe is viable. Proceed to Phase 3 next.")
    else:
        print("  NO — every L4/L5 cell either is positive or has CI crossing zero.")
        print("  --> headline reframe is dead. Re-evaluate central claim before proceeding.")


if __name__ == "__main__":
    main()
