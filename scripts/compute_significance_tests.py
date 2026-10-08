"""
compute_significance_tests.py

Statistical significance tests that directly back the paper's main claims.
Requested by reviewer/professor feedback: bootstrap 95% CIs are already in the
tables, but formal p-values strengthen every claim.

Tests
-----
1. Per-cell collapse certification.
     For each (model, modality, level), one-sample Wilcoxon signed-rank test,
     H0: median per-probe DS = 0, two-sided. Converts the cell's color-coded
     pass/weak/collapse label into a formal test with FDR-adjusted q-values.

2. Cascade (L1 -> L4) within each (model, modality).
     Paired Wilcoxon signed-rank on the 50 anchor-image DS pairs, one-sided
     H1: DS_L1 > DS_L4. Tests whether the coarse-to-fine-grained drop is
     statistically reliable for every model.

3. Retrieval-MedTEC decoupling (the central figure).
     Spearman rho between ROCO R@1 and mean DS at L1, L4, L5, L6 across all
     11 models. p-values from an exact permutation null (10000 shuffles),
     because n=11 is small for asymptotic Spearman p.

4. Geometry asymmetry.
     Paired Wilcoxon signed-rank on (D_eff_med, D_eff_nm) pairs across the
     11 models, one-sided H1: D_eff(med) < D_eff(nm). Certifies that the
     text cone narrows on medical captions.

5. Medical vs general-domain on L4 discrimination.
     Mann-Whitney U test comparing mean DS_L4 between the 8 medical-domain
     models and 3 general-domain models. Tests whether medical
     specialization confers a fine-grained advantage.

Output:
    experiments/results/significance_tests.json
"""
import json
from pathlib import Path
from functools import lru_cache

import numpy as np
from scipy.stats import wilcoxon, spearmanr, mannwhitneyu

REPO = Path(__file__).parent.parent
RESULTS = REPO / "experiments" / "results"

MODALITIES = {
    "CXR":       "medtec_cxr.json",
    "Derm":      "medtec_dermoscopy.json",
    "BrainMRI":  "medtec_brain_mri.json",
    "Fundus":    "medtec_fundus.json",
    "AbdCT":     "medtec_abdomen_ct.json",
    "Histopath": "medtec_histopath.json",
}
LEVELS = ["L1", "L2", "L3", "L4", "L5", "L6"]

MODEL_ORDER = [
    "biomedclip", "pubmedclip", "medclip", "pmc_clip", "chexzero",
    "biovil", "conch", "plip",
    "openclip_vitb32", "openclip_vitl14", "siglip",
]
MEDICAL = {"biomedclip", "pubmedclip", "medclip", "pmc_clip",
           "chexzero", "biovil", "conch", "plip"}


@lru_cache(maxsize=None)
def load_modality(modality_key: str) -> dict:
    with open(RESULTS / MODALITIES[modality_key], encoding="utf-8") as f:
        return json.load(f)


def per_probe_ds(level_entry: dict) -> np.ndarray:
    """Pull per-probe DS values from a level record (empty array if missing)."""
    per = level_entry.get("per_probe") or []
    xs = [p.get("discrimination_score") for p in per
          if isinstance(p.get("discrimination_score"), (int, float))]
    return np.asarray(xs, dtype=np.float64)


def per_image_ds(level_entry: dict) -> dict[str, float]:
    """DS aggregated per image_path (mean across positive variants if any)."""
    per = level_entry.get("per_probe") or []
    bucket: dict[str, list[float]] = {}
    for p in per:
        ds = p.get("discrimination_score")
        ip = p.get("image_path")
        if isinstance(ds, (int, float)) and ip:
            bucket.setdefault(ip, []).append(float(ds))
    return {ip: float(np.mean(v)) for ip, v in bucket.items()}


def bh_fdr(ps):
    """Benjamini-Hochberg FDR-adjusted q-values."""
    ps = np.asarray(ps, dtype=float)
    n = len(ps)
    if n == 0:
        return ps.copy()
    order = np.argsort(ps)
    ranked = ps[order]
    # q_i = min_{j >= i}  ranked[j] * n / (j+1)    (cummin from the right)
    adj = ranked * n / np.arange(1, n + 1)
    # cumulative minimum from right to left
    adj_rev_cummin = np.minimum.accumulate(adj[::-1])[::-1]
    adj_rev_cummin = np.minimum(adj_rev_cummin, 1.0)
    q = np.empty(n)
    q[order] = adj_rev_cummin
    return q


def _safe_wilcoxon(*args, **kwargs):
    """Wrap scipy.stats.wilcoxon to gracefully handle all-zero differences."""
    try:
        result = wilcoxon(*args, **kwargs)
        return float(result.statistic), float(result.pvalue)
    except ValueError:
        return float("nan"), 1.0


# ── Test 1: per-cell collapse ──────────────────────────────────────────────
def test_percell_collapse() -> list[dict]:
    out = []
    for mod_key in MODALITIES:
        d = load_modality(mod_key)
        for m in MODEL_ORDER:
            if m not in d:
                continue
            for lv in LEVELS:
                if lv not in d[m]:
                    continue
                xs = per_probe_ds(d[m][lv])
                if len(xs) < 5:
                    continue
                stat, p = _safe_wilcoxon(xs, alternative="two-sided",
                                         zero_method="wilcox")
                out.append({
                    "model": m,
                    "modality": mod_key,
                    "level": lv,
                    "n": int(len(xs)),
                    "median_ds": float(np.median(xs)),
                    "mean_ds": float(np.mean(xs)),
                    "wilcoxon_stat": None if np.isnan(stat) else stat,
                    "p_value": p,
                })
    ps = np.array([r["p_value"] for r in out])
    q = bh_fdr(ps)
    for r, qv in zip(out, q):
        r["p_fdr"] = float(qv)
    return out


# ── Test 2: cascade L1 -> L4 per (model, modality) ─────────────────────────
def test_cascade_l1_l4() -> list[dict]:
    """Paired Wilcoxon DS(L1) > DS(L4), paired by image_path.

    L1 has 3 positive variants per image (150 rows for 50 images); L4 has
    one row per image. We average L1 over variants within each image to
    produce a 50-row per-image L1 vector aligned with L4.
    """
    out = []
    for mod_key in MODALITIES:
        d = load_modality(mod_key)
        for m in MODEL_ORDER:
            if m not in d or "L1" not in d[m] or "L4" not in d[m]:
                continue
            l1_map = per_image_ds(d[m]["L1"])
            l4_map = per_image_ds(d[m]["L4"])
            shared = sorted(set(l1_map) & set(l4_map))
            if len(shared) < 5:
                continue
            a = np.array([l1_map[ip] for ip in shared])
            b = np.array([l4_map[ip] for ip in shared])
            stat, p = _safe_wilcoxon(a, b, alternative="greater",
                                     zero_method="wilcox")
            out.append({
                "model": m,
                "modality": mod_key,
                "n_images_paired": int(len(shared)),
                "median_l1": float(np.median(a)),
                "median_l4": float(np.median(b)),
                "median_drop": float(np.median(a - b)),
                "p_value": p,
            })
    ps = np.array([r["p_value"] for r in out])
    q = bh_fdr(ps)
    for r, qv in zip(out, q):
        r["p_fdr"] = float(qv)
    return out


# ── Test 3: retrieval-MedTEC decoupling ────────────────────────────────────
def test_retrieval_medtec_decoupling() -> dict:
    with open(RESULTS / "retrieval_roco.json", encoding="utf-8") as f:
        rd = json.load(f)
    r1 = {m: r["i2t"]["R@1"] for m, r in rd["models"].items() if "error" not in r}

    # Precompute mean DS per model per level across modalities
    ds_by_model_level = {m: {lv: [] for lv in LEVELS} for m in MODEL_ORDER}
    for mod_key in MODALITIES:
        d = load_modality(mod_key)
        for m in MODEL_ORDER:
            if m not in d:
                continue
            for lv in LEVELS:
                if lv in d[m]:
                    v = d[m][lv].get("avg_discrimination_score")
                    if isinstance(v, (int, float)) and not np.isnan(v):
                        ds_by_model_level[m][lv].append(v)

    out: dict = {}
    rng = np.random.default_rng(42)
    for lv in ["L1", "L4", "L5", "L6"]:
        pairs = []
        for m in MODEL_ORDER:
            if m not in r1:
                continue
            vals = ds_by_model_level[m][lv]
            if vals:
                pairs.append((r1[m], float(np.mean(vals)), m))
        if len(pairs) < 4:
            continue
        xs = np.array([p[0] for p in pairs])
        ys = np.array([p[1] for p in pairs])
        rho_obs, p_asy = spearmanr(xs, ys)
        rho_obs = float(rho_obs)

        # Exact permutation null over all 11! is too large; use 10000 random permutations.
        n_perm = 10000
        null = np.empty(n_perm)
        for i in range(n_perm):
            null[i] = spearmanr(xs, rng.permutation(ys))[0]
        p_perm = float((np.abs(null) >= abs(rho_obs)).mean())

        out[lv] = {
            "n_models": int(len(pairs)),
            "models_included": [p[2] for p in pairs],
            "spearman_rho": rho_obs,
            "p_asymptotic": float(p_asy),
            "p_permutation": p_perm,
            "n_permutations": n_perm,
        }
    return out


# ── Test 4: geometry asymmetry ─────────────────────────────────────────────
def test_geometry_asymmetry() -> dict:
    gfile = RESULTS / "geometry_text.json"
    if not gfile.exists():
        return {"_note": "geometry_text.json missing — test skipped"}
    with open(gfile, encoding="utf-8") as f:
        d = json.load(f)
    models = d["models"]
    mkeys = [m for m in MODEL_ORDER if m in models]
    med = np.array([models[m]["deff_med"] for m in mkeys])
    nm  = np.array([models[m]["deff_nm"]  for m in mkeys])
    stat, p = _safe_wilcoxon(med, nm, alternative="less", zero_method="wilcox")
    return {
        "n_models": int(len(med)),
        "models": mkeys,
        "median_deff_med": float(np.median(med)),
        "median_deff_nm":  float(np.median(nm)),
        "median_ratio_med_over_nm": float(np.median(med / nm)),
        "wilcoxon_stat": None if np.isnan(stat) else stat,
        "p_value": p,
        "alternative": "D_eff(med) < D_eff(nm)",
    }


# ── Test 5: medical vs general-domain family comparison ────────────────────
def test_family_comparison() -> dict:
    per_model: dict = {}
    for mod_key in MODALITIES:
        d = load_modality(mod_key)
        for m in MODEL_ORDER:
            if m in d and "L4" in d[m]:
                v = d[m]["L4"].get("avg_discrimination_score")
                if isinstance(v, (int, float)) and not np.isnan(v):
                    per_model.setdefault(m, []).append(v)
    scores = {m: float(np.mean(v)) for m, v in per_model.items() if v}
    med_scores = [scores[m] for m in scores if m in MEDICAL]
    gen_scores = [scores[m] for m in scores if m not in MEDICAL]
    if len(med_scores) < 2 or len(gen_scores) < 2:
        return {"_note": "insufficient models per family"}
    u, p = mannwhitneyu(med_scores, gen_scores, alternative="two-sided")
    return {
        "n_medical": len(med_scores),
        "n_general": len(gen_scores),
        "medical_models": [m for m in scores if m in MEDICAL],
        "general_models": [m for m in scores if m not in MEDICAL],
        "median_medical": float(np.median(med_scores)),
        "median_general": float(np.median(gen_scores)),
        "mean_medical":   float(np.mean(med_scores)),
        "mean_general":   float(np.mean(gen_scores)),
        "mannwhitney_u": float(u),
        "p_value": float(p),
    }


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    report = {
        "_config": {
            "n_permutations_spearman": 10000,
            "fdr_method": "Benjamini-Hochberg",
            "wilcoxon_zero_method": "wilcox",
            "seed": 42,
        }
    }

    print("Test 1: per-cell collapse (one-sample Wilcoxon)...")
    report["percell_collapse"] = test_percell_collapse()
    rows = report["percell_collapse"]
    n = len(rows)
    sig_raw = sum(1 for r in rows if r["p_value"] < 0.05)
    sig_fdr = sum(1 for r in rows if r["p_fdr"]   < 0.05)
    print(f"  {n} cells tested; {sig_raw} sig. at p<0.05  /  {sig_fdr} at q<0.05 (BH)")

    print("\nTest 2: cascade L1>L4 (paired Wilcoxon, one-sided)...")
    report["cascade_l1_l4"] = test_cascade_l1_l4()
    rows = report["cascade_l1_l4"]
    n = len(rows)
    sig_fdr = sum(1 for r in rows if r["p_fdr"] < 0.05)
    print(f"  {n} (model, modality) pairs; {sig_fdr}/{n} with L1>L4 at q<0.05")

    print("\nTest 3: retrieval-MedTEC decoupling (Spearman + permutation)...")
    report["retrieval_medtec_decoupling"] = test_retrieval_medtec_decoupling()
    for lv, r in report["retrieval_medtec_decoupling"].items():
        print(f"  {lv}: rho={r['spearman_rho']:+.3f}  "
              f"p_asym={r['p_asymptotic']:.3g}  p_perm={r['p_permutation']:.3g}  "
              f"n={r['n_models']}")

    print("\nTest 4: geometry asymmetry D_eff(med) < D_eff(nm) (paired Wilcoxon)...")
    report["geometry_asymmetry"] = test_geometry_asymmetry()
    g = report["geometry_asymmetry"]
    if "_note" not in g:
        print(f"  n={g['n_models']}  medianmed={g['median_deff_med']:.1f}  "
              f"mediannm={g['median_deff_nm']:.1f}  p={g['p_value']:.3g}")
    else:
        print(f"  {g['_note']}")

    print("\nTest 5: medical vs general-domain on DS_L4 (Mann-Whitney U)...")
    report["family_comparison"] = test_family_comparison()
    f = report["family_comparison"]
    if "_note" not in f:
        print(f"  med_n={f['n_medical']}  gen_n={f['n_general']}  "
              f"med_median={f['median_medical']:+.3f}  "
              f"gen_median={f['median_general']:+.3f}  "
              f"p={f['p_value']:.3g}")
    else:
        print(f"  {f['_note']}")

    out = RESULTS / "significance_tests.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
