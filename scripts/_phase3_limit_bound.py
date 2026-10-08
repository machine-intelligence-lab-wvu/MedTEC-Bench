"""
Phase 3.2 — LIMIT bound (Weller et al.) numerical calculation for the L4 CXR probe.

For a discrimination task with n distractor concepts and k correct positives,
the dimensional floor required to embed all C(n, k) distinct positive-subsets
with cosine margin gamma is:

    d_floor = log2( C(n, k) ) / log2( 1 + 1/gamma )

The bound is INFORMATIVE: any encoder with effective dimensionality below
d_floor cannot simultaneously preserve the geometry needed to discriminate
all positive subsets at the required margin. Models below the floor are
predicted to fail the L4 task; models above the floor MAY still fail for
other reasons (training mismatch, hubness), but at least the geometry is
sufficient.

Inputs:
    configs/prompts/cxr_prompts.json                    -> n, k from L4 negative pool
    experiments/results/medtec_cxr.json                 -> gamma from L3 DS distribution
    experiments/results/phase3_text_geometry.csv        -> per-model D_eff
    experiments/results/medtec_cxr.json                 -> per-model L4 avg DS (empirical pass/fail)

Outputs:
    experiments/results/phase3_limit_bound.csv          -> tidy gap table for paper
    experiments/results/phase3_limit_bound.json         -> includes the bound parameters

Reference:
    Weller, O., Boratko, M., Naim, I., Lee, J. (2024).
    "On the Theoretical Limitations of Embedding-Based Retrieval."
"""
import csv
import json
import math
from pathlib import Path

REPO = Path(__file__).parent.parent
RESULTS = REPO / "experiments" / "results"


def extract_n_k(prompt_cfg_path: Path):
    """L4 negative pool size and per-probe positive count for CXR."""
    cfg = json.load(open(prompt_cfg_path, encoding="utf-8"))
    l4 = cfg["levels"]["4"]["disease_prompts"]
    all_negatives = set()
    k_values = set()
    for cat, prompts in l4.items():
        all_negatives.update(prompts["negatives"])
        k_values.add(len(prompts["positives"]))
    if len(k_values) != 1:
        raise ValueError(f"k inconsistent across L4 disease categories: {k_values}")
    return len(all_negatives), k_values.pop(), sorted(l4.keys())


def gamma_candidates(medtec_path: Path):
    """Multiple gamma estimates from L3 per-image DS distribution."""
    import numpy as np
    m = json.load(open(medtec_path, encoding="utf-8"))
    l3_ds = []
    for mres in m.values():
        per = mres.get("L3", {}).get("per_probe", [])
        for p in per:
            ds = p.get("discrimination_score")
            if ds is not None:
                l3_ds.append(float(ds))
    arr = np.array(l3_ds)
    passing = arr[arr >= 0]
    return {
        "n_per_image_obs":   int(len(arr)),
        "n_passing":         int(len(passing)),
        "frac_passing":      float((arr >= 0).mean()),
        "gamma_p5_passing":  float(np.percentile(passing, 5)),
        "gamma_p25_passing": float(np.percentile(passing, 25)),
        "gamma_median_passing": float(np.median(passing)),
        "gamma_min_model_avg": min(
            mres["L3"]["avg_discrimination_score"]
            for mres in m.values()
            if mres.get("L3", {}).get("avg_discrimination_score", 0) > 0
        ),
    }


def d_floor(n: int, k: int, gamma: float) -> float:
    if gamma <= 0:
        return float("inf")
    return math.log2(math.comb(n, k)) / math.log2(1.0 + 1.0 / gamma)


def load_geometry(path: Path) -> dict:
    """{ model: {D_eff, embedding_dim, ...} }."""
    if not path.exists():
        print(f"WARNING: {path} missing; per-model gap column will be empty.")
        return {}
    with open(path, encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        return {row["model"]: row for row in rdr}


def load_l4_avg_ds(medtec_path: Path) -> dict:
    """{ model: avg_L4_DS } — for empirical pass/fail of the prediction."""
    m = json.load(open(medtec_path, encoding="utf-8"))
    return {mkey: float(mres["L4"]["avg_discrimination_score"])
            for mkey, mres in m.items()
            if mres.get("L4", {}).get("avg_discrimination_score") is not None}


def main():
    n, k, categories = extract_n_k(REPO / "configs" / "prompts" / "cxr_prompts.json")
    print(f"L4 CXR — n (distinct negative concepts) = {n}, k (positives per probe) = {k}")
    print(f"  disease categories = {categories}")
    print(f"  log2(C({n},{k})) = log2({math.comb(n,k)}) = {math.log2(math.comb(n,k)):.3f}")

    gammas = gamma_candidates(RESULTS / "medtec_cxr.json")
    print(f"\nL3 per-image DS pool: {gammas['n_per_image_obs']} obs, "
          f"{gammas['frac_passing']:.1%} >= 0")
    for k_, v in gammas.items():
        if k_.startswith("gamma_"):
            print(f"  {k_}: {v:.4f}  -> d_floor = {d_floor(n, k, v):.2f}")

    geo = load_geometry(RESULTS / "phase3_text_geometry.csv")
    l4_avg = load_l4_avg_ds(RESULTS / "medtec_cxr.json")

    # Use the median-passing gamma as the headline estimate (more
    # discriminating than p5 which is the noise floor).
    gamma_headline = gammas["gamma_median_passing"]
    d_floor_headline = d_floor(n, k, gamma_headline)
    print(f"\nHEADLINE: gamma = median passing L3 DS = {gamma_headline:.4f}")
    print(f"          d_floor = {d_floor_headline:.2f}")

    # Per-model gap table.
    rows = []
    for mkey in geo or l4_avg:
        g = geo.get(mkey, {})
        d_eff = float(g["D_eff"]) if g.get("D_eff") not in (None, "") else None
        D = int(g["embedding_dim"]) if g.get("embedding_dim") not in (None, "") else None
        avg_l4 = l4_avg.get(mkey)
        gap = (d_eff - d_floor_headline) if d_eff is not None else None
        predicted = ("FAIL" if (d_eff is not None and d_eff < d_floor_headline)
                     else "PASS" if d_eff is not None else "?")
        observed  = ("FAIL" if (avg_l4 is not None and avg_l4 < 0)
                     else "PASS" if avg_l4 is not None else "?")
        agreement = ("YES" if predicted == observed and predicted != "?" else
                     "NO"  if predicted != observed and "?" not in (predicted, observed) else "?")
        rows.append({
            "model": mkey,
            "display_name": g.get("display_name", mkey),
            "embedding_dim_D": D,
            "D_eff": d_eff,
            "d_floor": round(d_floor_headline, 3),
            "gap_Deff_minus_dfloor": gap,
            "L4_avg_DS_observed": avg_l4,
            "predicted_by_LIMIT": predicted,
            "observed_in_MedTEC": observed,
            "agree": agreement,
        })

    out_csv = RESULTS / "phase3_limit_bound.csv"
    out_json = RESULTS / "phase3_limit_bound.json"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "n": n, "k": k,
            "n_choose_k": math.comb(n, k),
            "gamma_candidates": gammas,
            "headline_gamma": gamma_headline,
            "headline_d_floor": d_floor_headline,
            "rows": rows,
        }, f, indent=2)

    print(f"\nPer-model gap table (headline gamma = {gamma_headline:.4f}):")
    print(f"  {'model':<22}  {'D':>4}  {'D_eff':>7}  {'d_floor':>7}  {'gap':>7}  "
          f"{'L4 DS':>8}  {'pred':<5}  {'obs':<5}  agree")
    print("  " + "-" * 88)
    for r in rows:
        d_eff_s = f"{r['D_eff']:>7.2f}" if r['D_eff'] is not None else "      -"
        gap_s   = f"{r['gap_Deff_minus_dfloor']:>+7.2f}" if r['gap_Deff_minus_dfloor'] is not None else "      -"
        l4_s    = f"{r['L4_avg_DS_observed']:>+8.4f}" if r['L4_avg_DS_observed'] is not None else "       -"
        D_s     = f"{r['embedding_dim_D']:>4}" if r['embedding_dim_D'] is not None else "   -"
        print(f"  {r['model']:<22}  {D_s}  {d_eff_s}  {r['d_floor']:>7.2f}  {gap_s}  {l4_s}  "
              f"{r['predicted_by_LIMIT']:<5}  {r['observed_in_MedTEC']:<5}  {r['agree']}")
    print(f"\nWrote {out_csv}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
