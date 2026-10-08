"""
Build the Exp C (negation realism) tables from run_negation_realism.py output.

Two tables per modality:
  (A) per-subtype DS on present anchors  -> table_negrealism_<mod>_A.tex + .csv
      cols = definite / hedged / ruled_out / historical (macro-median DS over findings)
  (B) certainty-ladder monotonicity      -> table_negrealism_<mod>_B.tex + .csv
      cols = present rho, absent rho (both with 95% bootstrap CI), mean dynamic range

Interpretation:
  A: DS <= 0 => the model does not prefer the present-affirmation over the
     subtype phrasing (collapse persists under that negation style).
  B: a model that understands graded negation shows present-rho -> +1 and
     absent-rho -> -1; a collapsed model shows both ~ 0 (flat ladder).

Usage:
    python scripts/_build_negation_realism_table.py \
        --input experiments/results/negrealism/cxr_negation_realism.json \
        --modality cxr
"""

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
_SUBTYPES = ["definite", "hedged", "ruled_out", "historical"]
_SEED = 42
_B = 10000


def _boot_ci(vals, seed=_SEED, B=_B):
    vals = np.asarray([v for v in vals if v is not None], dtype=np.float64)
    if len(vals) == 0:
        return (None, None, None)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(B, len(vals)))
    means = vals[idx].mean(axis=1)
    return (float(vals.mean()),
            float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))


def _pooled_rho(model_entry, polarity):
    """Per-image rho pooled across findings for one polarity -> np.array."""
    pooled = []
    for f in model_entry["findings"].values():
        pooled.extend(f[polarity].get("rho_values", []))
    return np.asarray([v for v in pooled if v is not None], dtype=np.float64)


def _macro_rho(model_entry, polarity):
    """Pool per-image rho across findings for one polarity -> (mean, lo, hi)."""
    return _boot_ci(_pooled_rho(model_entry, polarity))


def _grounding_gap(model_entry, seed=_SEED, B=_B):
    """Delta-rho = present_rho - absent_rho (image-grounding of the ladder).
    Ideal +2 (present->+1, absent->-1); ~0 = image-blind text-prior ordering.
    CI via independent bootstrap of the two (disjoint) anchor sets."""
    p = _pooled_rho(model_entry, "present")
    a = _pooled_rho(model_entry, "absent")
    if len(p) == 0 or len(a) == 0:
        return (None, None, None)
    rng = np.random.default_rng(seed)
    pm = p[rng.integers(0, len(p), size=(B, len(p)))].mean(axis=1)
    am = a[rng.integers(0, len(a), size=(B, len(a)))].mean(axis=1)
    d = pm - am
    return (float(p.mean() - a.mean()),
            float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


def _macro_delta(model_entry, polarity="present"):
    vals = [f[polarity]["mean_delta"] for f in model_entry["findings"].values()
            if f[polarity]["mean_delta"] is not None]
    return float(np.mean(vals)) if vals else None


def _macro_subtype_ds(model_entry, subtype):
    """Mean over findings of the per-finding median DS (present anchors)."""
    vals = []
    for f in model_entry["findings"].values():
        ds = f["subtype_DS_present"].get(subtype, {})
        if ds.get("median_DS") is not None:
            vals.append(ds["median_DS"])
    return float(np.mean(vals)) if vals else None


def _subtype_ds_ci(model_entry, subtype, seed=_SEED, B=_B):
    """Point estimate + bootstrap 95% CI for the per-subtype DS, as the POOLED MEAN
    over all present anchors across findings (anchors weighted equally, so a class
    with 111 anchors does not count the same as one with 200).

    Requires the per-anchor 'values' lists written by run_negation_realism.py.
    Returns (point, lo, hi, n_anchors) or (None, None, None, 0) if values are absent.
    """
    pooled = []
    for f in model_entry["findings"].values():
        pooled.extend(f["subtype_DS_present"].get(subtype, {}).get("values", []))
    if not pooled:
        return (None, None, None, 0)
    v = np.asarray(pooled, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = v[rng.integers(0, len(v), size=(B, len(v)))].mean(axis=1)
    return (float(v.mean()), float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)), int(len(v)))


def build(input_path, modality, outdir):
    data = json.load(open(input_path, encoding="utf-8"))
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows_A, rows_B = [], []
    for mk, entry in data.items():
        name = entry.get("display_name", mk)
        rows_A.append([name] + [_macro_subtype_ds(entry, s) for s in _SUBTYPES])
        pr = _macro_rho(entry, "present")
        ar = _macro_rho(entry, "absent")
        gap = _grounding_gap(entry)
        rows_B.append([name, pr, ar, _macro_delta(entry, "present"), gap])

    # ---- A: CSV + LaTeX ----
    def _f(x): return "--" if x is None else f"{x:+.4f}"
    csvA = outdir / f"table_negrealism_{modality}_A.csv"
    with open(csvA, "w") as fh:
        fh.write("model," + ",".join(_SUBTYPES) + "\n")
        for r in rows_A:
            fh.write(r[0] + "," + ",".join(_f(x) for x in r[1:]) + "\n")

    # ---- A with bootstrap CIs (rebuttal T7-T11) ----
    csvA_ci = outdir / f"table_negrealism_{modality}_A_ci.csv"
    with open(csvA_ci, "w") as fh:
        cols = []
        for s in _SUBTYPES:
            cols += [s, f"{s}_lo", f"{s}_hi", f"{s}_n", f"{s}_collapse_certain"]
        fh.write("model," + ",".join(cols) + "\n")
        for mk, entry in data.items():
            name = entry.get("display_name", mk)
            cells = []
            for s in _SUBTYPES:
                m, lo, hi, n = _subtype_ds_ci(entry, s)
                if m is None:
                    cells += ["", "", "", "0", ""]
                else:
                    cells += [f"{m:+.4f}", f"{lo:+.4f}", f"{hi:+.4f}", str(n),
                              str(int(hi < 0))]
            fh.write(name + "," + ",".join(cells) + "\n")

    texA = outdir / f"table_negrealism_{modality}_A.tex"
    with open(texA, "w") as fh:
        fh.write("% Exp C (A): per-subtype DS on present anchors (macro-median over findings)\n")
        fh.write("\\begin{tabular}{l" + "r" * len(_SUBTYPES) + "}\n\\toprule\n")
        fh.write("Model & Definite & Hedged & Ruled-out & Historical \\\\\n\\midrule\n")
        for r in rows_A:
            fh.write(r[0] + " & " + " & ".join(_f(x) for x in r[1:]) + " \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")

    # ---- B: CSV + LaTeX ----
    def _rho(t):
        m, lo, hi = t
        return "--" if m is None else f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"
    csvB = outdir / f"table_negrealism_{modality}_B.csv"
    with open(csvB, "w") as fh:
        fh.write("model,present_rho,present_lo,present_hi,absent_rho,absent_lo,absent_hi,"
                 "grounding_gap,gap_lo,gap_hi,mean_delta\n")
        for name, pr, ar, d, g in rows_B:
            fh.write(",".join([name] + [f"{x:+.4f}" if x is not None else "" for x in
                     [pr[0], pr[1], pr[2], ar[0], ar[1], ar[2], g[0], g[1], g[2], d]]) + "\n")

    texB = outdir / f"table_negrealism_{modality}_B.tex"
    with open(texB, "w") as fh:
        fh.write("% Exp C (B): ladder grounding. Grounding gap = present_rho - absent_rho\n")
        fh.write("% (ideal +2; ~0 = image-blind text-prior ordering). CI = 95% bootstrap.\n")
        fh.write("\\begin{tabular}{lcccr}\n\\toprule\n")
        fh.write("Model & Present $\\rho$ & Absent $\\rho$ & Grounding gap $\\Delta\\rho$ (ideal $+2$) & Mean $\\Delta s$ \\\\\n\\midrule\n")
        for name, pr, ar, d, g in rows_B:
            fh.write(f"{name} & {_rho(pr)} & {_rho(ar)} & {_rho(g)} & {_f(d)} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")

    # ---- console summary ----
    print(f"\n=== Exp C negation realism - {modality} ({len(data)} models) ===")
    print(f"[A] per-subtype DS (present anchors), lower/negative = collapse:")
    hdr = "  " + "model".ljust(16) + "".join(s.rjust(12) for s in _SUBTYPES)
    print(hdr)
    for r in rows_A:
        print("  " + r[0][:16].ljust(16) + "".join(_f(x).rjust(12) for x in r[1:]))
    print(f"\n[B] ladder grounding. Grounding gap Drho = present_rho - absent_rho")
    print(f"    (ideal +2; ~0 = IMAGE-BLIND text-prior ordering, i.e. collapse):")
    print("  " + "model".ljust(16) + "present".rjust(10) + "absent".rjust(10) +
          "gap Drho".rjust(12) + "mean_ds".rjust(11))
    gaps = []
    for name, pr, ar, d, g in rows_B:
        p = "--" if pr[0] is None else f"{pr[0]:+.3f}"
        a = "--" if ar[0] is None else f"{ar[0]:+.3f}"
        gg = "--" if g[0] is None else f"{g[0]:+.3f}"
        if g[0] is not None:
            gaps.append(g[0])
        print("  " + name[:16].ljust(16) + p.rjust(10) + a.rjust(10) + gg.rjust(12) + _f(d).rjust(11))
    if gaps:
        print(f"\n  grounding gap Drho: max={max(gaps):+.3f}  mean={np.mean(gaps):+.3f}  "
              f"(ideal +2.0; all << 2 => negation is image-blind)")
    print(f"\nWrote: {texA.name}, {csvA.name}, {texB.name}, {csvB.name}  (in {outdir})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--modality", required=True)
    ap.add_argument("--outdir", default="experiments/results")
    args = ap.parse_args()
    inp = REPO_ROOT / args.input if not Path(args.input).is_absolute() else Path(args.input)
    build(inp, args.modality, REPO_ROOT / args.outdir
          if not Path(args.outdir).is_absolute() else Path(args.outdir))


if __name__ == "__main__":
    main()
