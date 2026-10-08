"""
Build the Exp A (prompt robustness) table from per-style run_medtec.py outputs.

Reads   <dir>/medtec_<mod>_<style>.json   for style in the --styles list
(default: template report ontology alt_clinical) and, per (model, level in
{4,5}), reports the DS under each style, the SD across styles, and whether the
collapse verdict is stable.

Central claim: if L4/L5 collapse is a property of the encoder (not the wording),
then (a) no style should lift a template-collapsed cell above 0, and (b) the DS
spread across styles should be small next to the collapse magnitude.

Outputs table_promptrobust_<mod>.tex + .csv and a console summary.

Usage:
    python scripts/_build_prompt_robustness_table.py --modality cxr \
        --dir experiments/results/promptrobust
"""

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
_SEED, _B = 42, 10000
PASS = 0.05   # run_medtec's "OK" boundary — DS above this = meaningful discrimination


def _boot_ci(vals, seed=_SEED, B=_B):
    vals = np.asarray(vals, dtype=np.float64)
    if len(vals) == 0:
        return (None, None, None)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(B, len(vals)))
    means = vals[idx].mean(axis=1)
    return (float(vals.mean()), float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))


def _ds_for(model_entry, level):
    """Return (mean_DS, per_probe_DS list) for a level, or (None, [])."""
    lv = model_entry.get(f"L{level}")
    if not lv:
        return None, []
    per = [p["discrimination_score"] for p in lv.get("per_probe", [])]
    mean = lv.get("avg_discrimination_score",
                  (float(np.mean(per)) if per else None))
    return mean, per


def build(modality, dirpath, styles, levels, outdir):
    dirpath = Path(dirpath)
    data = {}
    for st in styles:
        fp = dirpath / f"medtec_{modality}_{st}.json"
        if fp.exists():
            data[st] = json.load(open(fp, encoding="utf-8"))
        else:
            print(f"  [warn] missing {fp.name} — style '{st}' skipped")
    if not data:
        raise SystemExit(f"No per-style files found in {dirpath}")
    present_styles = list(data)

    # union of models across styles, preserve first-seen order
    models = []
    for st in present_styles:
        for mk in data[st]:
            if mk not in models:
                models.append(mk)

    rows = []          # (display, level, {style: (mean,lo,hi)}, sd, rescued, consistent)
    for mk in models:
        name = next((data[st][mk].get("display_name", mk)
                     for st in present_styles if mk in data[st]), mk)
        for level in levels:
            per_style = {}
            for st in present_styles:
                if mk in data[st]:
                    m, per = _ds_for(data[st][mk], level)
                    if m is not None:
                        per_style[st] = _boot_ci(per) if per else (m, None, None)
            if not per_style:
                continue
            means = [v[0] for v in per_style.values()]
            sd = float(np.std(means)) if len(means) > 1 else 0.0
            tmpl = per_style.get("template", (means[0], None, None))[0]
            # "rescued" = template collapsed AND some style reaches MEANINGFUL
            # discrimination (DS > PASS). PASS=0.05 is run_medtec's "OK" boundary;
            # near-zero sign flips (|DS|<PASS) are chance-level, not rescue.
            rescued = (tmpl is not None and tmpl <= 0 and
                       any(m > PASS for m in means))
            consistent = all(m <= 0 for m in means) or all(m > 0 for m in means)
            rows.append((name, level, per_style, sd, rescued, consistent))

    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    def _c(t):
        m, lo, hi = t
        if m is None:
            return "--"
        return f"{m:+.4f}" if lo is None else f"{m:+.4f} [{lo:+.4f},{hi:+.4f}]"

    # CSV
    csv = outdir / f"table_promptrobust_{modality}.csv"
    with open(csv, "w") as fh:
        fh.write("model,level," + ",".join(present_styles) +
                 ",sd_across_styles,rescued,verdict_consistent\n")
        for name, lv, ps, sd, resc, cons in rows:
            fh.write(f"{name},L{lv}," +
                     ",".join(f"{ps[st][0]:+.4f}" if st in ps else "" for st in present_styles) +
                     f",{sd:.4f},{int(resc)},{int(cons)}\n")

    # CSV with bootstrap 95% CIs (rebuttal T1-T6): one lo/hi pair per style, plus
    # the collapse-certain flag (CI strictly < 0) that the paper uses in App. E.4.
    csv_ci = outdir / f"table_promptrobust_{modality}_ci.csv"
    with open(csv_ci, "w") as fh:
        cols = []
        for st in present_styles:
            cols += [f"{st}", f"{st}_lo", f"{st}_hi", f"{st}_collapse_certain"]
        fh.write("model,level," + ",".join(cols) + ",sd_across_styles,rescued\n")
        for name, lv, ps, sd, resc, cons in rows:
            cells = []
            for st in present_styles:
                if st not in ps:
                    cells += ["", "", "", ""]
                    continue
                m, lo, hi = ps[st]
                certain = int(hi is not None and hi < 0)
                cells += [f"{m:+.4f}",
                          "" if lo is None else f"{lo:+.4f}",
                          "" if hi is None else f"{hi:+.4f}",
                          str(certain)]
            fh.write(f"{name},L{lv}," + ",".join(cells) + f",{sd:.4f},{int(resc)}\n")

    # LaTeX
    tex = outdir / f"table_promptrobust_{modality}.tex"
    with open(tex, "w") as fh:
        fh.write("% Exp A prompt robustness: DS per style, SD across styles.\n")
        fh.write("\\begin{tabular}{ll" + "r" * len(present_styles) + "r}\n\\toprule\n")
        fh.write("Model & Lvl & " + " & ".join(present_styles) + " & SD \\\\\n\\midrule\n")
        for name, lv, ps, sd, resc, cons in rows:
            cells = " & ".join(f"{ps[st][0]:+.4f}" if st in ps else "--"
                               for st in present_styles)
            fh.write(f"{name} & L{lv} & {cells} & {sd:.4f} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")

    # console + headline stats
    print(f"\n=== Exp A prompt robustness - {modality} (styles: {present_styles}) ===")
    hdr = "  " + "model".ljust(16) + "lvl".ljust(5) + "".join(s[:10].rjust(11) for s in present_styles) + "SD".rjust(9) + "  flags"
    print(hdr)
    for name, lv, ps, sd, resc, cons in rows:
        cells = "".join((f"{ps[st][0]:+.4f}" if st in ps else "--").rjust(11) for st in present_styles)
        flags = ("RESCUED " if resc else "") + ("" if cons else "SPLIT")
        print("  " + name[:16].ljust(16) + f"L{lv}".ljust(5) + cells + f"{sd:.4f}".rjust(9) + "  " + flags)

    n = len(rows)
    rescued_n = sum(r[4] for r in rows)
    max_sd = max((r[3] for r in rows), default=0.0)
    # max DS achieved by ANY (cell, style) — headline: does any wording reach PASS?
    max_ds = max((v[0] for r in rows for v in r[2].values() if v[0] is not None),
                 default=0.0)
    collapsed_template = [r for r in rows if r[2].get("template", (1,))[0] is not None
                          and r[2]["template"][0] <= 0]
    held = sum(1 for r in collapsed_template if not r[4])
    print(f"\n  cells={n}  max SD across styles={max_sd:.4f}")
    print(f"  MAX DS across all (cell x style) at L4/L5 = {max_ds:+.4f}  "
          f"(PASS boundary = {PASS}; {'NONE reach it' if max_ds <= PASS else 'SOME reach it'})")
    print(f"  cells 'rescued' to real discrimination (DS>{PASS}) by any style: {rescued_n}/{n}")
    if collapsed_template:
        print(f"  template-collapsed cells that STAY <= {PASS} under all styles: "
              f"{held}/{len(collapsed_template)}")
    print(f"  Wrote {tex.name}, {csv.name} (in {outdir})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", required=True)
    ap.add_argument("--dir", default="experiments/results/promptrobust")
    ap.add_argument("--styles", nargs="+",
                    default=["template", "report", "ontology", "alt_clinical"])
    ap.add_argument("--levels", nargs="+", type=int, default=[4, 5])
    ap.add_argument("--outdir", default="experiments/results")
    args = ap.parse_args()
    build(args.modality,
          REPO_ROOT / args.dir if not Path(args.dir).is_absolute() else Path(args.dir),
          args.styles, args.levels,
          REPO_ROOT / args.outdir if not Path(args.outdir).is_absolute() else Path(args.outdir))


if __name__ == "__main__":
    main()
