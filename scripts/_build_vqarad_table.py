"""
Build the Exp B6 (VQA-RAD) tables from run_vqarad.py output.

Reports, per model and prompt protocol:
  acc / acc_yes / acc_no / balanced_acc   (chance = 0.500)
  yes_rate            vs the gold yes-rate (~0.49). A model whose negation handling
                      has collapsed answers from a fixed polarity prior, so yes_rate
                      drifts far from gold while balanced_acc sits at chance.
  shuffle_agreement   how often the predicted answer is UNCHANGED when the image is
                      replaced by an unrelated scan. 1.0 = the answer never depended
                      on the image.

Also: accuracy stratified by the MedTEC level each question interrogates. The level
tags are rule-based (see _prep_vqarad.py) and the full tagged list is released as
vqarad_level_tagging_review.csv, so this stratification is reported as exploratory.

Usage:
    python scripts/_build_vqarad_table.py
"""
import csv, json
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent
RES = REPO / "experiments" / "results"
SEED, B = 42, 10000
LEVELS = ["L2", "L3", "L4", "L5", "other"]


def _ranks(a):
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), float); r[order] = np.arange(1, len(a) + 1)
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    s = np.zeros(len(cnt)); np.add.at(s, inv, r)
    return (s / cnt)[inv]


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y)); x, y = x[ok], y[ok]
    if len(x) < 3:
        return None, len(x)
    rx, ry = _ranks(x), _ranks(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return (float((rx * ry).sum() / d) if d else None), len(x)


def boot_ci(vals, seed=SEED, n=B):
    v = np.asarray(vals, float)
    if len(v) == 0:
        return (None, None, None)
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    res = json.load(open(RES / "vqarad" / "vqarad_results.json", encoding="utf-8"))

    for proto in ("qa", "decl"):
        rows = []
        for mk, ent in res.items():
            if proto not in ent:
                continue
            blk = ent[proto]; o = blk["overall"]
            items = blk["items"]
            # bootstrap CI on balanced accuracy via the per-item correctness in each arm
            cy = [int(r["pred_yes"] == r["gold_yes"]) for r in items if r["gold_yes"] == 1]
            cn = [int(r["pred_yes"] == r["gold_yes"]) for r in items if r["gold_yes"] == 0]
            ay, aylo, ayhi = boot_ci(cy); an, anlo, anhi = boot_ci(cn)
            bal = 0.5 * (ay + an)
            rng = np.random.default_rng(SEED)
            by = np.asarray(cy, float)[rng.integers(0, len(cy), size=(B, len(cy)))].mean(1)
            bn = np.asarray(cn, float)[rng.integers(0, len(cn), size=(B, len(cn)))].mean(1)
            bb = 0.5 * (by + bn)
            # Shuffle agreement is mechanically inflated by a model's own polarity
            # bias: a model that answers "no" 80% of the time agrees with itself
            # 0.68 of the time by chance alone. Report the excess over that baseline.
            p = o["yes_rate"]
            chance_agree = p * p + (1 - p) * (1 - p)
            excess = ((blk["shuffle_agreement"] - chance_agree) / (1 - chance_agree)
                      if chance_agree < 1 else None)
            rows.append({
                "chance_agree": chance_agree, "excess_agree": excess,
                "model": ent["display_name"], "n": o["n"],
                "acc": o["acc"], "acc_yes": ay, "acc_no": an,
                "bal": bal, "bal_lo": float(np.percentile(bb, 2.5)),
                "bal_hi": float(np.percentile(bb, 97.5)),
                "yes_rate": o["yes_rate"], "gold_yes": o["gold_yes_rate"],
                "shuffle": blk["shuffle_agreement"],
                "by_level": {lv: blk["by_level"].get(lv, {}).get("acc") for lv in LEVELS},
                "by_level_n": {lv: blk["by_level"].get(lv, {}).get("n") for lv in LEVELS},
            })
        if not rows:
            continue
        rows.sort(key=lambda r: -r["bal"])

        print(f"\n=== Exp B6  VQA-RAD closed-form yes/no  [protocol = {proto}] ===")
        print("  " + "model".ljust(19) + "n".rjust(6) + "acc".rjust(7) + "acc_y".rjust(7) +
              "acc_n".rjust(7) + "bal [95% CI]".rjust(22) + "yes_rt".rjust(8) + "shuf".rjust(7))
        for r in rows:
            print("  " + r["model"][:19].ljust(19) + str(r["n"]).rjust(6) +
                  f"{r['acc']:.3f}".rjust(7) + f"{r['acc_yes']:.3f}".rjust(7) +
                  f"{r['acc_no']:.3f}".rjust(7) +
                  f"{r['bal']:.3f} [{r['bal_lo']:.3f},{r['bal_hi']:.3f}]".rjust(22) +
                  f"{r['yes_rate']:.3f}".rjust(8) + f"{r['shuffle']:.3f}".rjust(7))
        bals = [r["bal"] for r in rows]
        above = [r for r in rows if r["bal_lo"] > 0.5]
        print(f"\n  gold yes-rate = {rows[0]['gold_yes']:.3f}   chance balanced acc = 0.500")
        print(f"  balanced acc range {min(bals):.3f} to {max(bals):.3f}   "
              f"models whose 95% CI clears chance: {len(above)}/{len(rows)}")
        print(f"  mean shuffle agreement (answer unchanged with a random image): "
              f"{np.mean([r['shuffle'] for r in rows]):.3f}"
              f"  (bias-expected {np.mean([r['chance_agree'] for r in rows]):.3f},"
              f" excess over bias {np.mean([r['excess_agree'] for r in rows]):.3f})")
        print("  " + "model".ljust(19) + "shuffle".rjust(9) + "bias-exp".rjust(10) +
              "excess".rjust(9))
        for r in rows:
            print("  " + r["model"][:19].ljust(19) + f"{r['shuffle']:.3f}".rjust(9) +
                  f"{r['chance_agree']:.3f}".rjust(10) + f"{r['excess_agree']:.3f}".rjust(9))

        print("\n  accuracy by MedTEC level of the question (exploratory, rule-based tags)")
        print("  " + "model".ljust(19) + "".join(lv.rjust(9) for lv in LEVELS))
        for r in rows:
            print("  " + r["model"][:19].ljust(19) +
                  "".join((f"{r['by_level'][lv]:.3f}" if r["by_level"].get(lv) is not None
                           else "--").rjust(9) for lv in LEVELS))
        print("  " + "n items".ljust(19) +
              "".join(str(rows[0]["by_level_n"].get(lv) or "--").rjust(9) for lv in LEVELS))
        for lv in LEVELS:
            v = [r["by_level"][lv] for r in rows if r["by_level"].get(lv) is not None]
            if v:
                print(f"    {lv}: mean acc across models = {np.mean(v):.3f}")

        csvp = RES / f"table_vqarad_{proto}.csv"
        with open(csvp, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["model", "n", "acc", "acc_yes", "acc_no", "balanced_acc",
                        "bal_lo", "bal_hi", "yes_rate", "gold_yes_rate",
                        "shuffle_agreement"] + [f"acc_{lv}" for lv in LEVELS])
            for r in rows:
                w.writerow([r["model"], r["n"], f"{r['acc']:.4f}", f"{r['acc_yes']:.4f}",
                            f"{r['acc_no']:.4f}", f"{r['bal']:.4f}", f"{r['bal_lo']:.4f}",
                            f"{r['bal_hi']:.4f}", f"{r['yes_rate']:.4f}",
                            f"{r['gold_yes']:.4f}", f"{r['shuffle']:.4f}"] +
                           [("" if r["by_level"].get(lv) is None else f"{r['by_level'][lv]:.4f}")
                            for lv in LEVELS])
        print(f"  wrote {csvp.name}")

        # correlation with MedTEC CXR L5 DS (the reviewer's requested cross-model link)
        med = json.load(open(RES / "medtec_cxr_n1000.json", encoding="utf-8"))
        xs, ys, used = [], [], []
        for mk, ent in res.items():
            if proto not in ent or mk not in med:
                continue
            l5 = med[mk].get("L5", {}).get("avg_discrimination_score")
            if l5 is None:
                continue
            r = next((q for q in rows if q["model"] == ent["display_name"]), None)
            if r:
                xs.append(l5); ys.append(r["bal"]); used.append(ent["display_name"])
        if len(xs) >= 3:
            rho, n = spearman(xs, ys)
            print(f"  cross-model Spearman(MedTEC CXR L5 DS, VQA balanced acc) = "
                  f"{rho:+.3f}  n={n}  [{', '.join(used)}]")


if __name__ == "__main__":
    main()
