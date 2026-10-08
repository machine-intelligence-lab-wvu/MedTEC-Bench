"""
N-stability analysis: how the MedTEC discrimination score (DS) and its
bootstrap 95% CI change with the number of anchor images.

Per-image DS values are read from a full (per-probe) MedTEC result JSON,
subsampled without replacement at each n (seed 42), and summarised as the
mean DS with a percentile bootstrap CI over images (B = 10,000).

Usage:
    python scripts/_n_stability.py \
        --input experiments/results/full_per_probe/medtec_cxr_n1000.json \
        --models biomedclip pmc_clip siglip --levels 4 5
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np


def per_image_ds(level_dict):
    """Mean DS per anchor image (averaging over the probes that share it)."""
    acc = defaultdict(list)
    for p in level_dict["per_probe"]:
        acc[p["image_path"]].append(p["discrimination_score"])
    keys = sorted(acc)
    return np.array([np.mean(acc[k]) for k in keys])


def boot_ci(x, B, rng):
    idx = rng.integers(0, len(x), size=(B, len(x)))
    m = x[idx].mean(axis=1)
    return np.percentile(m, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--models", nargs="+", default=["biomedclip", "pmc_clip", "siglip"])
    ap.add_argument("--levels", nargs="+", type=int, default=[4, 5])
    ap.add_argument("--sizes", nargs="+", type=int, default=[100, 250, 500, 1000])
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="experiments/results/n_stability_cxr.json")
    args = ap.parse_args()

    res = json.loads(Path(args.input).read_text(encoding="utf-8"))
    out = {}
    print(f"{'model, level':<18}" + "".join(f"{'n=' + str(n):>12}" for n in args.sizes) + "   CI width")
    for m in args.models:
        for L in args.levels:
            x = per_image_ds(res[m][f"L{L}"])
            row, widths = {}, {}
            for n in args.sizes:
                rng = np.random.default_rng(args.seed)
                sub = x if n >= len(x) else x[rng.choice(len(x), n, replace=False)]
                lo, hi = boot_ci(sub, args.n_boot, np.random.default_rng(args.seed))
                row[n] = {"ds": float(sub.mean()), "ci": [float(lo), float(hi)], "n": int(len(sub))}
                widths[n] = hi - lo
            out[f"{m}_L{L}"] = row
            print(f"{m + ', L' + str(L):<18}" + "".join(f"{row[n]['ds']:>+12.4f}" for n in args.sizes)
                  + f"   {widths[args.sizes[0]]:.4f} -> {widths[args.sizes[-1]]:.4f}")
    Path(args.output).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
