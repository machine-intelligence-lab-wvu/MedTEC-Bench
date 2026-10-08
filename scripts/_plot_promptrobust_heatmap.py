"""
Exp A prompt-robustness DS heatmap: per modality, models x {L4,L5} x 4 styles.
Diverging colormap centered at 0 (blue=collapse, red=discriminate, white=chance);
every cell annotated with its DS value (identity never color-alone -> CVD/print safe);
cells reaching the DS>0.05 discrimination boundary are outlined. n ~= 1000 anchor
images per cell. Reads experiments/results/table_promptrobust_<mod>.csv.

Render with:
    python scripts/_plot_promptrobust_heatmap.py
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import TwoSlopeNorm

REPO = Path(__file__).parent.parent
RES = REPO / "experiments" / "results"
OUT = REPO / "experiments" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

MODS = [("cxr", "Chest X-ray"), ("fundus", "Retinal fundus"),
        ("dermoscopy", "Dermoscopy"), ("abdomen_ct", "Abdomen CT"),
        ("brain_mri", "Brain MRI"), ("histopath", "Histopathology")]
STYLES = ["template", "report", "ontology", "alt_clinical"]
STYLE_SHORT = ["tmpl", "rep", "ont", "alt"]
PASS = 0.05


def short(name):
    return (name.replace("OpenCLIP ViT-B-32", "OpenCLIP-B32")
                .replace("OpenCLIP ViT-L-14", "OpenCLIP-L14"))


def load(mod):
    rows = {}
    with open(RES / f"table_promptrobust_{mod}.csv") as fh:
        for r in csv.DictReader(fh):
            rows.setdefault(r["model"], {})[r["level"]] = [float(r[s]) for s in STYLES]
    models = list(rows)
    # matrix rows=models, cols = L4[4 styles] + L5[4 styles]
    M = np.full((len(models), 8), np.nan)
    for i, m in enumerate(models):
        if "L4" in rows[m]:
            M[i, 0:4] = rows[m]["L4"]
        if "L5" in rows[m]:
            M[i, 4:8] = rows[m]["L5"]
    return models, M


# global symmetric color scale
allvals = []
data = {}
for mod, _ in MODS:
    models, M = load(mod)
    data[mod] = (models, M)
    allvals += list(M[~np.isnan(M)])
vmax = max(0.09, float(np.nanmax(np.abs(allvals))))
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
cmap = plt.get_cmap("RdBu_r").copy()
cmap.set_bad("#dddddd")

fig, axes = plt.subplots(3, 2, figsize=(13.5, 15.5))
fig.suptitle("MedTEC discrimination score is invariant to prompt wording\n"
             "L4 finding + L5 negation, 4 phrasing styles, n$\\approx$1000 anchor images/cell "
             "(blue = collapse, white = chance, red = discriminates)",
             fontsize=13, fontweight="bold", y=0.985)

for ax, (mod, title) in zip(axes.ravel(), MODS):
    models, M = data[mod]
    im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(8))
    ax.set_xticklabels(STYLE_SHORT * 2, fontsize=8)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([short(m) for m in models], fontsize=8)
    # L4 | L5 group labels + divider
    ax.axvline(3.5, color="black", lw=1.5)
    ax.text(1.5, -0.85, "L4 finding", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.text(5.5, -0.85, "L5 negation", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=18)
    # annotate values; outline cells reaching the discrimination boundary
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                continue
            tc = "white" if abs(v) > 0.55 * vmax else "black"
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=6.5, color=tc)
            if v >= PASS:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor="#111111", lw=2.0))
    ax.set_xticks(np.arange(-.5, 8, 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(models), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.2)
    ax.tick_params(which="minor", length=0)

cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.04, ticks=[-vmax, -PASS, 0, PASS, vmax])
cbar.set_label("Discrimination Score (DS)   —   DS $\\leq$ 0 = collapse", fontsize=10)
cbar.ax.axhline(PASS, color="#111111", lw=1.5)
cbar.set_ticklabels([f"{-vmax:.2f}", f"-{PASS}", "0", f"+{PASS}", f"+{vmax:.2f}"])
fig.text(0.5, 0.005,
         "Black outline = DS $\\geq$ 0.05 (real discrimination). "
         "Only easier modalities (derm/histo) and in-domain specialists (CONCH) ever cross it; "
         "no wording rescues a collapsed cell.",
         ha="center", fontsize=8.5, style="italic")

png = OUT / "promptrobust_ds_heatmap.png"
fig.savefig(png, dpi=150, bbox_inches="tight", facecolor="white")
print("wrote", png)
