"""Camera-ready Fig. 3: per-modality radar of MedTEC DS (spokes = models, polygons = levels).

Changes vs. submission figure (reviewers qunz, eTVi):
  * adds the disease-labeled abdominal-CT (Merlin) subplot with the MERLIN 3D specialist;
  * levels differ by line style AND marker, not only colour (grayscale / colour-blind safe).
Visibility pass:
  * drawn at print size (5.5 in = NeurIPS \\linewidth), so fonts are true 6-8 pt in the PDF;
  * signed square-root radial axis (SQRT=True) so the near-zero L2-L5 band is readable;
    tick labels show true DS values;
  * shaded disc = DS < 0 (collapse side), grey dashed ring = 0.05 discrimination boundary;
  * L4/L5 drawn on top.
Reads experiments/results/camera_ready_verdicts.json (written by _camera_ready_verdicts.py).
Run with the `lgmd` env (matplotlib savefig crashes in lm_mi):
    python scripts/_fig3_radar_camera_ready.py <out_dir>
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "experiments/figures"
V = json.load(open("experiments/results/camera_ready_verdicts.json"))
SQRT = True  # False -> linear radial axis

PANELS = [("cxr", "Chest X-ray"), ("brain_mri", "Brain MRI"), ("abdomen_ct_disease", "Abdominal CT"),
          ("fundus", "Retinal fundus"), ("dermoscopy", "Dermoscopy"), ("histopath", "Histopathology")]
SHARED = ["biomedclip", "pmc_clip", "pubmedclip", "openclip_vitb32", "openclip_vitl14", "siglip", "medclip"]
EXTRA = {"cxr": ["chexzero", "biovil"], "histopath": ["conch", "plip"], "abdomen_ct_disease": ["merlin"]}
NAMES = {"biomedclip": "BiomedCLIP", "pmc_clip": "PMC-CLIP", "pubmedclip": "PubMedCLIP",
         "openclip_vitb32": "OpenCLIP\nB/32", "openclip_vitl14": "OpenCLIP\nL/14", "siglip": "SigLIP",
         "medclip": "MedCLIP", "chexzero": "CheXzero", "biovil": "BioViL", "conch": "CONCH",
         "plip": "PLIP", "merlin": "MERLIN\n(3D)"}
# modality specialists per panel (bold green labels)
SPEC = {"cxr": {"medclip", "chexzero", "biovil"}, "histopath": {"conch", "plip"},
        "abdomen_ct_disease": {"merlin"}}
# colour-blind-safe (Okabe-Ito) + distinct dash pattern + marker per level; (label, colour, ls, marker, lw, z)
LEVELS = [("L1 domain", "#0072B2", "-", "o", 1.0, 3), ("L2 modality", "#E69F00", "--", "s", 1.0, 3),
          ("L3 anatomy", "#009E73", "-.", "^", 1.0, 3), ("L4 finding", "#CC79A7", ":", "D", 1.4, 4),
          ("L5 negation", "#000000", (0, (4, 1, 1, 1)), "v", 1.2, 5)]
BOUNDARY = 0.05


def tr(x):
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.sqrt(np.abs(x)) if SQRT else x


RMIN, RMAX = tr(-0.32), tr(0.26)
TICKS = [-0.25, 0.0, 0.05, 0.25] if SQRT else [-0.25, 0.0, 0.25]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5})
fig, axes = plt.subplots(2, 3, figsize=(5.5, 3.45), subplot_kw=dict(polar=True))
ring = np.linspace(0, 2 * np.pi, 361)
for ax, (key, title) in zip(axes.flat, PANELS):
    models = [m for m in SHARED + EXTRA.get(key, []) if m in V[key]]
    ang = np.linspace(0, 2 * np.pi, len(models), endpoint=False)
    closed = np.append(ang, ang[0])
    # collapse side (DS < 0) shaded; 0.05 boundary ring
    ax.fill_between(ring, RMIN, 0, color="#d62728", alpha=0.07, lw=0, zorder=0)
    ax.plot(ring, np.zeros_like(ring), color="#d62728", lw=0.9, zorder=2)
    ax.plot(ring, np.full_like(ring, tr(BOUNDARY)), color="0.45", ls=(0, (2, 1.5)), lw=0.7, zorder=2)
    for L, col, ls, mk, lw, z in LEVELS:
        r = tr([max(V[key][m][L.split()[0]]["mean"], -0.32) for m in models])
        ax.plot(closed, np.append(r, r[0]), color=col, ls=ls, lw=lw, marker=mk, ms=2.6,
                mew=0, zorder=z, label=L)
    ax.set_ylim(RMIN, RMAX)
    ax.set_yticks(tr(TICKS))
    ax.set_yticklabels([])  # rings are named in the legend; exact values are in Tables 4, 8, 18
    ax.set_xticks(ang)
    ax.set_xticklabels([])
    for m, a in zip(models, ang):
        # spoke labels drawn by hand, anchored on the side facing away from the rim
        c, s = np.cos(a), np.sin(a)
        spec = m in SPEC.get(key, set())
        ax.text(a, RMAX + 0.04, NAMES[m], fontsize=6, linespacing=0.9, clip_on=False,
                ha="left" if c > 0.15 else "right" if c < -0.15 else "center",
                va="bottom" if s > 0.3 else "top" if s < -0.3 else "center",
                color="#1a7f1a" if spec else "black", fontweight="bold" if spec else "normal")
    ax.grid(color="0.8", lw=0.4)
    ax.spines["polar"].set_linewidth(0.5)
    ax.set_title(title, fontsize=7.5, fontweight="bold", pad=17)

h, l = axes.flat[0].get_legend_handles_labels()
h += [plt.Line2D([], [], color="#d62728", lw=0.9),
      plt.Line2D([], [], color="0.45", ls=(0, (2, 1.5)), lw=0.7)]
l += ["DS = 0 (shaded: DS < 0)", f"DS = {BOUNDARY} boundary"]
order = [0, 4, 1, 5, 2, 6, 3]  # legend fills column-wise; this reads row-wise L1-L4 / L5, rings
h, l = [h[i] for i in order], [l[i] for i in order]
fig.legend(h, l, loc="lower center", ncol=4, fontsize=6, frameon=False, handlelength=3.2,
           columnspacing=1.2, bbox_to_anchor=(0.5, -0.01))
fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.13, wspace=1.2, hspace=0.62)
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/main_results_radar_cr.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
print("wrote", OUT)
