"""
Build the disease-labeled abdominal-CT probe set from the Merlin (Stanford AIMI)
dataset, streaming one volume at a time to respect disk limits.

For each selected anchor scan (single-finding, from the Merlin zero-shot label CSV):
  1. download {study_id}.nii.gz from the container SAS URL to a temp file
  2. 2D track  -> extract a representative windowed axial slice -> images/{id}.png
  3. 3D track  -> Merlin.encode_image(volume) -> merlin_emb/{id}.npy  (512-d, L2-normalized)
  4. delete the volume

Runs in a separate environment (needs merlin-vlm, monai, nibabel, requests, torch+cu128).

Usage:
  python scripts/build_ct_disease_probes.py \
      --sas-file <path to file containing the container SAS url> \
      --selection data/ct_disease_probes/selection_oversampled.json \
      --per-class 150 --out-dir data/ct_disease_probes --device cuda
  # smoke test: add  --limit-per-class 2
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse, json, tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests
import nibabel as nib
import torch
from PIL import Image


def windowed_axial_png(nii_path: str, level: int = 40, width: int = 400) -> Image.Image:
    """Middle axial slice, soft-tissue window (L=40/W=400), RGB PNG.
    NOTE: label is scan-level; the slice is a *representative* central slice, not
    localized to the finding (documented caveat for the 2D track)."""
    img = nib.as_closest_canonical(nib.load(nii_path))   # RAS
    vol = img.get_fdata()
    z = vol.shape[2] // 2
    sl = vol[:, :, z]
    sl = np.rot90(sl)                                    # standard axial display
    lo, hi = level - width / 2.0, level + width / 2.0
    sl = np.clip((sl - lo) / (hi - lo), 0, 1)
    arr = (sl * 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sas-file", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--per-class", type=int, default=150)
    ap.add_argument("--out-dir", default="data/ct_disease_probes")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit-per-class", type=int, default=None,
                    help="smoke test: process at most this many per class")
    args = ap.parse_args()

    sas = open(args.sas_file).read().strip()
    base, q = sas.split("?", 1)
    target = args.limit_per_class or args.per_class

    out = Path(args.out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "merlin_emb").mkdir(parents=True, exist_ok=True)

    sel = json.load(open(args.selection))
    by_class = defaultdict(list)
    for r in sel:
        by_class[r["disease_category"]].append(r["study_id"])

    # load Merlin once
    from merlin import Merlin
    from merlin.data.monai_transforms import ImageTransforms
    dev = args.device if torch.cuda.is_available() else "cpu"
    print(f"device={dev}; loading Merlin ...", flush=True)
    model = Merlin(); model.eval().to(dev)

    manifest = []
    done = defaultdict(int); attempted = defaultdict(int); failed = []
    for cls, ids in by_class.items():
        for sid in ids:
            if done[cls] >= target:
                break
            attempted[cls] += 1
            url = f"{base}/merlin_data/{sid}.nii.gz?{q}"
            tmp = None
            try:
                r = requests.get(url, timeout=600)
                if r.status_code != 200:
                    failed.append((sid, cls, f"http{r.status_code}")); continue
                fd, tmp = tempfile.mkstemp(suffix=".nii.gz"); os.close(fd)
                open(tmp, "wb").write(r.content)

                # 2D slice
                png = windowed_axial_png(tmp)
                png_path = out / "images" / f"{sid}.png"
                png.save(png_path)

                # 3D Merlin embedding
                vol = ImageTransforms({"image": tmp})["image"].unsqueeze(0).to(dev)
                with torch.no_grad():
                    cf, _ehr = model.model.encode_image(vol)
                    cf = cf / cf.norm(dim=-1, keepdim=True)
                emb = cf.float().cpu().numpy().squeeze()
                np.save(out / "merlin_emb" / f"{sid}.npy", emb)

                manifest.append({
                    "image_id": sid,
                    "study_id": sid,
                    "disease_category": cls,
                    "image_path": f"{args.out_dir}/images/{sid}.png",
                    "merlin_emb": f"{args.out_dir}/merlin_emb/{sid}.npy",
                    "source": "merlin_abdominal_ct",
                })
                done[cls] += 1
                if done[cls] % 25 == 0 or done[cls] == target:
                    print(f"  {cls:18s} {done[cls]}/{target} (attempted {attempted[cls]})", flush=True)
            except Exception as e:
                failed.append((sid, cls, type(e).__name__ + ":" + str(e)[:80]))
            finally:
                if tmp and os.path.exists(tmp):
                    os.remove(tmp)
        print(f"[class done] {cls}: {done[cls]}/{target}", flush=True)

    json.dump(manifest, open(out / "manifest.json", "w"), indent=1)
    summary = {"per_class_done": dict(done), "per_class_attempted": dict(attempted),
               "n_total": len(manifest), "n_failed": len(failed), "failed": failed[:50]}
    json.dump(summary, open(out / "build_summary.json", "w"), indent=2)
    print(f"\nDONE: {len(manifest)} anchors -> {out/'manifest.json'}; failed={len(failed)}", flush=True)
    print("per-class:", dict(done), flush=True)


if __name__ == "__main__":
    main()
