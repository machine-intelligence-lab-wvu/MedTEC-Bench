"""Write checkpoints_manifest.json: the exact weights behind every number in the paper.

For Hugging Face checkpoints it records the commit revision resolved from the local HF
cache that produced the results, plus the SHA-256 of each weight file. For weights
distributed outside the Hub (CheXzero, MedCLIP) it records the SHA-256 of the local file.

Usage:
    python scripts/make_checkpoints_manifest.py --hf-cache D:/huggingface_cache \
        --weights-root ../MedVLN-Probe [--out checkpoints_manifest.json]
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

WEIGHT_EXT = (".bin", ".safetensors", ".pt", ".pth", ".ckpt")

# (model key, display name, HF repo or None, local weight path relative to --weights-root, note)
MODELS = [
    ("biomedclip", "BiomedCLIP", "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", None, ""),
    ("pmc_clip", "PMC-CLIP", "ryanyip7777/pmc_vit_l_14", None, ""),
    ("pubmedclip", "PubMedCLIP", "flaviagiammarino/pubmed-clip-vit-base-patch32", None, ""),
    ("openclip_vitb32", "OpenCLIP ViT-B/32", "laion/CLIP-ViT-B-32-laion2B-s34B-b79K", None,
     "open_clip ViT-B-32 / laion2b_s34b_b79k"),
    ("openclip_vitl14", "OpenCLIP ViT-L/14", "laion/CLIP-ViT-L-14-laion2B-s32B-b82K", None,
     "open_clip ViT-L-14 / laion2b_s32b_b82k"),
    ("siglip", "SigLIP", "google/siglip-base-patch16-224", None, ""),
    ("medclip", "MedCLIP", None, "pretrained/medclip-vit/pytorch_model.bin",
     "ViT variant from GitHub RyanWangZf/MedCLIP; place at pretrained/medclip-vit/"),
    ("chexzero", "CheXzero", None, "pretrained/chexzero/pytorch_model.bin",
     "GitHub rajpurkarlab/CheXzero release; place at pretrained/chexzero/"),
    ("biovil", "BioViL", "microsoft/BiomedVLP-CXR-BERT-specialized", None,
     "text encoder; image encoder via hi-ml-multimodal get_biovil_image_encoder()"),
    ("conch", "CONCH", "MahmoodLab/conch", None, "gated on Hugging Face; request access"),
    ("plip", "PLIP", "vinid/plip", None, ""),
    ("merlin", "MERLIN", "stanfordmimi/Merlin", None, "loaded via pip install merlin-vlm"),
]


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def readable(p):
    try:
        return p.is_file()
    except OSError:  # dangling symlink in a Windows HF cache
        return False


def find_repo(caches, repo):
    """Return (revision, weight files) from the first cache holding readable weights."""
    name = "models--" + repo.replace("/", "--")
    for c in caches:
        d = c / name
        if not (d / "refs" / "main").exists():
            continue
        rev = (d / "refs" / "main").read_text().strip()
        snap = d / "snapshots" / rev
        files = sorted(p for p in snap.rglob("*") if p.suffix in WEIGHT_EXT and readable(p))
        if files:
            return rev, snap, files
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-cache", required=True, help="HF_HOME (searched in hub/ and transformers/)")
    ap.add_argument("--weights-root", required=True, help="folder holding pretrained/ for non-HF weights")
    ap.add_argument("--extra", default=None, help="optional JSON of extra entries keyed by model key")
    ap.add_argument("--out", default="checkpoints_manifest.json")
    args = ap.parse_args()

    root = Path(args.hf_cache)
    caches = [root / "hub", root / "transformers", root / "huggingface" / "hub"]
    extra = json.load(open(args.extra)) if args.extra else {}
    entries, missing = [], []
    for key, display, repo, local, note in MODELS:
        e = {"model": key, "display_name": display}
        if repo:
            e.update(source="huggingface", repo=repo, url=f"https://huggingface.co/{repo}")
            found = find_repo(caches, repo)
            if found is not None:
                rev, snap, files = found
                e["revision"] = rev
                e["files"] = [{"path": p.relative_to(snap).as_posix(), "bytes": p.stat().st_size,
                               "sha256": sha256(p)} for p in files]
        else:
            p = Path(args.weights_root) / local
            e.update(source="local", path=local)
            if p.exists():
                e["files"] = [{"path": local, "bytes": p.stat().st_size, "sha256": sha256(p)}]
        e.update(extra.get(key, {}))
        if note:
            e["note"] = note
        if "revision" not in e and "files" not in e:
            missing.append(key)
        entries.append(e)
        print(f"{key:16s} {e.get('revision', '-')[:12]:12s} {len(e.get('files', []))} file(s)", flush=True)

    out = {"description": "Exact checkpoints used for every result in MedTEC-Bench (NeurIPS 2026 E&D). "
                          "Pin Hugging Face loads with revision=<revision>; verify files by sha256.",
           "models": entries}
    if missing:
        out["unresolved"] = missing
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print(f"wrote {args.out}" + (f"; unresolved: {missing}" if missing else ""))


if __name__ == "__main__":
    main()
