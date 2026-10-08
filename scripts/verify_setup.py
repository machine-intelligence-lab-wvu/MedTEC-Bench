"""
Fast setup verification — no model loading, no GPU required.

Checks for every model:
  1. Required Python packages are importable
  2. Local weight files exist (where applicable)
  3. HuggingFace models are already cached locally
  4. HF token is set for gated models

Statuses:
  READY    — everything in place, model can be loaded offline
  DOWNLOAD — package OK, but weights not cached yet (will auto-download on first use)
  ACTION   — manual step required before model can be used
  MISSING  — required package or file not found

Usage:
  python scripts/verify_setup.py
  python scripts/verify_setup.py --fix   # print exact commands to fix each issue
"""

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Optional

# ── locate project root (works whether run from root or scripts/) ───────────
ROOT = Path(__file__).resolve().parent.parent

# ── HuggingFace cache helpers ────────────────────────────────────────────────

def _hf_cache_root() -> Path:
    """Return the HuggingFace hub cache directory."""
    return Path(os.environ.get("HF_HOME",
                os.environ.get("HUGGINGFACE_HUB_CACHE",
                Path.home() / ".cache" / "huggingface" / "hub")))


def _is_hf_cached(repo_id: str) -> bool:
    """
    Return True if the repo appears to be fully (or partially) cached locally.
    Uses huggingface_hub.scan_cache_dir when available; falls back to a
    simple directory check.
    """
    try:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir()
        cached_ids = {r.repo_id for r in info.repos}
        return repo_id in cached_ids
    except Exception:
        # Directory-based fallback: HF hub stores models as
        # <cache>/models--<org>--<name>/snapshots/...
        safe_name = "models--" + repo_id.replace("/", "--")
        snap_dir = _hf_cache_root() / safe_name / "snapshots"
        if snap_dir.exists():
            return any(snap_dir.iterdir())
        return False


def _hf_token() -> Optional[str]:
    """Return the HuggingFace token from env or ~/.huggingface/token."""
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        token_file = Path.home() / ".huggingface" / "token"
        if token_file.exists():
            token = token_file.read_text().strip() or None
    return token


def _pkg_ok(name: str) -> bool:
    """Return True if the package can be imported."""
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


def _pkg_version(name: str) -> str:
    try:
        import importlib.metadata
        return importlib.metadata.version(name)
    except Exception:
        return "?"


def _file_ok(rel_path: str) -> bool:
    return (ROOT / rel_path).exists()


# ── open_clip LAION weight cache check ──────────────────────────────────────

_OPENCLIP_LAION_HF: dict[str, str] = {
    # (model_id, pretrained) → HF repo_id that open_clip downloads from
    ("ViT-B-32", "laion2b_s34b_b79k"): "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    ("ViT-L-14", "laion2b_s32b_b82k"): "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
}


# ── per-model check spec ─────────────────────────────────────────────────────

CHECKS = {
    # key: {
    #   display      – human name
    #   packages     – list of importable names to verify
    #   local_files  – list of relative paths that must exist
    #   hf_repos     – list of HF repo IDs to check in cache
    #   openclip     – (model_id, pretrained) for LAION models
    #   gated        – True if HF token + access request needed
    #   notes        – extra info line
    # }
    "biomedclip": {
        "display":     "BiomedCLIP",
        "packages":    ["open_clip"],
        "hf_repos":    ["microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"],
    },
    "pubmedclip": {
        "display":     "PubMedCLIP",
        "packages":    ["transformers"],
        "hf_repos":    ["flaviagiammarino/pubmed-clip-vit-base-patch32"],
    },
    "medclip": {
        "display":     "MedCLIP",
        "packages":    ["medclip"],
        "local_files": ["pretrained/medclip-vit/pytorch_model.bin"],
        "notes":       "Weights already downloaded; only package install needed",
    },
    "chexzero": {
        "display":     "CheXzero",
        "packages":    ["clip"],
        "local_files": ["pretrained/chexzero/pytorch_model.bin"],
        "notes":       "Weights must be manually downloaded from Google Drive",
    },
    "biovil": {
        "display":     "BioViL",
        "packages":    ["transformers", "health_multimodal"],
        "hf_repos":    ["microsoft/BiomedVLP-CXR-BERT-specialized"],
        "notes":       "health_multimodal = hi-ml-multimodal package (image encoder)",
    },
    "pmc_clip": {
        "display":     "PMC-CLIP",
        "packages":    ["open_clip"],
        "hf_repos":    ["ryanyip7777/pmc_vit_l_14"],
    },
    "conch": {
        "display":     "CONCH",
        "packages":    ["conch"],
        "hf_repos":    ["MahmoodLab/conch"],
        "gated":       True,
        "notes":       "Requires institutional email for HF access approval",
    },
    "plip": {
        "display":     "PLIP",
        "packages":    ["transformers"],
        "hf_repos":    ["vinid/plip"],
    },
    "openclip_vitb32": {
        "display":     "OpenCLIP ViT-B-32",
        "packages":    ["open_clip"],
        "openclip":    ("ViT-B-32", "laion2b_s34b_b79k"),
    },
    "openclip_vitl14": {
        "display":     "OpenCLIP ViT-L-14",
        "packages":    ["open_clip"],
        "openclip":    ("ViT-L-14", "laion2b_s32b_b82k"),
    },
    "siglip": {
        "display":     "SigLIP",
        "packages":    ["transformers"],
        "hf_repos":    ["google/siglip-base-patch16-224"],
    },
    "rad_dino": {
        "display":     "RAD-DINO",
        "packages":    ["transformers"],          # rad_dino pkg optional
        "hf_repos":    ["microsoft/rad-dino"],
        "notes":       "rad-dino pkg optional; transformers fallback used if missing",
    },
}

FIX_CMDS = {
    "biomedclip":      ["pip install open-clip-torch --upgrade"],
    "pubmedclip":      ["pip install transformers --upgrade"],
    "medclip":         ["pip install git+https://github.com/RyanWangZf/MedCLIP.git"],
    "chexzero":        [
        "pip install git+https://github.com/openai/CLIP.git",
        "# Download weights → pretrained/chexzero/pytorch_model.bin",
        "# URL: https://drive.google.com/drive/folders/1makFLiEMbSleYltaRxw81aBhEDMpVwno",
        "# File: best_64_0.0001_original_17000_0.8518.pt  →  rename to pytorch_model.bin",
    ],
    "biovil":          ["pip install hi-ml-multimodal"],
    "pmc_clip":        ["pip install open-clip-torch --upgrade"],
    "conch":           [
        "# 1. Request access (institutional email required):",
        "#    https://huggingface.co/MahmoodLab/conch",
        "pip install git+https://github.com/mahmoodlab/CONCH.git",
        "# 2. Set token:  export HUGGINGFACE_HUB_TOKEN=hf_xxxx",
    ],
    "plip":            ["pip install transformers --upgrade"],
    "openclip_vitb32": ["pip install open-clip-torch --upgrade"],
    "openclip_vitl14": ["pip install open-clip-torch --upgrade"],
    "siglip":          ["pip install transformers --upgrade"],
    "rad_dino":        ["pip install rad-dino   # optional; transformers fallback exists"],
}


# ── result collector ─────────────────────────────────────────────────────────

def check_model(key: str) -> dict:
    spec = CHECKS[key]
    issues = []      # list of (severity, message)  severity: "MISSING" | "ACTION" | "DOWNLOAD"

    # 1. Package checks
    pkg_missing = []
    for pkg in spec.get("packages", []):
        if not _pkg_ok(pkg):
            pkg_missing.append(pkg)
    if pkg_missing:
        issues.append(("MISSING", f"Package(s) not installed: {', '.join(pkg_missing)}"))

    # 2. Local file checks
    files_missing = []
    for rel in spec.get("local_files", []):
        if not _file_ok(rel):
            files_missing.append(rel)
    if files_missing:
        issues.append(("ACTION", f"Local file(s) missing: {', '.join(files_missing)}"))

    # 3. HF cache checks
    hf_not_cached = []
    for repo_id in spec.get("hf_repos", []):
        if not _is_hf_cached(repo_id):
            hf_not_cached.append(repo_id)
    if hf_not_cached:
        issues.append(("DOWNLOAD", f"Not cached: {', '.join(hf_not_cached)}"))

    # 4. OpenCLIP LAION cache check
    if "openclip" in spec:
        mid, pretrained = spec["openclip"]
        hf_repo = _OPENCLIP_LAION_HF.get((mid, pretrained))
        if hf_repo and not _is_hf_cached(hf_repo):
            issues.append(("DOWNLOAD", f"Not cached: {hf_repo}"))
        elif not hf_repo:
            # Unknown mapping — can't verify, assume auto-download
            issues.append(("DOWNLOAD", f"Weights will auto-download on first use"))

    # 5. Gated model / token check
    if spec.get("gated"):
        token = _hf_token()
        if not token:
            issues.append(("ACTION",
                "HF token not set — run: huggingface-cli login  OR  "
                "export HUGGINGFACE_HUB_TOKEN=hf_xxxx"))
        # Even with a token, the user may not have access yet
        # We can't verify access without a network call, so just warn
        issues.append(("ACTION",
            "Gated model — ensure you have requested & received access at "
            f"https://huggingface.co/{spec['hf_repos'][0]}"))

    # Derive overall status
    severities = [s for s, _ in issues]
    if "MISSING" in severities:
        status = "MISSING"
    elif "ACTION" in severities:
        status = "ACTION"
    elif "DOWNLOAD" in severities:
        status = "DOWNLOAD"
    else:
        status = "READY"

    return {
        "key":    key,
        "name":   spec["display"],
        "status": status,
        "issues": issues,
        "notes":  spec.get("notes", ""),
    }


# ── display ──────────────────────────────────────────────────────────────────

STATUS_ICON = {
    "READY":    "✓",
    "DOWNLOAD": "↓",
    "ACTION":   "!",
    "MISSING":  "✗",
}
STATUS_LABEL = {
    "READY":    "READY   ",
    "DOWNLOAD": "DOWNLOAD",
    "ACTION":   "ACTION  ",
    "MISSING":  "MISSING ",
}

def print_report(results: list[dict], show_fix: bool) -> None:
    W = 80
    print("\n" + "=" * W)
    print(f"{'MEDVLM-PROBE  —  SETUP VERIFICATION':^{W}}")
    print("=" * W)
    print(f"  Project root : {ROOT}")
    print(f"  HF cache     : {_hf_cache_root()}")
    token = _hf_token()
    token_display = ("SET (" + token[:8] + "...)") if token else "NOT SET"
    print(f"  HF token     : {token_display}")
    print("=" * W)

    # Package versions summary
    key_pkgs = {"open_clip": "open-clip-torch", "transformers": "transformers",
                "torch": "torch", "medclip": "medclip", "clip": "openai-clip",
                "health_multimodal": "hi-ml-multimodal", "conch": "conch",
                "rad_dino": "rad-dino"}
    installed = {name: _pkg_version(name) for name, _ in key_pkgs.items() if _pkg_ok(name)}
    not_installed = [pip_name for pkg, pip_name in key_pkgs.items() if not _pkg_ok(pkg)]
    print(f"\n  Packages installed : {', '.join(f'{k} {v}' for k, v in installed.items()) or 'none'}")
    if not_installed:
        print(f"  Packages missing   : {', '.join(not_installed)}")

    print("\n" + "-" * W)
    print(f"  {'Model':<22} {'Status':<10} Details")
    print("-" * W)

    counts = {"READY": 0, "DOWNLOAD": 0, "ACTION": 0, "MISSING": 0}
    need_fix = []

    for r in results:
        s = r["status"]
        counts[s] += 1
        icon = STATUS_ICON[s]
        label = STATUS_LABEL[s]
        print(f"  {icon} {r['name']:<22} {label}", end="")

        # Compact inline summary
        if s == "READY":
            print("  all checks passed")
        elif r["issues"]:
            # Show only the first issue inline; rest shown below
            print(f"  {r['issues'][0][1]}")
            for _, msg in r["issues"][1:]:
                print(f"    {'':>24}  {msg}")
        else:
            print()

        if r["notes"] and s != "READY":
            print(f"    {'':>24}  note: {r['notes']}")

        if s in ("MISSING", "ACTION"):
            need_fix.append(r)

    print("-" * W)
    print(f"\n  {counts['READY']} ready  |  "
          f"{counts['DOWNLOAD']} will auto-download  |  "
          f"{counts['ACTION']} need action  |  "
          f"{counts['MISSING']} package missing\n")

    if need_fix and show_fix:
        print("=" * W)
        print(f"{'FIX COMMANDS':^{W}}")
        print("=" * W)
        for r in need_fix:
            print(f"\n[{r['name']}]")
            for cmd in FIX_CMDS.get(r["key"], ["# see src/models/custom_loaders.py"]):
                print(f"  {cmd}")
        print()
    elif need_fix:
        print(f"  Run with --fix to see exact commands for the {len(need_fix)} model(s) needing action.\n")


# ── entrypoint ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Verify MedVLM-Probe model setup")
    parser.add_argument("--fix", action="store_true",
                        help="Print exact install/download commands for broken models")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Only check these model keys (default: all)")
    args = parser.parse_args()

    keys = args.models if args.models else list(CHECKS.keys())
    invalid = [k for k in keys if k not in CHECKS]
    if invalid:
        print(f"Unknown model keys: {invalid}")
        print(f"Valid keys: {list(CHECKS.keys())}")
        sys.exit(1)

    results = [check_model(k) for k in keys]
    print_report(results, show_fix=args.fix)


if __name__ == "__main__":
    main()
