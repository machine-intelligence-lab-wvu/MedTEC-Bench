"""
MedTEC-Bench: Text Embedding Collapse Benchmark
6-level probe taxonomy for diagnosing text embedding collapse in medical VLMs.

All modalities load their prompt templates from configs/prompts/<modality>_prompts.json.
Levels 1-3 and 6 use shared (image-agnostic) probes; levels 4 and 5 are image-specific
when a probe-set manifest is provided via build_image_probes / get_probes_by_level.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent.parent.parent / "configs" / "prompts"


@dataclass
class Probe:
    """A single probe: anchor image + positive text + negative texts."""
    probe_id: str
    level: int  # 1-6
    level_name: str
    modality: str  # "cxr", "dermoscopy", "brain_mri", "abdomen_ct", "fundus", "histopath"
    positive_texts: List[str]
    negative_texts: List[str]
    image_path: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


# =============================================================
# Level Definitions
# =============================================================
LEVEL_DEFINITIONS = {
    1: {
        "name": "Cross-domain",
        "description": "Medical image vs everyday objects",
        "difficulty": "Easy",
        "insight": "Is the text encoder fundamentally broken?",
    },
    2: {
        "name": "Cross-modality",
        "description": "Same region, different imaging modality",
        "difficulty": "Easy",
        "insight": "Can it distinguish imaging types?",
    },
    3: {
        "name": "Cross-anatomy",
        "description": "Same modality, different anatomical region",
        "difficulty": "Medium",
        "insight": "Does it understand body regions?",
    },
    4: {
        "name": "Fine-grained",
        "description": "Same modality+region, different finding",
        "difficulty": "Hard",
        "insight": "Can it identify pathology?",
    },
    5: {
        "name": "Negation",
        "description": "Presence vs absence of finding",
        "difficulty": "Hard",
        "insight": "Is it sensitive to negation?",
    },
    6: {
        "name": "Attribute binding",
        "description": "Modifier sensitivity (large vs small)",
        "difficulty": "Hard",
        "insight": "Can it bind attributes correctly?",
    },
}


# =============================================================
# Modality registry — maps modality key -> (config filename stem, probe-id prefix)
# =============================================================
#
# Config file lives at configs/prompts/<stem>_prompts.json.
# The probe-id prefix is used only when synthesizing dynamic probe ids for
# L4/L5 fallback entries (the shared L1-3/L6 probe ids are taken verbatim
# from the JSON's "probe_id" fields).
# =============================================================

_MODALITY_CONFIG = {
    "cxr":        {"config": "cxr",        "id_prefix": "cxr"},
    "dermoscopy": {"config": "dermoscopy", "id_prefix": "derm"},
    "brain_mri":  {"config": "brain_mri",  "id_prefix": "brain"},
    "abdomen_ct": {"config": "abdomen_ct", "id_prefix": "ct"},
    "fundus":     {"config": "fundus",     "id_prefix": "fundus"},
    "histopath":  {"config": "histopath",  "id_prefix": "histopath"},
}

SUPPORTED_MODALITIES = list(_MODALITY_CONFIG.keys())


# =============================================================
# Generic JSON-backed probe loader
# =============================================================

def _prompts_path(modality: str) -> Path:
    cfg = _MODALITY_CONFIG.get(modality)
    if cfg is None:
        raise ValueError(
            f"Unknown modality: {modality}. Supported: {SUPPORTED_MODALITIES}"
        )
    return _PROMPTS_DIR / f"{cfg['config']}_prompts.json"


def _load_prompts(modality: str, prompts_path: Optional[str] = None) -> dict:
    """Load the prompt config for a modality.

    prompts_path: optional override to load an alternative prompt file (e.g. a
    prompt-robustness or negation-realism variant) instead of the modality's
    default configs/prompts/<modality>_prompts.json. The modality argument is
    still used everywhere else (model↔modality filtering, manifest, id prefix);
    only the text source changes.
    """
    path = Path(prompts_path) if prompts_path else _prompts_path(modality)
    if not path.is_absolute():
        # Resolve variant paths relative to the repo root, matching runner convention.
        path = _PROMPTS_DIR.parent.parent / path
    if not path.exists():
        raise FileNotFoundError(
            f"{modality} prompts file not found: {path}\n"
            + ("(override via prompts_path)" if prompts_path else
               f"Expected at configs/prompts/{_MODALITY_CONFIG[modality]['config']}_prompts.json")
        )
    with open(path) as f:
        return json.load(f)


def _build_probes_from_json(
    modality: str, prompts_path: Optional[str] = None
) -> Dict[int, List[Probe]]:
    """Build shared (image-agnostic) probes for all levels of a modality."""
    data = _load_prompts(modality, prompts_path)
    levels_data = data["levels"]
    prefix = _MODALITY_CONFIG[modality]["id_prefix"]
    result: Dict[int, List[Probe]] = {}

    # Levels with explicit shared templates (typically L1-3 and L6)
    for lvl_str, lvl_info in levels_data.items():
        lvl = int(lvl_str)
        if "probes" not in lvl_info:
            continue  # disease-specific levels handled separately
        lvl_name = LEVEL_DEFINITIONS[lvl]["name"]
        probes = []
        for p in lvl_info["probes"]:
            probes.append(Probe(
                probe_id=p["probe_id"],
                level=lvl,
                level_name=lvl_name,
                modality=modality,
                positive_texts=p["positives"],
                negative_texts=p["negatives"],
            ))
        result[lvl] = probes

    # Disease-keyed levels (typically L4 and L5): fallback — one Probe per
    # disease category when no manifest is available.
    for lvl_str, lvl_info in levels_data.items():
        lvl = int(lvl_str)
        if "disease_prompts" not in lvl_info:
            continue
        lvl_name = LEVEL_DEFINITIONS[lvl]["name"]
        probes = []
        for i, (category, texts) in enumerate(lvl_info["disease_prompts"].items(), start=1):
            probes.append(Probe(
                probe_id=f"{prefix}_l{lvl}_{i:03d}",
                level=lvl,
                level_name=lvl_name,
                modality=modality,
                positive_texts=texts["positives"],
                negative_texts=texts["negatives"],
                metadata={"disease_category": category},
            ))
        result[lvl] = probes

    return result


def build_image_probes(
    manifest_path: str, level: int, modality: str,
    prompts_path: Optional[str] = None,
) -> List[Probe]:
    """Build per-image probes for level 4 or 5 from a probe manifest.

    Each entry in manifest.json becomes its own Probe, with positive/negative
    texts drawn from the disease-specific prompts for that image's category.

    prompts_path: optional override prompt file (variant experiments).
    """
    if level not in (4, 5):
        raise ValueError(
            f"build_image_probes only supports levels 4 and 5, got {level}"
        )
    if modality not in _MODALITY_CONFIG:
        raise ValueError(
            f"Unknown modality: {modality}. Supported: {SUPPORTED_MODALITIES}"
        )

    prompts_data = _load_prompts(modality, prompts_path)
    lvl_info = prompts_data["levels"].get(str(level), {})
    disease_prompts = lvl_info.get("disease_prompts")
    if not disease_prompts:
        raise ValueError(
            f"Modality '{modality}' has no L{level} disease_prompts in its config"
        )

    lvl_name = LEVEL_DEFINITIONS[level]["name"]
    prefix = _MODALITY_CONFIG[modality]["id_prefix"]

    with open(manifest_path) as f:
        manifest = json.load(f)

    probes = []
    for entry in manifest:
        category = entry["disease_category"]
        if category not in disease_prompts:
            continue  # unknown category — skip gracefully
        texts = disease_prompts[category]
        probes.append(Probe(
            probe_id=f"{prefix}_l{level}_{entry['image_id']}",
            level=level,
            level_name=lvl_name,
            modality=modality,
            positive_texts=texts["positives"],
            negative_texts=texts["negatives"],
            image_path=entry.get("image_path"),
            metadata={
                "disease_category": category,
                "disease_labels": entry.get("disease_labels", {}),
                "source": entry.get("source", ""),
            },
        ))
    return probes


# =============================================================
# Back-compat aliases (older callers)
# =============================================================

def build_cxr_image_probes(manifest_path: str, level: int) -> List[Probe]:
    """Legacy alias. Prefer build_image_probes(manifest_path, level, 'cxr')."""
    return build_image_probes(manifest_path, level, "cxr")


def build_derm_image_probes(manifest_path: str, level: int) -> List[Probe]:
    """Legacy alias. Prefer build_image_probes(manifest_path, level, 'dermoscopy')."""
    return build_image_probes(manifest_path, level, "dermoscopy")


# =============================================================
# Probe cache (per modality, populated lazily)
# =============================================================

_PROBE_CACHE: Dict[tuple, Dict[int, List[Probe]]] = {}


def _get_probes(
    modality: str, prompts_path: Optional[str] = None
) -> Dict[int, List[Probe]]:
    # Key on (modality, prompts_path) so alternative prompt variants don't
    # collide with the default modality probes in the cache.
    key = (modality, prompts_path)
    if key not in _PROBE_CACHE:
        _PROBE_CACHE[key] = _build_probes_from_json(modality, prompts_path)
    return _PROBE_CACHE[key]


# =============================================================
# Public API
# =============================================================

def has_image_probes(
    modality: str, level: int, prompts_path: Optional[str] = None
) -> bool:
    """Return True if (modality, level) supports per-image probes via a manifest."""
    if modality not in _MODALITY_CONFIG or level not in (4, 5):
        return False
    try:
        prompts_data = _load_prompts(modality, prompts_path)
    except FileNotFoundError:
        return False
    lvl_info = prompts_data["levels"].get(str(level), {})
    return bool(lvl_info.get("disease_prompts"))


def get_all_probes(modality: str = "cxr") -> List[Probe]:
    """Get all shared probes for a given modality (image-agnostic)."""
    if modality not in _MODALITY_CONFIG:
        raise ValueError(
            f"Unknown modality: {modality}. Supported: {SUPPORTED_MODALITIES}"
        )
    probes: List[Probe] = []
    for level_probes in _get_probes(modality).values():
        probes.extend(level_probes)
    return probes


def get_probes_by_level(
    level: int,
    modality: str = "cxr",
    manifest_path: Optional[str] = None,
    prompts_path: Optional[str] = None,
) -> List[Probe]:
    """Get probes for a specific level and modality.

    If a manifest_path is supplied and the requested level supports per-image
    probes (L4/L5 where the config has `disease_prompts`), returns one Probe
    per manifest entry with prompts tailored to that image's disease label.
    Otherwise returns the shared template probes.

    prompts_path: optional override prompt file (variant experiments); keeps
    `modality` for filtering/manifest but swaps the text source.
    """
    if modality not in _MODALITY_CONFIG:
        raise ValueError(
            f"Unknown modality: {modality}. Supported: {SUPPORTED_MODALITIES}"
        )
    if manifest_path is not None and has_image_probes(modality, level, prompts_path):
        return build_image_probes(manifest_path, level, modality, prompts_path)
    return _get_probes(modality, prompts_path).get(level, [])


def save_probes(probes: List[Probe], output_path: str):
    """Save probes to JSON for reproducibility."""
    data = []
    for p in probes:
        data.append({
            "probe_id": p.probe_id,
            "level": p.level,
            "level_name": p.level_name,
            "modality": p.modality,
            "positive_texts": p.positive_texts,
            "negative_texts": p.negative_texts,
            "image_path": p.image_path,
            "metadata": p.metadata,
        })
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
