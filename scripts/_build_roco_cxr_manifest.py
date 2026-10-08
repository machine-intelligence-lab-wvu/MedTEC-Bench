"""
_build_roco_cxr_manifest.py — Filter ROCO down to a clean frontal-CXR subset
for the CXR-specific R@1 axis (Option B).

The filter is conservative: a caption must (a) explicitly mention chest
radiograph / chest x-ray / posteroanterior / anteroposterior chest, AND
(b) NOT mention any other modality or any non-thoracic body part. Borderline
cases (e.g. paediatric vs adult, post-surgical line/tube findings) are kept
as long as the modality is unambiguously CXR.

Output:
    data/roco_cxr/captions.json
    data/roco_cxr/manifest.json   (provenance + filter stats)
"""
import json
import re
import shutil
from pathlib import Path

REPO = Path(__file__).parent.parent
SRC = REPO / "data" / "roco" / "captions.json"
SRC_DIR = REPO / "data" / "roco"
DST_DIR = REPO / "data" / "roco_cxr"
DST_DIR.mkdir(parents=True, exist_ok=True)

# Inclusion: must contain at least one of these.
INCL_PATTERNS = [
    r"\bchest x-?ray(s)?\b",
    r"\bchest radiograph(s|y)?\b",
    r"\bcxr\b",
    r"\bposteroanterior chest\b",
    r"\bpa chest\b",
    r"\banteroposterior chest\b",
    r"\bap chest\b",
    r"\bfrontal chest\b",
    r"\bthoracic radiograph(s|y)?\b",
    r"\bplain (film of the )?chest\b",
]
INCL = re.compile("|".join(INCL_PATTERNS), re.I)

# Exclusion: any of these and we drop it (other modality, or other body part,
# or comparison rather than depiction).
EXCL_PATTERNS = [
    # other imaging modalities
    r"\bct (scan|chest|of the chest|imaging|image)\b",
    r"\b(computed|computerized) tomograph",
    r"\bmri\b", r"\bmagnetic resonance\b",
    r"\bultrasound\b", r"\bsonograph",
    r"\bechocardio",
    r"\bpet[\- ](ct|scan|imaging)\b",
    r"\bangiogra(m|phy|phic)\b",
    r"\bfluoroscop",
    r"\bbronchoscop",
    r"\bendoscop", r"\bcolonoscop",
    r"\bnuclear (medicine|scan|imaging)\b",
    r"\bbone scan\b",
    # other body parts
    r"\bpanoram(ic|a)\b", r"\bdental\b", r"\btooth\b", r"\bteeth\b",
    r"\bmaxilla", r"\bmandib", r"\bskull\b",
    r"\b(lumbar|cervical|thoracic) spine\b", r"\b(spine|spinal) (radiograph|x-?ray)\b",
    r"\bpelv(i|is|ic)\b", r"\bhip (radiograph|x-?ray|joint)\b",
    r"\bknee\b", r"\bfemur\b", r"\btibia\b", r"\bfibul",
    r"\bankle\b", r"\bfoot (radiograph|x-?ray)\b", r"\btoe\b",
    r"\bshoulder (radiograph|x-?ray|joint)\b",
    r"\belbow\b", r"\bwrist\b", r"\bhand (radiograph|x-?ray)\b",
    r"\bbreast\b", r"\bmammogram",
    r"\bderm(at|oscop|al)", r"\bskin lesion\b", r"\bskin biopsy\b",
    r"\bhistopath", r"\bbiopsy specimen\b", r"\bcytology\b",
    r"\bfundus\b", r"\bretina", r"\boptic disc\b", r"\bocular\b",
    r"\bbrain\b", r"\bcerebr",
    r"\bkidney\b", r"\bliver\b", r"\bspleen\b", r"\bgallbladder\b",
    r"\bpancrea", r"\bbowel\b", r"\bcolon\b",
    r"\babdomen\b", r"\babdominal (radiograph|x-?ray|ct|ultrasound)\b",
    r"\bneck\b",
]
EXCL = re.compile("|".join(EXCL_PATTERNS), re.I)


def keep(caption: str) -> bool:
    if not caption:
        return False
    if not INCL.search(caption):
        return False
    if EXCL.search(caption):
        return False
    return True


def main():
    with open(SRC, encoding="utf-8") as f:
        roco = json.load(f)

    kept = []
    for entry in roco:
        cap = entry.get("caption") or ""
        # rewrite image path so it points back to the original ROCO image dir
        # (we don't copy images; we just reference them by ../roco/images/...)
        if keep(cap):
            kept.append({
                "id": entry["id"],
                "image": str(Path("..") / "roco" / entry["image"]).replace("\\", "/"),
                "caption": cap,
            })

    out_captions = DST_DIR / "captions.json"
    with open(out_captions, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)

    manifest = {
        "source": "roco",
        "filter": "strict CXR (chest radiograph keywords minus other-modality / other-bodypart / comparison patterns)",
        "n_total_input": len(roco),
        "n_kept": len(kept),
        "kept_ratio": len(kept) / len(roco),
        "include_pattern_count": len(INCL_PATTERNS),
        "exclude_pattern_count": len(EXCL_PATTERNS),
    }
    with open(DST_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"ROCO-CXR strict filter: {len(kept)} / {len(roco)} kept ({100 * len(kept) / len(roco):.1f}%)")
    print(f"  saved {out_captions}")
    print()
    print("Sampled 10 kept captions:")
    for e in kept[:10]:
        print(f"  - {e['caption'][:140]}")
    print()
    print("Sampled 5 caption-end entries (sanity check tail of distribution):")
    for e in kept[-5:]:
        print(f"  - {e['caption'][:140]}")


if __name__ == "__main__":
    main()
