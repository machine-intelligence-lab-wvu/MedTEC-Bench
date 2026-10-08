"""Generate LITERATURE-GROUNDED negation ladders (kept separate from the originals).
Rung phrasings use verbatim cue terms from:
  - Panicek & Hricak (AJR 2016) standardized radiologist certainty lexicon:
      "consistent with" (>90%), "suspicious for"/"probable" (~75%),
      "possible" (~50%), "less likely" (~25%), "unlikely" (<10%)
  - NegEx (Chapman 2001) / ConText (Harkema 2009) negation + temporal cue sets:
      plain negation ("no", "negative for", "without", "absent"),
      evidential/ruled-out ("no evidence of", "ruled out", "excluded"),
      historical/temporal ("history of", "prior", "previously", "resolved").
Structure (rung keys, ordinal, subtype_map, absent_anchor, findings) is copied
from the existing ladders; only the phrasings change.

Output: configs/prompts/variants/lit/<mod>_negation_ladder_lit.json
"""
import json, glob
from pathlib import Path

SRC = "configs/prompts/variants"
OUT = Path("configs/prompts/variants/lit"); OUT.mkdir(parents=True, exist_ok=True)

IMG = {"cxr": "chest x-ray", "brain_mri": "brain MRI", "dermoscopy": "dermoscopy",
       "fundus": "fundus", "histopath": "histopathology", "abdomen_ct_disease": "abdominal CT"}
DISP = {"mel": "melanoma", "bcc": "basal cell carcinoma", "akiec": "actinic keratosis",
        "tum": "malignancy", "amd": "age-related macular degeneration"}


def disp(key):
    return DISP.get(key, key.replace("_", " "))


def rungs(img, f):
    return {
        # present: certainty >90% (Panicek "consistent with")
        "present": [f"{img} consistent with {f}",
                    f"findings consistent with {f} on {img}",
                    f"{img} demonstrating {f}"],
        # probable: ~75% (Panicek "suspicious for"/"probable"/"probably")
        "probable": [f"{img} suspicious for {f}",
                     f"probable {f} on {img}",
                     f"findings probably representing {f} on {img}"],
        # possible: ~50% (Panicek "possible/possibly" + NegEx pseudo-negation "cannot be excluded")
        "possible": [f"possible {f} on {img}",
                     f"{img} possibly showing {f}",
                     f"{f} cannot be excluded on {img}"],
        # unlikely: ~25%/<10% (Panicek "less likely"/"unlikely")
        "unlikely": [f"{f} less likely on {img}",
                     f"{f} unlikely on {img}",
                     f"{img} findings unlikely to represent {f}"],
        # absent_plain: definite negation (NegEx plain cues)
        "absent_plain": [f"{img} negative for {f}",
                         f"no {f} on {img}",
                         f"{img} without {f}",
                         f"{f} absent on {img}"],
        # absent_evidential: ruled-out (NegEx evidential cues)
        "absent_evidential": [f"no evidence of {f} on {img}",
                              f"{f} ruled out on {img}",
                              f"{f} excluded on {img}",
                              f"{img} shows no evidence of {f}"],
        # historical/comparative (ConText temporal cues)
        "historical": [f"history of {f}, resolved on {img}",
                       f"prior {f}, no longer seen on {img}",
                       f"previously documented {f}, resolved"],
    }


for src in sorted(glob.glob(f"{SRC}/*_negation_ladder.json")):
    mod = Path(src).name.replace("_negation_ladder.json", "")
    old = json.load(open(src, encoding="utf-8"))
    img = IMG.get(mod, mod.replace("_", " "))
    cfg = {"modality": old.get("modality", mod),
           "description": f"LITERATURE-GROUNDED negation ladder for {mod}. Cue terms verbatim from "
                          f"Panicek & Hricak (AJR 2016) certainty lexicon and NegEx/ConText (Chapman 2001; Harkema 2009).",
           "provenance": "Panicek&Hricak2016; NegEx Chapman2001; ConText Harkema2009",
           "ordinal": old["ordinal"], "subtype_map": old["subtype_map"],
           "absent_anchor_category": old["absent_anchor_category"], "findings": {}}
    for fkey in old["findings"]:
        f = disp(fkey)
        r = rungs(img, f)
        cfg["findings"][fkey] = {"rungs": r, "historical": r["historical"]}
    json.dump(cfg, open(OUT / f"{mod}_negation_ladder_lit.json", "w"), indent=2)
    print(f"wrote lit/{mod}_negation_ladder_lit.json  findings={list(old['findings'].keys())}")

# sanity print one
c = json.load(open(OUT / "cxr_negation_ladder_lit.json"))
print("\nCXR pneumonia rungs (lit):")
for k, v in c["findings"]["pneumonia"]["rungs"].items():
    print(f"  {k:18s} {v}")
