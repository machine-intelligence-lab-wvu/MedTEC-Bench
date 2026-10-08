"""Generate disease-labeled abdomen-CT prompt variants for Exp A (prompt robustness)
and Exp C (negation realism), matching the structure of the existing CXR variants.
Emits 4 files under configs/prompts/variants/:
  abdomen_ct_disease_{report,ontology,alt_clinical}.json   (Exp A, 4 pos / 10 neg parity)
  abdomen_ct_disease_negation_ladder.json                  (Exp C)
"""
import json, copy
from pathlib import Path

BASE = json.load(open("configs/prompts/abdomen_ct_disease_prompts.json", encoding="utf-8"))
V = Path("configs/prompts/variants"); V.mkdir(parents=True, exist_ok=True)

# ---- per-finding clinical vocabulary ----
F = {
 "hepatic_steatosis": dict(np="hepatic steatosis", neg="hepatic steatosis",
     syn=["fatty liver", "hepatic fatty infiltration", "diffuse fatty change of the liver"],
     onto="hepatic steatosis", feat="diffuse low hepatic attenuation, liver less dense than spleen"),
 "renal_cyst": dict(np="a renal cyst", neg="renal cyst",
     syn=["a simple kidney cyst", "a cystic renal lesion", "a fluid-density lesion in the kidney"],
     onto="renal cyst", feat="a well-defined thin-walled low-attenuation renal lesion"),
 "gallstones": dict(np="gallstones", neg="gallstones",
     syn=["cholelithiasis", "calculi in the gallbladder", "biliary calculi"],
     onto="cholelithiasis", feat="calcified foci layering within the gallbladder lumen"),
 "ascites": dict(np="ascites", neg="ascites",
     syn=["free intraperitoneal fluid", "peritoneal free fluid", "free abdominal fluid"],
     onto="ascites", feat="fluid tracking through the peritoneal cavity around the organs"),
 "pleural_effusion": dict(np="a pleural effusion", neg="pleural effusion",
     syn=["pleural fluid", "fluid in the pleural space", "dependent basal pleural fluid"],
     onto="pleural effusion", feat="dependent fluid at the lung bases in the pleural space"),
}
DISTRACT = ["splenomegaly", "hepatomegaly", "a pancreatic mass", "bowel obstruction", "hydronephrosis"]
CLASSES = ["hepatic_steatosis", "renal_cyst", "gallstones", "ascites", "pleural_effusion", "no_finding"]
NORMAL = {"report": "Impression: no acute intra-abdominal abnormality.",
          "ontology": "abdominal CT: no acute abnormality",
          "alt_clinical": "an unremarkable abdominal CT with no acute abnormality"}


def pos(style, key):
    if key == "no_finding":
        return {"report": ["Findings: normal solid organs and bowel. Impression: no acute intra-abdominal abnormality.",
                           "Findings: unremarkable abdomen and pelvis. Impression: normal abdominal CT.",
                           "Impression: no acute cardiopulmonary or intra-abdominal abnormality.",
                           "Findings: no focal lesion, fluid, or obstruction. Impression: within normal limits."],
                "ontology": ["abdominal CT: no acute abnormality", "abdominal CT: normal", "abdominal CT: unremarkable",
                             "abdominal CT: within normal limits"],
                "alt_clinical": ["an unremarkable abdominal CT with no acute abnormality",
                                 "a normal abdominal CT with no significant findings",
                                 "an abdominal CT with clear organs and no fluid",
                                 "an abdominal CT without focal lesion or obstruction"]}[style]
    f = F[key]
    if style == "report":
        return [f"Findings: {f['feat']}. Impression: {f['np']}.",
                f"Findings: features compatible with {f['np']}. Impression: {f['np']}.",
                f"Impression: {f['np']}, with {f['feat']}.",
                f"Findings: {f['feat']} on the abdominal CT. Impression: consistent with {f['np']}."]
    if style == "ontology":
        return [f"abdominal CT: {f['onto']}", f"abdominal CT — {f['onto']}",
                f"{f['onto']} on CT abdomen", f"CT abdomen: {f['onto']}"]
    # alt_clinical
    s = f["syn"]
    return [f"abdominal CT showing {f['np']}", f"abdominal CT with {s[0]}",
            f"abdominal CT demonstrating {s[1]}", f"abdominal CT showing {s[2]}"]


def neg(style, key):
    # 10 negatives: the other 4 findings + normal + 5 distractors
    others = [k for k in F if k != key] if key != "no_finding" else list(F.keys())
    items = [F[k]["np"] for k in others]        # 4 (or 5 for no_finding)
    if key != "no_finding":
        pass
    dis = DISTRACT[:]                            # 5
    if style == "report":
        base = [f"Impression: {x}." for x in items] + ([] if key == "no_finding" else [NORMAL["report"]]) + \
               [f"Impression: {d}." for d in dis]
    elif style == "ontology":
        base = [f"abdominal CT: {x}" for x in items] + ([] if key == "no_finding" else [NORMAL["ontology"]]) + \
               [f"abdominal CT: {d}" for d in dis]
    else:
        base = [f"abdominal CT showing {x}" for x in items] + ([] if key == "no_finding" else [NORMAL["alt_clinical"]]) + \
               [f"abdominal CT showing {d}" for d in dis]
    return base[:10]


def l5neg(style, key):
    # negation phrasings of the finding itself (10), matching the style register
    if key == "no_finding":
        # polarity flip: negatives assert presence of disease
        items = [F[k]["np"] for k in F] + DISTRACT
        if style == "report":
            return [f"Impression: {x}." for x in items][:10]
        if style == "ontology":
            return [f"abdominal CT: {x}" for x in items][:10]
        return [f"abdominal CT showing {x}" for x in items][:10]
    x = F[key]["neg"]
    return [f"abdominal CT showing no {x}", f"no evidence of {x} on abdominal CT",
            f"abdominal CT negative for {x}", f"{x} absent on abdominal CT",
            f"abdominal CT without {x}", f"no {x} visible on abdominal CT",
            f"abdominal CT clear of {x}", f"{x} excluded on abdominal CT",
            f"no signs of {x} on abdominal CT", f"{x} not seen on abdominal CT"]


for style in ["report", "ontology", "alt_clinical"]:
    cfg = {"modality": "abdomen_ct_disease", "variant": style,
           "description": f"Exp A prompt-robustness variant ({style}) for disease-labeled abdomen CT (Merlin).",
           "disease_categories": BASE["disease_categories"], "levels": {}}
    for L in ["1", "2", "3"]:
        cfg["levels"][L] = BASE["levels"][L]          # shared levels unchanged
    cfg["levels"]["4"] = {"name": "Fine-grained", "description": f"L4 finding ({style} phrasing).",
        "disease_prompts": {c: {"positives": pos(style, c), "negatives": neg(style, c)} for c in CLASSES}}
    cfg["levels"]["5"] = {"name": "Negation", "description": f"L5 negation ({style} phrasing).",
        "disease_prompts": {c: {"positives": pos(style, c), "negatives": l5neg(style, c)} for c in CLASSES}}
    for L in ("4", "5"):
        for c in CLASSES:
            dp = cfg["levels"][L]["disease_prompts"][c]
            assert len(dp["positives"]) == 4 and len(dp["negatives"]) == 10, (style, L, c, len(dp["positives"]), len(dp["negatives"]))
    json.dump(cfg, open(V / f"abdomen_ct_disease_{style}.json", "w"), indent=2)
    print(f"wrote abdomen_ct_disease_{style}.json")

# ---- negation ladder (Exp C) ----
def rungs(key):
    f = F[key]; np_, neg_ = f["np"], f["neg"]
    return {
      "present": [f"abdominal CT showing {np_}", f"{np_} is present on abdominal CT", f"findings of {np_} on abdominal CT"],
      "probable": [f"findings suggestive of {np_}", f"abdominal CT probably showing {np_}", f"likely {np_} on abdominal CT"],
      "possible": [f"possible {np_} on abdominal CT", f"{neg_} cannot be excluded", f"abdominal CT with equivocal findings for {neg_}"],
      "unlikely": [f"{neg_} is unlikely on abdominal CT", f"low suspicion for {neg_}", f"abdominal CT with findings not typical for {neg_}"],
      "absent_plain": [f"abdominal CT showing no {neg_}", f"no {neg_} on abdominal CT", f"abdominal CT without {neg_}"],
      "absent_evidential": [f"no CT evidence of {neg_}", f"abdominal CT with no findings of {neg_}", f"{neg_} excluded on abdominal CT"],
      "historical": [f"prior {neg_}, now resolved on abdominal CT", f"history of {neg_}, resolved", f"previously noted {neg_}, no longer seen"],
    }

ladder = {"modality": "abdomen_ct_disease",
  "description": "Exp C negation-realism certainty ladder for disease-labeled abdomen CT (Merlin).",
  "ordinal": {"present": 2, "probable": 1, "possible": 0, "unlikely": -1, "absent_plain": -2, "absent_evidential": -2},
  "subtype_map": {"definite": ["absent_plain"], "hedged": ["possible", "unlikely"],
                  "ruled_out": ["absent_evidential"], "historical": ["historical"]},
  "absent_anchor_category": "no_finding",
  "findings": {k: {"rungs": rungs(k), "historical": rungs(k)["historical"]} for k in F}}
json.dump(ladder, open(V / "abdomen_ct_disease_negation_ladder.json", "w"), indent=2)
print("wrote abdomen_ct_disease_negation_ladder.json; findings:", list(F.keys()))
