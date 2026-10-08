"""
Generate Exp A prompt-robustness style variants (report / ontology / alt_clinical)
for the modalities not hand-authored (brain_mri, abdomen_ct, dermoscopy, histopath).

Each output config has L4 + L5 with the SAME classes and 4-positive / 10-negative
counts as the modality's baseline configs/prompts/<mod>_prompts.json, so only the
wording STYLE varies (the Exp A controlled variable). cxr + fundus were hand-authored
with richer prose and are NOT regenerated here.

Provenance: templated from a per-class clinical content spec, then verified. Run:
    python scripts/_gen_promptrobust_variants.py
"""
import json
from pathlib import Path

REPO = Path(__file__).parent.parent
OUT = REPO / "configs" / "prompts" / "variants"

# ------------------------------------------------------------------ content spec
# per class: name (with article, for sentences), term (article-free, for negation/
# ontology), syns[2], feature (imaging feature clause), alt (alt-clinical core).
SPECS = {
    "dermoscopy": {
        "carrier": "a dermoscopy image", "normal_class": None,
        "classes": {
            "mel": {"name": "melanoma", "term": "Melanoma",
                    "syns": ["malignant melanoma", "melanocytic malignancy"],
                    "feature": "an asymmetric pigmented lesion with irregular borders and color variegation",
                    "alt": "features of melanoma"},
            "nv": {"name": "a melanocytic nevus", "term": "Melanocytic nevus",
                   "syns": ["a benign melanocytic nevus", "a common mole"],
                   "feature": "a symmetric pigmented lesion with uniform color and a regular pigment network",
                   "alt": "features of a melanocytic nevus"},
            "bcc": {"name": "basal cell carcinoma", "term": "Basal cell carcinoma",
                    "syns": ["a basal cell tumor", "basal cell skin cancer"],
                    "feature": "a pearly lesion with arborizing telangiectasia",
                    "alt": "features of basal cell carcinoma"},
            "akiec": {"name": "actinic keratosis", "term": "Actinic keratosis",
                      "syns": ["intraepithelial carcinoma", "Bowen disease"],
                      "feature": "an erythematous scaly lesion with surface keratosis",
                      "alt": "features of actinic keratosis"},
            "bkl": {"name": "a benign keratosis", "term": "Benign keratosis",
                    "syns": ["a seborrheic keratosis", "a solar lentigo"],
                    "feature": "a well-demarcated keratotic lesion with milia-like cysts",
                    "alt": "features of a benign keratosis"},
        },
        "distractors": [
            {"term": "Dermatofibroma", "report": "a dermatofibroma", "alt": "a dermatofibroma"},
            {"term": "Squamous cell carcinoma", "report": "squamous cell carcinoma", "alt": "features of squamous cell carcinoma"},
            {"term": "Vascular lesion", "report": "a vascular lesion", "alt": "a vascular lesion"},
            {"term": "Solar lentigo", "report": "a solar lentigo", "alt": "a solar lentigo"},
            {"term": "Angioma", "report": "an angioma", "alt": "an angioma"},
            {"term": "Sebaceous hyperplasia", "report": "sebaceous hyperplasia", "alt": "sebaceous hyperplasia"},
        ],
    },
    "abdomen_ct": {
        "carrier": "an abdominal CT slice", "normal_class": None,
        "classes": {
            "liver": {"name": "the liver", "term": "Liver", "syns": ["hepatic parenchyma", "the hepatic lobe"],
                      "feature": "homogeneous soft-tissue attenuation of the liver in the right upper quadrant", "alt": "the liver"},
            "spleen": {"name": "the spleen", "term": "Spleen", "syns": ["splenic parenchyma", "the splenic tissue"],
                       "feature": "the splenic parenchyma in the left upper quadrant", "alt": "the spleen"},
            "kidney_left": {"name": "the left kidney", "term": "Left kidney", "syns": ["the left renal parenchyma", "the left renal unit"],
                            "feature": "the left kidney in the retroperitoneum", "alt": "the left kidney"},
            "kidney_right": {"name": "the right kidney", "term": "Right kidney", "syns": ["the right renal parenchyma", "the right renal unit"],
                             "feature": "the right kidney in the retroperitoneum", "alt": "the right kidney"},
            "pancreas": {"name": "the pancreas", "term": "Pancreas", "syns": ["the pancreatic parenchyma", "the pancreatic gland"],
                         "feature": "the pancreatic parenchyma in the retroperitoneum", "alt": "the pancreas"},
            "gallbladder": {"name": "the gallbladder", "term": "Gallbladder", "syns": ["the gallbladder fossa", "the biliary reservoir"],
                            "feature": "the fluid-filled gallbladder beneath the liver", "alt": "the gallbladder"},
        },
        "distractors": [
            {"term": "Stomach", "report": "the stomach", "alt": "the stomach"},
            {"term": "Aorta", "report": "the aorta", "alt": "the aorta"},
            {"term": "Adrenal gland", "report": "an adrenal gland", "alt": "an adrenal gland"},
            {"term": "Colon", "report": "the colon", "alt": "the colon"},
            {"term": "Small bowel", "report": "a small bowel loop", "alt": "a small bowel loop"},
        ],
    },
    "brain_mri": {
        "carrier": "a brain MRI", "normal_class": "non_demented",
        "classes": {
            "non_demented": {"name": "no dementia", "term": "Cognitively normal brain",
                             "syns": ["no cortical atrophy", "normal brain volume"],
                             "feature": "preserved cortical volume without disproportionate atrophy", "alt": "no evidence of dementia"},
            "very_mild_dementia": {"name": "very mild dementia", "term": "Very mild dementia",
                                   "syns": ["very mild cognitive impairment", "early neurodegeneration"],
                                   "feature": "minimal hippocampal volume loss", "alt": "features of very mild dementia"},
            "mild_dementia": {"name": "mild dementia", "term": "Mild dementia",
                              "syns": ["mild Alzheimer-type dementia", "mild cognitive impairment"],
                              "feature": "hippocampal and medial temporal atrophy", "alt": "features of mild dementia"},
            "moderate_dementia": {"name": "moderate dementia", "term": "Moderate dementia",
                                  "syns": ["moderate Alzheimer-type dementia", "moderate neurodegeneration"],
                                  "feature": "widespread cortical atrophy with ventricular enlargement", "alt": "features of moderate dementia"},
        },
        "distractors": [
            {"term": "Ischemic infarct", "report": "an ischemic infarct", "alt": "an ischemic infarct"},
            {"term": "Brain tumor", "report": "a brain tumor", "alt": "a brain tumor"},
            {"term": "Multiple sclerosis", "report": "multiple sclerosis plaques", "alt": "features of multiple sclerosis"},
            {"term": "Hydrocephalus", "report": "hydrocephalus", "alt": "hydrocephalus"},
            {"term": "Small-vessel white matter disease", "report": "small-vessel white matter disease", "alt": "small-vessel ischemic change"},
            {"term": "Intracranial hemorrhage", "report": "an intracranial hemorrhage", "alt": "an intracranial hemorrhage"},
            {"term": "Encephalomalacia", "report": "encephalomalacia", "alt": "encephalomalacia"},
        ],
    },
    "histopath": {
        "carrier": "a histopathology image", "normal_class": "norm",
        "classes": {
            "tum": {"name": "colorectal adenocarcinoma", "term": "Adenocarcinoma epithelium",
                    "syns": ["tumor epithelium", "malignant epithelium"],
                    "feature": "crowded atypical glands with a high nuclear-to-cytoplasmic ratio", "alt": "features of tumor epithelium"},
            "norm": {"name": "normal colonic mucosa", "term": "Normal colon mucosa",
                     "syns": ["benign colonic mucosa", "normal glandular mucosa"],
                     "feature": "regular crypt architecture with goblet cells", "alt": "normal colonic mucosa"},
            "str": {"name": "cancer-associated stroma", "term": "Cancer-associated stroma",
                    "syns": ["desmoplastic stroma", "tumor stroma"],
                    "feature": "reactive fibroblastic stroma", "alt": "cancer-associated stroma"},
            "lym": {"name": "lymphocytes", "term": "Lymphocytes", "syns": ["a lymphocytic infiltrate", "lymphoid aggregates"],
                    "feature": "dense small round lymphocytes", "alt": "a lymphocytic infiltrate"},
            "mus": {"name": "smooth muscle", "term": "Smooth muscle", "syns": ["muscularis", "smooth muscle bundles"],
                    "feature": "eosinophilic smooth muscle fascicles", "alt": "smooth muscle"},
            "adi": {"name": "adipose tissue", "term": "Adipose tissue", "syns": ["fat tissue", "adipocytes"],
                    "feature": "clear adipocytes with peripheral nuclei", "alt": "adipose tissue"},
            "muc": {"name": "mucus", "term": "Mucus", "syns": ["mucin", "mucinous material"],
                    "feature": "extracellular mucin pools", "alt": "mucus"},
            "deb": {"name": "debris", "term": "Debris", "syns": ["necrotic debris", "cellular debris"],
                    "feature": "necrotic cellular debris", "alt": "debris"},
            "back": {"name": "background", "term": "Background", "syns": ["empty background", "non-tissue background"],
                     "feature": "empty slide background without tissue", "alt": "background"},
        },
        "distractors": [
            {"term": "Blood vessels", "report": "blood vessels", "alt": "blood vessels"},
            {"term": "Nerve tissue", "report": "a nerve bundle", "alt": "a nerve bundle"},
        ],
    },
}

# ------------------------------------------------------------------ style builders
def _lower(t):
    return t if (t[:1].islower() or " " in t and t.split()[0].islower()) else t[0].lower() + t[1:]


def _ont_pos(c):
    return [c["term"], c["syns"][0], c["syns"][1], f"{c['term']}, present"]


def _rep_pos(c):
    n, f = c["name"], c["feature"]
    return [f"Findings: {f}. Impression: {n}.",
            f"Impression: {n}.",
            f"Findings consistent with {n}. Impression: {n}.",
            f"Impression: {n}, showing {f}."]


def _alt_pos(carrier, c):
    n, a = c["name"], c["alt"]
    return [f"{carrier} with {a}", f"{carrier} demonstrating {n}",
            f"{carrier} compatible with {n}", f"{carrier} showing {a}"]


def _neg_pool(style, spec, exclude_key):
    """10 negatives naming OTHER classes + distractors, in the given style."""
    items = []
    for k, c in spec["classes"].items():
        if k == exclude_key:
            continue
        if style == "ontology":
            items.append(c["term"])
        elif style == "report":
            items.append(f"Impression: {c['name']}.")
        else:
            items.append(f"{spec['carrier']} demonstrating {c['name']}")
    for d in spec["distractors"]:
        if style == "ontology":
            items.append(d["term"])
        elif style == "report":
            items.append(f"Impression: {d['report']}.")
        else:
            items.append(f"{spec['carrier']} with {d['alt']}")
    return items[:10]


def _l5_neg(style, carrier, c, flip):
    t, tl, n = c["term"], _lower(c["term"]), c["name"]
    if flip:  # normal class: negatives affirm PRESENCE of disease
        if style == "ontology":
            return ["Abnormal", "Disease present", "Pathological findings present", "Positive for disease",
                    "Abnormal findings", "Active disease", "Significant abnormality", "Underlying pathology",
                    "Not within normal limits", "Disease identified"]
        if style == "report":
            return ["Impression: positive for disease.", "Impression: abnormal.",
                    "Impression: pathologic findings present.", "Impression: active disease.",
                    "Impression: a significant abnormality is identified.", "Impression: abnormal findings.",
                    "Impression: an imaging abnormality is demonstrated.", "Impression: not within normal limits.",
                    "Impression: underlying pathology is present.", "Impression: positive for an abnormality."]
        return [f"{carrier} demonstrating disease", f"{carrier} read as abnormal",
                f"{carrier} with pathology present", f"{carrier} positive for disease",
                f"{carrier} with an abnormality present", f"{carrier} showing active disease",
                f"{carrier} with a significant abnormality", f"{carrier} outside normal limits",
                f"{carrier} with underlying pathology", f"{carrier} demonstrating an abnormality"]
    if style == "ontology":
        return [f"No {tl}", f"Absence of {tl}", f"{t} absent", f"No evidence of {tl}", f"{t} excluded",
                f"{t} not present", f"Negative for {tl}", f"Without {tl}", f"No {tl} identified", f"{t} ruled out"]
    if style == "report":
        return [f"Findings: no {tl}. Impression: no {tl}.", f"Impression: negative for {tl}.",
                f"Impression: no evidence of {tl}.", f"Impression: {tl} is excluded.",
                f"Findings: no {tl}. Impression: {tl} not identified.", f"Impression: no findings of {tl}.",
                f"Impression: without {tl}.", f"Impression: {tl} is ruled out.",
                f"Impression: no {tl} seen.", f"Impression: {tl} not present."]
    return [f"{carrier} without {tl}", f"{carrier} negative for {tl}", f"{carrier} showing no {tl}",
            f"{carrier} with no evidence of {tl}", f"{carrier} with {tl} excluded", f"{carrier} free of {tl}",
            f"{carrier} in which {tl} is not seen", f"{carrier} with {tl} ruled out",
            f"{carrier} without {tl} identified", f"{carrier} with no {tl}"]


def _l5_pos(style, carrier, c, flip):
    if flip:  # normal class positives affirm ABSENCE
        if style == "ontology":
            return [c["term"], "No abnormality", "Normal study", "No disease"]
        if style == "report":
            return ["Impression: normal study.", "Impression: no acute abnormality.",
                    "Impression: negative for disease.", "Impression: no pathology identified."]
        return [f"{carrier} read as normal", f"{carrier} free of disease",
                f"{carrier} with no abnormality", f"{carrier} within normal limits"]
    if style == "ontology":
        return [c["term"], c["syns"][0], f"{c['term']}, present", c["syns"][1]]
    if style == "report":
        n, f = c["name"], c["feature"]
        return [f"Findings: {f}. Impression: {n}.", f"Impression: {n} is present.",
                f"Impression: {n}.", f"Findings consistent with {n}."]
    return [f"{carrier} with {c['alt']}", f"{carrier} demonstrating {c['name']}",
            f"{carrier} positive for {c['name']}", f"{carrier} showing {c['alt']}"]


def build_modality(mod, spec):
    carrier, normal = spec["carrier"], spec["normal_class"]
    for style in ("report", "ontology", "alt_clinical"):
        l4, l5 = {}, {}
        for key, c in spec["classes"].items():
            flip = (key == normal)
            if style == "ontology":
                l4pos = _ont_pos(c)
            elif style == "report":
                l4pos = _rep_pos(c)
            else:
                l4pos = _alt_pos(carrier, c)
            l4[key] = {"positives": l4pos, "negatives": _neg_pool(style, spec, key)}
            l5[key] = {"positives": _l5_pos(style, carrier, c, flip),
                       "negatives": _l5_neg(style, carrier, c, flip)}
        cfg = {
            "modality": mod, "variant": style,
            "description": f"Exp A prompt-robustness variant ({style}) for {mod}, "
                           f"templated from clinical content spec (scripts/_gen_promptrobust_variants.py). "
                           f"Same L4/L5 classes and 4pos/10neg counts as configs/prompts/{mod}_prompts.json. L4+L5 only.",
            "levels": {"4": {"name": "Fine-grained", "disease_prompts": l4},
                       "5": {"name": "Negation", "disease_prompts": l5}},
        }
        p = OUT / f"{mod}_{style}.json"
        json.dump(cfg, open(p, "w"), indent=2)
        print(f"  wrote {p.name}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for mod, spec in SPECS.items():
        print(f"[{mod}]")
        build_modality(mod, spec)
    print("done")
