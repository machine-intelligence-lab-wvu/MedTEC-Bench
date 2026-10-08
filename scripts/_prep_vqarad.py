"""
Prep VQA-RAD (closed-form yes/no subset) for Exp B6: clinical VQA as a downstream
check of the MedTEC taxonomy (reviewer jpwD Q2, which names "clinical VQA").

Why this dataset answers the question directly. For a dual-encoder VLM, a closed
yes/no question reduces to scoring two statements against the image:
    "there is a pleural effusion"   vs   "there is no pleural effusion"
That contrast IS the MedTEC L5 probe. So VQA-RAD is not an analogy to our benchmark,
it is our benchmark's contrast embedded in an externally authored clinical dataset.

We additionally tag each question with the MedTEC level it interrogates, which lets
us ask the sharper question: does downstream VQA accuracy degrade in the level order
the taxonomy predicts?
    L2  view / modality / plane        ("is this an axial image?")
    L3  anatomy presence               ("are the kidneys present in this image?")
    L4  finding present  (gold = yes)  ("is there airspace consolidation?")
    L5  finding absent   (gold = no)   (same question form, gold answer "no")

Tagging is rule-based and the full tagged list is dumped to a review CSV so every
item can be checked by hand before anything is reported.

No GPU. Deterministic. Usage:
    python scripts/_prep_vqarad.py
"""
import csv, json, re
from pathlib import Path

REPO = Path(__file__).parent.parent
OUT = REPO / "data" / "vqarad"
RES = REPO / "experiments" / "results" / "vqarad"

# --- view / modality / acquisition vocabulary -> L2 -------------------------
VIEW = re.compile(
    r"\b(axial|coronal|sagittal|transverse|plane|projection|view|"
    r"ct|mri|mr\b|x-?ray|radiograph|ultrasound|modality|contrast|"
    r"t1|t2|flair|weighted|noncontrast|non-contrast|supine|upright|pa\b|ap\b|lateral)\b",
    re.I)

# --- organ / structure vocabulary -> L3 when asked about presence/visibility -
ORGAN = re.compile(
    r"\b(kidney|kidneys|liver|spleen|stomach|pancreas|gallbladder|bladder|"
    r"lung|lungs|heart|brain|skull|rib|ribs|spine|vertebra|aorta|bowel|colon|"
    r"duodenum|esophagus|trachea|diaphragm|clavicle|femur|humerus|ventricle|"
    r"cerebellum|temporal lobe|frontal lobe|occipital|parietal|thyroid|uterus|"
    r"adrenal|appendix|prostate|airway|mediastinum|hilum)\b", re.I)
PRESENCE = re.compile(
    r"\b(present|visible|seen|shown|included|appear|contain|image of|"
    r"in (?:this|the) image|part of)\b", re.I)

# --- pathology / abnormality vocabulary -> L4/L5 -----------------------------
PATHO = re.compile(
    r"\b(abnormal\w*|lesion|mass|tumou?r|cancer|carcinoma|malignan\w+|benign|"
    r"metasta\w+|effusion\w*|consolidat\w+|pneumonia|pneumothorax|pneumoperitoneum|"
    r"edema|oedema|atelectasis|infiltrat\w+|opacit\w+|nodul\w+|cavit\w+|fracture\w*|"
    r"hemorrhag\w+|haemorrhag\w+|bleed\w+|blood|h[ae]matoma|infarct\w*|isch[ae]mi\w+|"
    r"aneurysm|stenos\w+|obstruct\w+|occlu\w+|embol\w+|herniat\w+|hydrocephalus|"
    r"hydronephrosis|enlarge\w+|cardiomegaly|dilat\w+|inflamm\w+|infect\w+|abscess|"
    r"cyst\w*|calcif\w+|emphysema|fibrosis|atroph\w+|necro\w+|sclero\w+|degenerat\w+|"
    r"ascites|hyperinflat\w+|stranding|elevation|involve\w+|failure|"
    r"spondylo\w+|scolios\w+|kyphos\w+|perforat\w+|rupture\w*|laceration|contusion|"
    r"disease|patholog\w+|evidence of|acute|chronic|shift\w*|deform\w+|displace\w+|"
    r"compress\w+|narrow\w+|swell\w+|thicken\w+|effac\w+|collapse\w*|"
    r"free air|air fluid|fluid collection|collection|findings?|"
    r"subcutaneous air|gas|foreign body|devi(?:ate|ation)\w*)\b", re.I)

# "is X normal / unremarkable / within normal limits" inverts polarity:
# gold "yes" asserts ABSENCE of pathology (an L5 judgement), gold "no" asserts presence.
NORMALITY = re.compile(
    r"\b(normal|unremarkable|within normal limits|healthy|intact|"
    r"no abnormalit\w+|nothing abnormal)\b", re.I)


def tag_level(q):
    """Return (bucket, rule). bucket 'finding' is split into L4/L5 by gold answer;
    'finding_inv' is the normality phrasing, where that split is inverted."""
    ql = q.lower()
    if NORMALITY.search(ql):
        return "finding_inv", "normality-phrasing"
    if PATHO.search(ql):
        return "finding", "patho-vocab"
    # view/modality only counts when no organ-presence question is being asked
    if VIEW.search(ql) and not (ORGAN.search(ql) and PRESENCE.search(ql)):
        return "L2", "view-vocab"
    if ORGAN.search(ql) and PRESENCE.search(ql):
        return "L3", "organ+presence"
    if ORGAN.search(ql):
        return "L3", "organ-only"
    return "other", "unmatched"


# --- statement construction --------------------------------------------------
_IS_THERE = re.compile(r"^(?:is|are)\s+there\s+(?:any\s+|an\s+|a\s+)?(.*?)\s*\?*$", re.I)
_IS_THE = re.compile(r"^(?:is|are)\s+(?:the\s+|this\s+|these\s+)?(.*?)\s+(present|visible|seen|shown|enlarged|abnormal|normal|dilated)\s*\?*$", re.I)
_DOES = re.compile(r"^does\s+(?:the\s+|this\s+)?(.*?)\s+(?:show|have|contain|demonstrate)\s+(?:any\s+|an\s+|a\s+)?(.*?)\s*\?*$", re.I)


def declarative(q):
    """Rule-based (affirmative, negative) declarative pair, or None if no rule fits."""
    q = q.strip()
    m = _IS_THERE.match(q)
    if m:
        obj = m.group(1).strip()
        return f"there is {obj}", f"there is no {obj}"
    m = _IS_THE.match(q)
    if m:
        subj, adj = m.group(1).strip(), m.group(2).strip()
        return f"the {subj} is {adj}", f"the {subj} is not {adj}"
    m = _DOES.match(q)
    if m:
        subj, obj = m.group(1).strip(), m.group(2).strip()
        return f"the {subj} shows {obj}", f"the {subj} shows no {obj}"
    return None


def main():
    from datasets import load_dataset
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "images").mkdir(exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("flaviagiammarino/vqa-rad")
    items, seen_img = [], {}
    for split in ["test", "train"]:
        for i, rec in enumerate(ds[split]):
            ans = rec["answer"].strip().lower()
            if ans not in ("yes", "no"):
                continue
            q = rec["question"].strip()
            # de-duplicate images by content hash of the PIL bytes
            img = rec["image"].convert("RGB")
            key = (img.size, img.tobytes()[:2048])
            if key not in seen_img:
                iid = f"vqarad_{len(seen_img):04d}"
                p = OUT / "images" / f"{iid}.png"
                img.save(p)
                seen_img[key] = (iid, str(p))
            iid, ipath = seen_img[key]

            lvl, rule = tag_level(q)
            if lvl == "finding":
                # gold yes -> model must assert the finding is PRESENT  (L4)
                # gold no  -> model must assert the finding is ABSENT   (L5)
                lvl = "L4" if ans == "yes" else "L5"
            elif lvl == "finding_inv":
                # "is it normal?" -> gold yes means pathology ABSENT (L5)
                lvl = "L5" if ans == "yes" else "L4"
            decl = declarative(q)
            items.append({
                "uid": f"{split}_{i}", "split": split,
                "image_id": iid, "image_path": ipath,
                "question": q, "answer": ans,
                "level": lvl, "tag_rule": rule,
                "stmt_pos": decl[0] if decl else None,
                "stmt_neg": decl[1] if decl else None,
            })

    p = RES / "vqarad_prepared.json"
    json.dump({"n_items": len(items), "n_images": len(seen_img), "items": items},
              open(p, "w"), indent=1)

    # review CSV so the level tagging can be checked by hand
    rc = RES / "vqarad_level_tagging_review.csv"
    with open(rc, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "level", "tag_rule", "answer", "question", "stmt_pos", "stmt_neg"])
        for x in sorted(items, key=lambda z: (z["level"], z["tag_rule"])):
            w.writerow([x["split"], x["level"], x["tag_rule"], x["answer"],
                        x["question"], x["stmt_pos"] or "", x["stmt_neg"] or ""])

    from collections import Counter
    print(f"yes/no items : {len(items)}  (unique images {len(seen_img)})")
    print(f"by split     : {dict(Counter(x['split'] for x in items))}")
    print(f"by level     : {dict(Counter(x['level'] for x in items))}")
    print(f"test by level: {dict(Counter(x['level'] for x in items if x['split']=='test'))}")
    print(f"declarative rule fired: {sum(1 for x in items if x['stmt_pos'])}/{len(items)}")
    print(f"gold yes-rate: {sum(1 for x in items if x['answer']=='yes')/len(items):.3f}")
    print(f"wrote {p}\nwrote {rc}  <-- review this by hand")


if __name__ == "__main__":
    main()
