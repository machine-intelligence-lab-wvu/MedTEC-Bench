"""Write croissant.json (Croissant 1.0 + Responsible-AI 1.0 metadata) for MedTEC-Bench.

The released benchmark is text and pointers: prompt JSONs plus per-modality anchor
manifests that name images by their ID in the original source datasets. No images are
redistributed. Each file is listed with its SHA-256; each anchor manifest gets a RecordSet.

Usage (from the repo root):
    python scripts/make_croissant.py [--out croissant.json]
Validate:
    pip install mlcroissant && mlcroissant validate --jsonld croissant.json
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

REPO = "https://github.com/machine-intelligence-lab-wvu/MedTEC-Bench"
RAW = "https://raw.githubusercontent.com/machine-intelligence-lab-wvu/MedTEC-Bench/main/"

# (record-set id, manifest path, description, source dataset)
ANCHORS = [
    ("cxr", "data/cxr_probes_n1000/manifest.json",
     "Chest X-ray anchors: 1,000 frontal single-label images, 5 findings.", "NIH ChestX-ray14"),
    ("brain_mri", "data/mri_probes_n1000/manifest.json",
     "Brain MRI anchors: 1,000 axial slices, 4 dementia stages.", "OASIS-1"),
    ("abdominal_ct", "data/ct_disease_probes/manifest.json",
     "Abdominal CT anchors: 900 scans (150 per finding), 6 findings.", "Merlin abdominal CT"),
    ("fundus", "data/fundus_probes_n1000/manifest.json",
     "Retinal fundus anchors: 1,004 images, 7 ocular disease classes.", "ODIR-5K"),
    ("dermoscopy", "data/derm_probes_n1000/manifest.json",
     "Dermoscopy anchors: 1,000 images, 5 lesion types.", "ISIC 2018 Task 3 / HAM10000"),
    ("histopathology", "data/histopath_probes_n1000/manifest.json",
     "Histopathology anchors: 999 patches, 9 tissue classes.", "NCT-CRC-HE-100K (CRC-VAL-HE-7K)"),
    ("openi_cxr", "data/openi_probes_full/manifest.json",
     "OpenI chest radiographs used for the unified evaluation (Section 5.3).", "OpenI Indiana University CXR"),
    ("skin_tone", "data/isic_skintone/manifest.json",
     "Dermoscopy subset with Fitzpatrick skin type, for the subgroup analysis.", "ISIC archive collection 413"),
    ("sex_age", "data/isic_sexage/manifest.json",
     "Dermoscopy subset with patient sex and approximate age, for the subgroup analysis.", "HAM10000"),
]
EXTRA_FILES = [
    ("roco_cxr_captions", "data/roco_cxr/captions.json",
     "Captions of the 354-pair chest-radiograph subset of the ROCO test split (retrieval pool)."),
    ("roco_cxr_filter", "data/roco_cxr/manifest.json", "Filter definition for the ROCO-CXR pool."),
]
FIELDS = [("image_id", "Image identifier in the source dataset."),
          ("disease_category", "Ground-truth label used to select positive and negative prompts."),
          ("source", "Source dataset of the anchor image.")]


def sha256(path):
    # hash the committed blob (what raw.githubusercontent.com serves), not a CRLF working copy
    try:
        data = subprocess.run(["git", "show", f"HEAD:{path}"], capture_output=True, check=True).stdout
    except subprocess.CalledProcessError:
        data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def file_object(fid, path, desc, fmt="application/json"):
    return {"@type": "cr:FileObject", "@id": fid, "name": fid, "description": desc,
            "contentUrl": RAW + path, "encodingFormat": fmt, "sha256": sha256(path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="croissant.json")
    args = ap.parse_args()

    dist = [{"@type": "cr:FileObject", "@id": "repo", "name": "repo",
             "description": "MedTEC-Bench GitHub repository.", "contentUrl": REPO + ".git",
             "encodingFormat": "git+https", "sha256": "main"},
            {"@type": "cr:FileSet", "@id": "prompt-files", "name": "prompt-files",
             "description": "Per-modality L1-L5 prompt sets, wording variants (clinical, ontology, "
                            "report style) and literature-grounded negation ladders.",
             "containedIn": {"@id": "repo"}, "encodingFormat": "application/json",
             "includes": "configs/prompts/**/*.json"}]
    record_sets = []
    for rid, path, desc, src in ANCHORS:
        fid = f"{rid}-manifest"
        n = len(json.load(open(path)))
        dist.append(file_object(fid, path, f"{desc} Source: {src}. {n} records."))
        record_sets.append({
            "@type": "cr:RecordSet", "@id": rid, "name": rid, "description": desc,
            "field": [{"@type": "cr:Field", "@id": f"{rid}/{k}", "name": k, "description": d,
                       "dataType": "sc:Text",
                       "source": {"fileObject": {"@id": fid}, "extract": {"jsonPath": f"$[*].{k}"}}}
                      for k, d in FIELDS]})
    for fid, path, desc in EXTRA_FILES:
        dist.append(file_object(fid, path, desc))

    meta = {
        "@context": {
            "@language": "en", "@vocab": "https://schema.org/",
            "citeAs": "cr:citeAs", "column": "cr:column", "conformsTo": "dct:conformsTo",
            "cr": "http://mlcommons.org/croissant/", "rai": "http://mlcommons.org/croissant/RAI/",
            "data": {"@id": "cr:data", "@type": "@json"}, "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
            "dct": "http://purl.org/dc/terms/", "examples": {"@id": "cr:examples", "@type": "@json"},
            "extract": "cr:extract", "field": "cr:field", "fileProperty": "cr:fileProperty",
            "fileObject": "cr:fileObject", "fileSet": "cr:fileSet", "format": "cr:format",
            "includes": "cr:includes", "isLiveDataset": "cr:isLiveDataset", "jsonPath": "cr:jsonPath",
            "key": "cr:key", "md5": "cr:md5", "parentField": "cr:parentField", "path": "cr:path",
            "recordSet": "cr:recordSet", "references": "cr:references", "regex": "cr:regex",
            "repeated": "cr:repeated", "replace": "cr:replace", "sc": "https://schema.org/",
            "samplingRate": "cr:samplingRate", "equivalentProperty": "cr:equivalentProperty",
            "separator": "cr:separator", "source": "cr:source", "subField": "cr:subField",
            "transform": "cr:transform"},
        "@type": "sc:Dataset",
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        "name": "MedTEC-Bench",
        "description": "A five-level diagnostic benchmark that probes whether the text encoder of "
                       "dual-encoder medical vision-language models separates semantically distinct "
                       "descriptions of a fixed image: domain, modality, anatomy, finding, and negation. "
                       "Covers chest X-ray, brain MRI, abdominal CT, retinal fundus, dermoscopy, and "
                       "histopathology. Released as prompt sets and anchor-image manifests; images are "
                       "obtained from the original datasets.",
        "url": REPO,
        "license": "https://opensource.org/licenses/MIT",
        "version": "1.0.0",
        "datePublished": "2026-11-06",
        "keywords": ["medical vision-language models", "benchmark", "CLIP", "negation",
                     "semantic specificity", "evaluation"],
        "creator": [{"@type": "sc:Person", "name": n} for n in
                    ["Ahsan Habib Akash", "Alina Devkota", "Donald A. Adjeroh", "Binod Bhattarai",
                     "Prashnna Kumar Gyawali"]],
        "citeAs": "@inproceedings{akash2026medtec, title={When Medical VLMs Stop Understanding: "
                  "MedTEC-Bench for Probing Semantic Specificity}, author={Akash, Ahsan Habib and "
                  "Devkota, Alina and Adjeroh, Donald A. and Bhattarai, Binod and Gyawali, Prashnna Kumar}, "
                  "booktitle={Advances in Neural Information Processing Systems (Evaluations and Datasets "
                  "Track)}, year={2026}}",
        "isLiveDataset": False,
        # Responsible-AI metadata
        "rai:dataCollection": "Anchor images are sampled (seed 42, stratified, single-label) from public "
                              "medical imaging datasets; prompts were written by the authors from clinical "
                              "label vocabularies, with negation phrasings drawn from NegEx, ConText and the "
                              "Panicek-Hricak certainty lexicon.",
        "rai:dataCollectionType": ["Secondary Data Analysis", "Manual Human Curation"],
        "rai:dataPreprocessingProtocol": "Single-label filtering and per-class stratified sampling to about "
                                         "1,000 anchors per modality; CT anchors use a windowed representative "
                                         "axial slice for 2D models and the full volume for MERLIN. Selection "
                                         "scripts: scripts/select_*_probes.py, scripts/build_ct_disease_probes.py.",
        "rai:dataAnnotationProtocol": "No new image annotation. Labels come from the source datasets; prompt "
                                      "positives and negatives are assigned from those labels.",
        "rai:dataUseCases": "Diagnostic evaluation of the text-side semantic discrimination of dual-encoder "
                            "medical VLMs. Not validated for image-side grounding, generative or "
                            "cross-attention models, or for any clinical decision.",
        "rai:dataLimitations": "English prompts only; template-based wording (robustness variants are "
                               "provided); one representative slice per CT scan for 2D models; source-dataset "
                               "label noise is inherited. Passing every level is necessary, not sufficient, "
                               "for clinical use.",
        "rai:dataBiases": "Demographic composition is inherited from the source datasets. Subgroup analysis "
                          "(paper Appendix L) finds a small skin-tone gradient (gap at most 0.044 DS at "
                          "L1-L3) and sex and age gaps at most 0.019; finding-level skin-tone evaluation "
                          "is not covered.",
        "rai:personalSensitiveInformation": "None released. Manifests contain only source-dataset image IDs "
                                            "and labels, plus the de-identified sex, age and skin-type "
                                            "fields already public in ISIC/HAM10000. No images or reports "
                                            "are redistributed.",
        "rai:dataSocialImpact": "Intended to expose clinically relevant failures of medical VLMs before "
                                "deployment. Risk: over-reliance on the benchmark as a certification; it "
                                "must not replace clinical validation or non-AI safeguards.",
        "rai:hasSyntheticData": False,
        "rai:dataReleaseMaintenancePlan": "Maintained by the authors at the GitHub repository; issues and "
                                          "corrections are tracked there and versioned releases are tagged.",
        "distribution": dist,
        "recordSet": record_sets,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {args.out}: {len(dist)} distribution entries, {len(record_sets)} record sets")


if __name__ == "__main__":
    main()
