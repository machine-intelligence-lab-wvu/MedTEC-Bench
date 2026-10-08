# MedTEC-Bench

**When Medical VLMs Stop Understanding: MedTEC-Bench for Probing Semantic Specificity**
*NeurIPS 2026, Evaluations and Datasets Track (poster)*

Ahsan Habib Akash<sup>1</sup>, Alina Devkota<sup>1</sup>, Donald A. Adjeroh<sup>1</sup>, Binod Bhattarai<sup>2,3,4</sup>, Prashnna Kumar Gyawali<sup>1</sup>
<sup>1</sup>West Virginia University, USA · <sup>2</sup>University of Aberdeen, UK · <sup>3</sup>Fogsphere (Redev.AI Ltd), UK · <sup>4</sup>University College London, UK

[Project page](https://machine-intelligence-lab-wvu.github.io/MedTEC-Bench/) · [Paper](#) · [OpenReview](#)

MedTEC-Bench is a diagnostic benchmark for the text side of dual-encoder medical vision–language models. For a fixed anchor image, each probe pairs correct descriptions with distractors that differ along a single semantic axis, organised into five levels of increasing specificity:

| Level | Axis | Example positive | Example negative |
|---|---|---|---|
| L1 | Domain | a chest x-ray | a red apple |
| L2 | Modality | a chest radiograph | a chest CT scan |
| L3 | Anatomy | a chest x-ray | a knee x-ray |
| L4 | Finding | chest x-ray showing pneumonia | chest x-ray showing cardiomegaly |
| L5 | Negation | chest x-ray showing pneumonia | chest x-ray showing no pneumonia |

The primary metric is the **Discrimination Score**, the gap between the best-matching positive and the strongest negative:

```
DS(I, T+, T−) = max_i cos(v, t+_i) − max_j cos(v, t−_j)
```

DS ≤ 0 is recorded as a probe failure (collapse). All headline values carry bootstrap 95% confidence intervals over images (B = 10,000, seed 42).

## Main findings

Across 11 models and six imaging modalities (chest X-ray, brain MRI, abdomen CT, retinal fundus, dermoscopy, histopathology):

- **Chest-radiograph specialists fail at the domain level.** MedCLIP, CheXzero and BioViL do not reliably separate chest radiographs from everyday objects (BioViL L1 DS = −0.253).
- **Broad-biomedical and general-domain models fail at the clinical levels.** They pass L1–L3 but collapse on fine-grained findings (L4) and negation (L5).
- **Negation fails universally.** The largest L5 score in the panel is +0.008, against a 0.05 discrimination boundary.
- **Standard protocols disagree.** ROCO retrieval, zero-shot AUROC and MedTEC probing rank the same models differently, including on identical OpenI images.

Chest radiography, n = 1,000 stratified anchors:

| Family | Model | L1 | L2 | L3 | L4 | L5 |
|---|---|---:|---:|---:|---:|---:|
| General-domain | OpenCLIP ViT-B/32 | +0.231 | +0.039 | +0.067 | +0.002 | −0.003 |
|  | OpenCLIP ViT-L/14 | +0.228 | +0.044 | +0.081 | +0.006 | −0.007 |
|  | SigLIP | +0.131 | +0.011 | +0.059 | +0.001 | −0.003 |
| Broad biomedical | BiomedCLIP | +0.223 | +0.048 | +0.110 | −0.009 | −0.011 |
|  | PMC-CLIP | +0.114 | +0.070 | +0.127 | −0.018 | −0.024 |
|  | PubMedCLIP | +0.038 | +0.027 | +0.049 | +0.002 | −0.003 |
| CXR specialist | MedCLIP | −0.012 | +0.001 | −0.000 | −0.000 | −0.004 |
|  | CheXzero | −0.008 | −0.007 | −0.002 | −0.001 | −0.004 |
|  | BioViL | −0.253 | −0.300 | −0.001 | −0.279 | −0.008 |

## Repository layout

```
configs/prompts/             MedTEC prompt inventories, one JSON per modality
configs/prompts/variants/    Prompt-robustness styles (report, ontology, alt_clinical)
configs/prompts/variants/lit Literature-grounded negation subtypes (NegEx, ConText, Panicek & Hricak)
src/models/                  Unified loader for the 11-model panel
src/benchmarks/              MedTEC-Bench probe construction and evaluation
src/metrics/                 Discrimination score and collapse metrics
scripts/                     Probe selection, evaluation, analysis and table export
data/*/manifest.json         Anchor manifests (JSON only; images come from upstream sources)
experiments/results/         Summary results for every table in the paper and extended experiments
docs/                        Project page (GitHub Pages)
```

## Installation

```bash
conda create -n medtec python=3.10
conda activate medtec
pip install -r requirements.txt
pip install -e .
python scripts/verify_setup.py
```

Two checkpoints need manual steps (see `src/models/custom_loaders.py`):

- **CheXzero**: download weights from the upstream release to `pretrained/chexzero/pytorch_model.bin`.
- **CONCH**: request access to the gated Hugging Face repository.

On Windows with Intel MKL builds, set `KMP_DUPLICATE_LIB_OK=TRUE`.

## Datasets

Images are not redistributed. Obtain each dataset under its own licence and place it under `data/<name>/`; the released manifests then resolve. Anchor sets are single-label and stratified by class (seed 42), and can be regenerated with the `select_*_probes.py` scripts.

| Modality | Source | Anchors | Labels |
|---|---|---:|---|
| Chest X-ray | NIH ChestX-ray14 | 1,000 | 5 single-label findings |
| Brain MRI | OASIS-1 | 1,000 | 4 dementia stages |
| Abdomen CT | Merlin (Stanford AIMI) | 900 | 6 disease classes |
| Retinal fundus | ODIR-5K | 1,004 | 7 ocular disease classes |
| Dermoscopy | ISIC 2018 Task 3 | 1,000 | 5 lesion types |
| Histopathology | NCT-CRC-HE-100K | 999 | 9 tissue classes |
| Retrieval / unified evaluation | ROCO (test), OpenI | 354 / 3,666 | captions / reports |

The organ-labelled AMOS-2022 CT set (`data/ct_probes_n1000`) is kept for reference; the paper's CT results use the disease-labelled Merlin set (`data/ct_disease_probes`).

## Reproducing the paper

```bash
# MedTEC L1–L5 on one modality (swap --modality / --manifest for the others)
python scripts/run_medtec.py --models all --modality cxr \
    --manifest data/cxr_probes_n1000/manifest.json --levels 1 2 3 4 5 \
    --output experiments/results/medtec_cxr_n1000.json

# Bootstrap 95% CIs for every (model, level, modality) cell
python scripts/_compute_medtec_bootstrap_n1000.py

# Standard protocols
python scripts/run_retrieval.py --pool roco_cxr
python scripts/run_retrieval.py --pool openi
python scripts/run_nih_zeroshot.py
python scripts/run_openi_zeroshot.py

# Text-embedding geometry (appendix)
python scripts/run_geometry.py --corpus openi
```

The full panel (11 models × 6 modalities × 5 levels, plus retrieval, AUROC and bootstrap) takes roughly 10–14 GPU-hours on a single 48 GB GPU.

## Extended experiments

These analyses were added during review and are included in the camera-ready version.

**Disease-labelled abdomen CT and MERLIN.** Each Merlin anchor yields a windowed axial slice for the 2D models and a 3D MERLIN embedding. MERLIN runs in its own environment (`pip install merlin-vlm monai nibabel`).

```bash
python scripts/build_ct_disease_probes.py --sas-file <file with container SAS URL> \
    --selection data/ct_disease_probes/selection_oversampled.json \
    --per-class 150 --out-dir data/ct_disease_probes
python scripts/run_medtec.py --models all --modality abdomen_ct \
    --manifest data/ct_disease_probes/manifest.json \
    --prompts-file configs/prompts/abdomen_ct_disease_prompts.json \
    --output experiments/results/medtec_ct_disease_2d.json
python scripts/run_ct_disease_merlin.py
python scripts/_ct_disease_bootstrap_ci.py
```

**Prompt robustness.** L4 and L5 under four phrasing styles (template, report, ontology, alternative clinical).

```bash
python scripts/_gen_promptrobust_variants.py
python scripts/run_medtec.py --modality cxr --manifest data/cxr_probes_n1000/manifest.json \
    --levels 4 5 --prompts-file configs/prompts/variants/cxr_report.json \
    --output experiments/results/promptrobust/medtec_cxr_report.json
python scripts/_build_prompt_robustness_table.py --modality cxr --dir experiments/results/promptrobust
```

**Negation subtypes.** Definite, hedged, ruled-out and historical negation.

```bash
python scripts/_gen_negation_ladder_lit.py
python scripts/run_negation_realism.py --modality cxr \
    --ladder configs/prompts/variants/lit/cxr_negation_ladder_lit.json \
    --manifest data/cxr_probes_n1000/manifest.json \
    --output experiments/results/negrealism/cxr_negation_realism.json
python scripts/run_negation_realism_merlin.py
python scripts/_build_negation_realism_table.py \
    --input experiments/results/negrealism/cxr_negation_realism.json --modality cxr
```

**Downstream task (VQA-RAD).** Closed yes/no questions scored as an affirmation-versus-negation contrast.

```bash
python scripts/_prep_vqarad.py
python scripts/run_vqarad.py
python scripts/_build_vqarad_table.py
```

**Demographic subgroups (dermoscopy).** Fitzpatrick skin tone (ISIC collection 413) and sex/age (HAM10000 joined to ISIC-2019 metadata).

```bash
python scripts/build_skintone_probes.py
python scripts/build_isic2019_sexage_probes.py
python scripts/run_medtec.py --modality dermoscopy --levels 1 2 3 \
    --manifest data/isic_skintone/manifest.json \
    --output experiments/results/skintone/medtec_skintone_413.json
python scripts/_demographic_gaps.py --attribute fitzpatrick \
    --input experiments/results/skintone/medtec_skintone_413.json \
    --manifest data/isic_skintone/manifest.json
```

**Sample size.** DS and CI width as a function of the number of anchors.

```bash
python scripts/_n_stability.py --input experiments/results/full_per_probe/medtec_cxr_n1000.json
```

## Result files

`experiments/results/` holds the per-(model, level) summaries used by every table, with per-image arrays removed to keep the repository small. Analyses that resample images (bootstrap CIs, sample size, subgroups) need the full per-probe files, attached to the [GitHub release](https://github.com/machine-intelligence-lab-wvu/MedTEC-Bench/releases) as `full_per_probe.zip`; unzip it into `experiments/results/`.

## Citation

```bibtex
@inproceedings{akash2026medtec,
  title     = {When Medical VLMs Stop Understanding: MedTEC-Bench for Probing Semantic Specificity},
  author    = {Akash, Ahsan Habib and Devkota, Alina and Adjeroh, Donald A. and Bhattarai, Binod and Gyawali, Prashnna Kumar},
  booktitle = {Advances in Neural Information Processing Systems (Evaluations and Datasets Track)},
  year      = {2026}
}
```

## License

Code is released under the MIT License (see `LICENSE`). Datasets and model checkpoints remain subject to their own licences.
