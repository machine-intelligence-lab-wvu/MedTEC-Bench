"""Build a sex/age-stratified dermoscopy probe set (reviewer eTVi Q1 fairness).
Real labels only: disease from ISIC-2018 Task3 GroundTruth (HAM10000), sex+age from
ISIC-2019 metadata, joined on image_id. Single-label anchors for the 5 classes the
dermoscopy prompt config covers (mel/nv/bcc/akiec/bkl). Images extracted from the
ISIC-2018 Task3 Input.zip.
"""
import json, zipfile
from pathlib import Path
import pandas as pd

GT = "data/isic/ISIC2018_Task3_Training_GroundTruth.csv"
META = "data/ISIC2018_Task3/ISIC_2019_Training_Metadata.csv"
IMG_ZIP = "data/ISIC2018_Task3/ISIC2018_Task3_Training_Input.zip"
OUT = Path("data/isic_sexage")
CLASSMAP = {"MEL": "mel", "NV": "nv", "BCC": "bcc", "AKIEC": "akiec", "BKL": "bkl"}
PER_CLASS = 200
SEED = 42


def main():
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    gt = pd.read_csv(GT); md = pd.read_csv(META)
    gt["image"] = gt["image"].str.strip(); md["image"] = md["image"].str.strip()
    j = gt.merge(md[["image", "sex", "age_approx", "anatom_site_general"]], on="image", how="inner")
    cls = list(CLASSMAP)
    j["dx"] = j[cls].idxmax(axis=1); j["npos"] = j[cls].sum(axis=1)
    j = j[(j["npos"] == 1) & (j["sex"].isin(["male", "female"])) & (j["age_approx"].notna())]

    picks = []
    for c in cls:
        sub = j[j["dx"] == c].sample(n=min(PER_CLASS, (j["dx"] == c).sum()), random_state=SEED)
        picks.append(sub)
    sel = pd.concat(picks)
    ids = set(sel["image"])
    print("selected:", len(sel), "| per class:", sel["dx"].value_counts().to_dict())
    print("sex:", sel["sex"].value_counts().to_dict())

    # extract images from zip (try with/without folder prefix)
    zf = zipfile.ZipFile(IMG_ZIP)
    names = {Path(n).stem: n for n in zf.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))}
    got = 0
    for iid in ids:
        member = names.get(iid)
        dst = OUT / "images" / f"{iid}.jpg"
        if dst.exists():
            got += 1; continue
        if member is None:
            continue
        with zf.open(member) as src:
            dst.write_bytes(src.read())
        got += 1
    print(f"extracted {got}/{len(ids)} images")

    manifest = []
    for _, r in sel.iterrows():
        p = OUT / "images" / f"{r['image']}.jpg"
        if not p.exists():
            continue
        age = float(r["age_approx"])
        manifest.append({"image_id": r["image"],
                         "image_path": f"data/isic_sexage/images/{r['image']}.jpg",
                         "disease_category": CLASSMAP[r["dx"]],
                         "sex": r["sex"], "age_approx": age,
                         "age_group": ("<=50" if age <= 50 else ">50"),
                         "source": "HAM10000 (ISIC2018-GT + ISIC2019-meta)"})
    json.dump(manifest, open(OUT / "manifest.json", "w"), indent=1)
    print(f"DONE: {len(manifest)} anchors -> {OUT/'manifest.json'}")


if __name__ == "__main__":
    main()
