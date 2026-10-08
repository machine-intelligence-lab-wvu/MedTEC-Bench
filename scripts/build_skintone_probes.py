"""Build a skin-tone (Fitzpatrick) stratified dermoscopy probe set from ISIC
collection 413 (MSKCC Skin Tone Labeling Dataset) for the demographic-fairness
analysis (reviewer eTVi Q1). Real Fitzpatrick I-VI labels; dermoscopic images.
All lesions are benign, so this supports L1-L3 (recognition-level) fairness.

Balanced sample: N per Fitzpatrick group, seed 42. Downloads full images from the
public ISIC S3 URLs and writes a manifest with the skin-tone label.
"""
import argparse, json, collections
from pathlib import Path
import requests

API = "https://api.isic-archive.com/api/v2/images/search/"
OUT = Path("data/isic_skintone")


def fetch_dermoscopic(collection="413"):
    rows = []
    url, params = API, {"collections": collection, "limit": 100}
    while url:
        j = requests.get(url, params=params, timeout=60).json()
        for im in j.get("results", []):
            md = im.get("metadata", {})
            if md.get("acquisition", {}).get("image_type") != "dermoscopic":
                continue
            fitz = md.get("clinical", {}).get("fitzpatrick_skin_type")
            furl = im.get("files", {}).get("full", {}).get("url")
            if fitz and furl:
                rows.append({"isic_id": im["isic_id"], "fitzpatrick": fitz, "url": furl,
                             "sex": md.get("clinical", {}).get("sex"),
                             "age_approx": md.get("clinical", {}).get("age_approx")})
        url, params = j.get("next"), None
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-group", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    import random
    rng = random.Random(args.seed)

    (OUT / "images").mkdir(parents=True, exist_ok=True)
    print("fetching collection 413 dermoscopic metadata ...", flush=True)
    rows = fetch_dermoscopic()
    by_fitz = collections.defaultdict(list)
    for r in rows:
        by_fitz[r["fitzpatrick"]].append(r)
    print("available per Fitzpatrick:", {k: len(v) for k, v in sorted(by_fitz.items())}, flush=True)

    manifest = []
    for fitz in sorted(by_fitz):
        pool = by_fitz[fitz]; rng.shuffle(pool)
        got = 0
        for r in pool:
            if got >= args.per_group:
                break
            p = OUT / "images" / f"{r['isic_id']}.jpg"
            if not p.exists():
                try:
                    resp = requests.get(r["url"], timeout=60)
                    if resp.status_code != 200:
                        continue
                    p.write_bytes(resp.content)
                except Exception:
                    continue
            manifest.append({"image_id": r["isic_id"],
                             "image_path": f"data/isic_skintone/images/{r['isic_id']}.jpg",
                             "fitzpatrick_skin_type": fitz,
                             "sex": r.get("sex"), "age_approx": r.get("age_approx"),
                             "disease_category": "skin_lesion", "source": "isic_collection_413"})
            got += 1
        print(f"  Fitzpatrick {fitz}: {got} images", flush=True)
    json.dump(manifest, open(OUT / "manifest.json", "w"), indent=1)
    print(f"\nDONE: {len(manifest)} images -> {OUT/'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
