"""
Exp B6 (rebuttal, reviewer jpwD Q2): closed-form clinical VQA as a downstream task.

For a dual-encoder VLM a yes/no question is answered by scoring two statements
against the image and taking the larger cosine. The two statements differ only in
polarity, so the decision IS a MedTEC L5 contrast evaluated on an externally
authored clinical dataset (VQA-RAD).

Two prompt protocols (the second doubles as a wording-robustness check, cf. Q1):
  qa    : "question: {q} answer: yes"      vs "question: {q} answer: no"
  decl  : rule-built declarative pair, e.g.
          "there is a pleural effusion"    vs "there is no pleural effusion"
          (only for items where the rule fires; see _prep_vqarad.py)

Primary reported quantities need NO semantic annotation of the questions:
  acc            overall accuracy (chance = 0.50; majority baseline reported too)
  acc_yes/acc_no accuracy split by gold answer. A model that cannot represent
                 negation answers one polarity well and the other near zero.
  yes_rate       fraction of items answered "yes" (gold yes-rate is ~0.49)
  shuffle control: the same statement pair scored against K mismatched images.
                 agreement = how often the prediction is unchanged by swapping the
                 image. High agreement => the answer is a text prior, not grounding.

Output: experiments/results/vqarad/vqarad_results.json  (incremental per model)

Usage:
    python scripts/run_vqarad.py
    python scripts/run_vqarad.py --split test --models chexzero
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "src"))
from models.model_registry import load_model, list_models, MODEL_REGISTRY

SEED = 42


def _resolve(p):
    p = Path(p)
    return p if p.is_absolute() else REPO / p


def _encode_images(model, paths, bs=32):
    cache = {}
    uniq = list(dict.fromkeys(str(p) for p in paths))
    for i in tqdm(range(0, len(uniq), bs), desc="  imgs", leave=False, ncols=70):
        batch, valid = [], []
        for p in uniq[i:i + bs]:
            try:
                batch.append(Image.open(_resolve(p)).convert("RGB")); valid.append(p)
            except Exception as e:
                print(f"  warn open {p}: {e}")
        if not batch:
            continue
        emb = model.encode_image(batch).cpu().numpy()
        for p, e in zip(valid, emb):
            cache[p] = np.asarray(e, np.float32).squeeze()
    return cache


def _encode_texts(model, texts, bs=64):
    out = []
    for i in tqdm(range(0, len(texts), bs), desc="  text", leave=False, ncols=70):
        out.append(model.encode_text(texts[i:i + bs]).cpu().numpy())
    return np.concatenate(out, 0).astype(np.float32)


def _stats(recs, key="pred_yes"):
    if not recs:
        return {}
    gold = np.array([r["gold_yes"] for r in recs], dtype=float)
    pred = np.array([r[key] for r in recs], dtype=float)
    corr = (pred == gold).astype(float)
    out = {
        "n": len(recs),
        "acc": float(corr.mean()),
        "yes_rate": float(pred.mean()),
        "gold_yes_rate": float(gold.mean()),
        "majority_baseline": float(max(gold.mean(), 1 - gold.mean())),
    }
    for nm, msk in (("yes", gold == 1), ("no", gold == 0)):
        out[f"acc_{nm}"] = float(corr[msk].mean()) if msk.any() else None
        out[f"n_{nm}"] = int(msk.sum())
    out["balanced_acc"] = (None if out["acc_yes"] is None or out["acc_no"] is None
                           else 0.5 * (out["acc_yes"] + out["acc_no"]))
    return out


def run_model(model, items, k_shuffle):
    icache = _encode_images(model, [x["image_path"] for x in items])
    uniq = [p for p in dict.fromkeys(str(x["image_path"]) for x in items) if p in icache]
    pool = np.stack([icache[p] for p in uniq])
    pidx = {p: i for i, p in enumerate(uniq)}
    rng = np.random.default_rng(SEED)

    protocols = {
        "qa": lambda x: (f"question: {x['question']} answer: yes",
                         f"question: {x['question']} answer: no"),
        "decl": lambda x: ((x["stmt_pos"], x["stmt_neg"])
                           if x.get("stmt_pos") else None),
    }

    out = {}
    for proto, build in protocols.items():
        sel = [x for x in items if build(x) is not None and str(x["image_path"]) in icache]
        if not sel:
            continue
        pairs = [build(x) for x in sel]
        pos = _encode_texts(model, [p[0] for p in pairs])
        neg = _encode_texts(model, [p[1] for p in pairs])

        recs = []
        for k, x in enumerate(sel):
            p = str(x["image_path"])
            v = icache[p]
            d = float(v @ pos[k]) - float(v @ neg[k])
            own = pidx[p]
            cand = rng.choice(len(uniq), size=min(k_shuffle + 1, len(uniq)), replace=False)
            cand = [c for c in cand if c != own][:k_shuffle]
            dsh = pool[cand] @ pos[k] - pool[cand] @ neg[k]
            recs.append({
                "uid": x["uid"], "split": x["split"], "level": x["level"],
                "gold_yes": int(x["answer"] == "yes"),
                "pred_yes": int(d > 0), "margin": d,
                "pred_yes_shuffled": float((dsh > 0).mean()),
                "agree_shuffled": float(((dsh > 0) == (d > 0)).mean()),
            })

        blk = {"overall": _stats(recs), "by_level": {}, "by_split": {}}
        for lv in sorted({r["level"] for r in recs}):
            blk["by_level"][lv] = _stats([r for r in recs if r["level"] == lv])
        for sp in sorted({r["split"] for r in recs}):
            blk["by_split"][sp] = _stats([r for r in recs if r["split"] == sp])
        blk["shuffle_agreement"] = float(np.mean([r["agree_shuffled"] for r in recs]))
        blk["items"] = recs
        out[proto] = blk
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["all"])
    ap.add_argument("--prepared", default="experiments/results/vqarad/vqarad_prepared.json")
    ap.add_argument("--output", default="experiments/results/vqarad/vqarad_results.json")
    ap.add_argument("--split", default="all", choices=["all", "test", "train"])
    ap.add_argument("--k-shuffle", type=int, default=25)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    data = json.load(open(_resolve(args.prepared), encoding="utf-8"))
    items = data["items"]
    if args.split != "all":
        items = [x for x in items if x["split"] == args.split]

    keys = list_models() if "all" in args.models else args.models
    keys = [k for k in keys if k != "biovil"]
    out_path = _resolve(args.output); out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"items={len(items)}  split={args.split}  models={keys}\nout: {out_path}")

    for mk in keys:
        print(f"\n{'='*54}\n  {MODEL_REGISTRY[mk]['display_name']}\n{'='*54}")
        try:
            model = load_model(mk, device=args.device)
        except Exception as e:
            print(f"  load failed: {e}"); continue
        try:
            res = run_model(model, items, args.k_shuffle)
        except Exception as e:
            print(f"  run failed: {e}"); del model; torch.cuda.empty_cache(); continue
        res["display_name"] = MODEL_REGISTRY[mk]["display_name"]
        for proto in ("qa", "decl"):
            if proto not in res:
                continue
            o = res[proto]["overall"]
            print(f"  [{proto:4s}] n={o['n']:4d} acc={o['acc']:.3f} "
                  f"(yes={o['acc_yes']:.3f} no={o['acc_no']:.3f} bal={o['balanced_acc']:.3f}) "
                  f"yes_rate={o['yes_rate']:.3f} shuffle_agree={res[proto]['shuffle_agreement']:.3f}")

        merged = {}
        if out_path.exists():
            try: merged = json.load(open(out_path, encoding="utf-8"))
            except Exception: merged = {}
        merged[mk] = res
        json.dump(merged, open(out_path, "w"), indent=1)
        print(f"  saved ({len(merged)} models)")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
