"""Cold start for NEW ITEMS: the two-tower item tower over every item of the catalogue with a known category (UNK id +
category + parent, `embed_new_items`), not only the 69,784 vocabulary items. No training, no other model changes.
Reports how many dev targets become reachable and the recall@10 of this "two_tower+catalogue" variant: on the newly
reachable targets, on the vocabulary targets (does a flood of category vectors hurt them?) and overall.

    python scripts/cold_items.py [--split dev] [--k 10]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# sets OMP_NUM_THREADS and checks the macOS OpenMP fix before torch / faiss / lightgbm load
import recsys  # noqa: E402,F401
from recsys.config import CACHE, FIRST_ITEM, MODELS, RESULTS, SPLITS, UNK
from recsys.data import Dataset
from recsys.pipeline import load_model


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["rerank", "dev", "test"])
    ap.add_argument("--final", action="store_true", help="required for --split test")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--splits-dir", type=Path, default=SPLITS)
    ap.add_argument("--cache-dir", type=Path, default=CACHE)
    ap.add_argument("--models-dir", type=Path, default=MODELS)
    ap.add_argument("--out", type=Path, default=RESULTS)
    a = ap.parse_args(argv)
    if a.split == "test" and not a.final:
        print("refusing to touch the test split without --final", file=sys.stderr)
        return 3
    import torch

    t0 = time.time()
    ds = Dataset.load(a.splits_dir)
    tt = load_model("two_tower", a.models_dir)
    cats = pd.read_parquet(a.cache_dir / "categories.parquet")  # raw item → raw category, parent (every item with a categoryid)
    items = ds.items
    cat_map = items.loc[items["category_raw"] >= 0, ["category_raw", "category"]].drop_duplicates("category_raw").set_index("category_raw")["category"]
    parent_map = items.loc[items["parent_raw"] >= 0, ["parent_raw", "parent"]].drop_duplicates("parent_raw").set_index("parent_raw")["parent"]
    new = cats[~cats["item"].isin(items["item"])].reset_index(drop=True)
    new_cat = new["category"].map(cat_map).fillna(0).to_numpy(dtype=np.int64)  # 0 = a category no vocabulary item has (unknown)
    new_parent = new["parent"].map(parent_map).fillna(0).to_numpy(dtype=np.int64)
    v_new = tt.embed_new_items(new_cat, new_parent).astype(np.float32)
    v_all = np.vstack([tt.item_vectors().astype(np.float32), v_new])  # rows 0..n_items−1 = PAD, UNK, vocabulary; then the new items
    n_items, n_new = ds.n_items, len(new)
    distinct_vectors = int(len(np.unique(np.stack([new_cat, new_parent], 1), axis=0)))
    cases = ds.cases(a.split)
    ext_of_raw = pd.Series(n_items + np.arange(n_new), index=new["item"].to_numpy())
    target_ext = np.where(cases.target != UNK, cases.target, ext_of_raw.reindex(cases.raw_item).fillna(-1).to_numpy(dtype=np.int64))
    in_vocab = cases.target != UNK
    newly = (~in_vocab) & (target_ext >= 0)
    print(f"{a.split}: {len(cases):,} cases; vocabulary {n_items - FIRST_ITEM:,} items, +{n_new:,} catalogue items with a category "
          f"({distinct_vectors:,} distinct category/parent vectors); reachable {in_vocab.mean():.1%} → {(target_ext >= 0).mean():.1%}")
    V = torch.from_numpy(v_all)
    hit_ext = np.zeros(len(cases), dtype=bool)
    hit_vocab = np.zeros(len(cases), dtype=bool)
    rank_ext = np.zeros(len(cases), dtype=np.int64)
    for start in range(0, len(cases), a.batch):
        sl = slice(start, min(start + a.batch, len(cases)))
        u = torch.from_numpy(tt.user_vectors(cases.history[sl]).astype(np.float32))
        s = u @ V.T  # (batch, n_items + n_new)
        s[:, :FIRST_ITEM] = float("-inf")
        top_ext = s.topk(a.k, dim=1).indices.numpy()
        top_voc = s[:, :n_items].topk(a.k, dim=1).indices.numpy()
        tgt = target_ext[sl][:, None]
        hit_ext[sl] = (top_ext == tgt).any(1)
        pos = (top_ext == tgt).argmax(1) + 1
        rank_ext[sl] = np.where(hit_ext[sl], pos, 0)
        hit_vocab[sl] = (top_voc == cases.target[sl][:, None]).any(1) & in_vocab[sl]

    def share(mask):
        return round(float(mask.mean()), 4) if mask.size else None

    rep = {"split": a.split, "k": a.k, "n_cases": len(cases), "vocabulary_items": n_items - FIRST_ITEM, "catalogue_items_with_category": int(len(cats)),
           "new_items_added": n_new, "distinct_new_item_vectors": distinct_vectors,
           "share_reachable_vocabulary": share(in_vocab), "share_reachable_with_catalogue": share(target_ext >= 0),
           "newly_reachable_cases": int(newly.sum()), "share_newly_reachable": share(newly),
           "recall@10": {"two_tower_vocabulary_only_overall": share(hit_vocab), "two_tower+catalogue_overall": share(hit_ext),
                         "two_tower+catalogue_on_newly_reachable": share(hit_ext[newly]), "two_tower+catalogue_on_vocabulary_targets": share(hit_ext[in_vocab]),
                         "two_tower_vocabulary_only_on_vocabulary_targets": share(hit_vocab[in_vocab])},
           "mrr@10_newly_reachable": round(float(np.where(rank_ext[newly] > 0, 1.0 / np.maximum(rank_ext[newly], 1), 0.0).mean()), 4) if newly.any() else None,
           "seconds": round(time.time() - t0, 1)}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"cold_items-{a.split}.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep["recall@10"], indent=2), f"\nnewly reachable: {rep['newly_reachable_cases']:,} cases ({rep['share_newly_reachable']:.1%}); {rep['seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
