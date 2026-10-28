"""Export what the API and the demo serve → serving/ (gitignored; uploaded to the Space):

  * the eight models: popularity, repeat, hybrid, markov, item_knn, two_tower, sasrec and the reranker, as trained
  * popularity_recent.joblib: the last training week's popularity, served when no history item is known
  * similar_sasrec.npy: the L2-normalised SASRec item matrix for /similar (exact cosine search)
  * items.parquet: iidx, raw item id, category, parent, popularity rank, category path (root › … › leaf, category_tree.csv)
  * shoppers.json: 2,000 anonymous dev-window shoppers, 500 per history bucket, one case per visitor: the raw item ids of the
    history the models read (last 50) with the window each event fell in, and the item they opened next. No visitor id, no
    timestamp.
  * manifest.json: date, git commit, file sizes and hashes, configurations, and the checks below

Checks on the way out: every shopper's served history equals the history the evaluation built for that case, and the
served reranked top-10 equals the evaluation path's top-10 whenever the history holds a vocabulary item. A history with
none gets the recent-popularity fallback instead of the reranker; `--fallback-check` measures what that changes on dev
(eval/results/fallback_dev.json).

    python scripts/export_serving.py [--shoppers 2000] [--out serving]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# sets OMP_NUM_THREADS and checks the macOS OpenMP fix before torch / faiss / lightgbm load
import recsys  # noqa: E402,F401
from recsys.baselines import Popularity
from recsys.config import EVENT_NAMES, HISTORY_BUCKETS, MODELS, RAW, RAW_FILES, RESULTS, SOURCES, SPLITS, UNK
from recsys.data import WINDOW_CODE, WINDOW_NAMES, Dataset, load_tree
from recsys.pipeline import MODEL_FILES, load_model, source_topk
from recsys.rerank import Reranker
from recsys.serving import FALLBACK_FILE, SIMILAR_FILE, Service
from recsys.torch_utils import left_pad

BUCKET_LABEL = {(1, 1): "1", (2, 4): "2–4", (5, 19): "5–19", (20, 10_000): "20+"}


def category_path(c: int, tree: dict[int, int]) -> str:
    """Category ids from the root to `c` ("unknown" for an item without a category)."""
    if c < 0:
        return "unknown"
    chain, seen = [], set()
    while c >= 0 and c not in seen and len(chain) < 20:
        chain.append(c)
        seen.add(c)
        c = int(tree.get(c, -1))
    return " › ".join(str(x) for x in reversed(chain))


def pick_shoppers(ds: Dataset, n: int, seed: int) -> list[dict]:
    """One dev case per visitor (the rule of `Dataset.cases`: an event with at least one earlier event of the same visitor),
    n / 4 visitors per history bucket, ordered by history length."""
    ev = ds.events
    uidx, win, iidx, raw, et = (ev[c].to_numpy() for c in ("uidx", "window", "iidx", "item", "etype"))
    starts, _ = ds._bounds()
    rows = np.flatnonzero(win == WINDOW_CODE["dev"])
    rows = rows[rows > starts[uidx[rows]]]
    rng = np.random.default_rng(seed)
    perm = rows[rng.permutation(len(rows))]
    _, first = np.unique(uidx[perm], return_index=True)
    per_user = perm[first]  # a random dev case of every visitor
    hist_len = per_user - starts[uidx[per_user]]
    chosen = []
    for lo, hi in HISTORY_BUCKETS:
        pool = per_user[(hist_len >= lo) & (hist_len <= hi)]
        chosen.append(rng.choice(pool, min(n // len(HISTORY_BUCKETS), len(pool)), replace=False))
    chosen = np.concatenate(chosen)
    chosen = chosen[np.lexsort((rng.random(len(chosen)), chosen - starts[uidx[chosen]]))]
    out = []
    for k, r in enumerate(chosen):
        s = starts[uidx[r]]
        lo_row = max(s, r - ds.rules.max_len)
        n_before = int(r - s)
        bucket = next(BUCKET_LABEL[b] for b in HISTORY_BUCKETS if b[0] <= n_before <= b[1])
        out.append({"id": k, "history": raw[lo_row:r].tolist(), "window": [WINDOW_NAMES[w] for w in win[lo_row:r]], "events_before": n_before, "bucket": bucket,
                    "next_item": int(raw[r]), "next_event": EVENT_NAMES[int(et[r])], "next_in_vocabulary": bool(iidx[r] != UNK),
                    "next_seen_before": bool(iidx[r] != UNK and (iidx[s:r] == iidx[r]).any()), "_rows": (int(lo_row), int(r))})
    return out


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:12]


def git_sha(root: Path = ROOT) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5, check=False).stdout.strip() or "dev"
    except (OSError, subprocess.SubprocessError):
        return "dev"


def export(ds: Dataset, models_dir: Path, tree: dict[int, int], out: Path, n_shoppers: int = 2000, seed: int = 11, check: int = 200, log=print) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    for name in SOURCES:
        shutil.copy2(models_dir / MODEL_FILES[name], out / MODEL_FILES[name])
    shutil.copy2(models_dir / "reranker.joblib", out / "reranker.joblib")
    Popularity(recent=True).fit(ds).save(out / FALLBACK_FILE)
    v = load_model("sasrec", models_dir).item_vectors().astype(np.float32)
    np.save(out / SIMILAR_FILE, v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12))
    items = ds.items[["iidx", "item", "category_raw", "parent_raw", "popularity_rank"]].rename(columns={"category_raw": "category", "parent_raw": "parent"})
    items = items.reset_index(drop=True)
    items["category_path"] = [category_path(int(c), tree) for c in items["category"]]
    items.to_parquet(out / "items.parquet", index=False)
    shoppers = pick_shoppers(ds, n_shoppers, seed)
    rows = [s.pop("_rows") for s in shoppers]
    (out / "shoppers.json").write_text(json.dumps(shoppers), encoding="utf-8")
    rr = Reranker.load(models_dir / "reranker.joblib")
    sas, tt = load_model("sasrec", models_dir), load_model("two_tower", models_dir)
    manifest = {"created": str(date.today()), "git_commit": git_sha(), "n_items": int(len(items)), "max_len": int(ds.rules.max_len), "shoppers": len(shoppers),
                "shopper_buckets": {b: sum(s["bucket"] == b for s in shoppers) for b in BUCKET_LABEL.values()},
                "reranker": {"candidates_per_source": rr.rules.candidates_per_source, "best_iteration": int(rr.booster.best_iteration), "features": len(rr.features)},
                "sasrec": sas.cfg.__dict__, "two_tower": tt.cfg.__dict__, "similar": "cosine over L2-normalised SASRec item embeddings, exact (FAISS IndexFlatIP)"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # checks: the served history and the served reranked list against the evaluation path
    svc = Service.load(out)
    iidx = ds.events["iidx"].to_numpy()
    hist_match = all(np.array_equal(svc.history_matrix(s["history"])[0][0], left_pad([iidx[a:b]], ds.rules.max_len)[0]) for s, (a, b) in zip(shoppers, rows))
    known = [i for i, s in enumerate(shoppers) if svc.history_matrix(s["history"])[1]["any_known"]]
    sample = known[:: max(1, len(known) // max(check, 1))][:check]
    H = np.vstack([left_pad([iidx[rows[i][0] : rows[i][1]]], ds.rules.max_len) for i in sample])
    batch_idx, _ = svc.reranker.rerank(H, source_topk(svc.models, H, svc.reranker.rules.candidates_per_source), 10, models=svc.models)
    same = 0
    for j, i in enumerate(sample):
        served = [r.item for r in svc.recommend(shoppers[i]["history"], 10, "reranked").items]
        same += served == [int(svc.raw[x]) for x in batch_idx[j] if x >= 2]
    hits = {}
    for s in shoppers:
        if s["next_in_vocabulary"]:
            hits.setdefault(s["bucket"], []).append(s["next_item"] in {r.item for r in svc.recommend(s["history"], 10, "reranked").items})
        else:
            hits.setdefault(s["bucket"], []).append(False)
    manifest["checks"] = {"served_history_equals_evaluation_history": bool(hist_match),
                          "served_reranked_equals_batched_evaluation_top10_known_histories": f"{same}/{len(sample)}",
                          "shoppers_on_the_fallback_no_known_history": len(shoppers) - len(known),
                          "reranked_recall@10_on_these_shoppers": {b: round(float(np.mean(v)), 3) for b, v in hits.items()}}
    # the shopper the demo opens on: the first (by id) whose reranked list finds a next item they had not opened, which the hybrid
    # misses, with a seen-before item in the list — chosen to show the difference; "Random shopper" shows typical cases
    manifest["demo_shopper"] = None
    for s in shoppers:
        if s["bucket"] not in ("2–4", "5–19") or not s["next_in_vocabulary"] or s["next_seen_before"]:
            continue
        lists, meta = svc.lists(s["history"], 10, ("hybrid", "reranked"))
        rer, hyb = [r.item for r in lists["reranked"]], [r.item for r in lists["hybrid"]]
        if not meta["fallback"] and s["next_item"] in rer and s["next_item"] not in hyb and any(r.seen_before for r in lists["reranked"]):
            manifest["demo_shopper"] = s["id"]
            break
    manifest["checks"]["reranked_recall@10_on_all_shoppers"] = round(float(np.mean([h for v in hits.values() for h in v])), 3)
    files = sorted(p for p in out.iterdir() if p.is_file() and p.name != "manifest.json")
    manifest["files"] = {p.name: {"bytes": p.stat().st_size, "sha256": _sha(p)} for p in files}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    total = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    if log:
        for p in files:
            log(f"  {p.name:28s} {p.stat().st_size / 1e6:7.1f} MB")
        log(f"total {total / 1e6:.1f} MB in {out} (target < 120 MB)")
        log(f"checks: {json.dumps(manifest['checks'])}")
    manifest["total_bytes"] = total
    return manifest


def fallback_check(ds: Dataset, svc: Service, out: Path, resamples: int = 500) -> dict:
    """Dev cases whose history holds no vocabulary item: recall@10 of the evaluated reranker against the served fallback."""
    from recsys.metrics import paired_user_bootstrap, ranks_in_topk

    dev = ds.cases("dev")
    mask = (dev.history >= 2).sum(1) == 0
    sub = dev.subset(mask)
    idx, _ = svc.reranker.rerank(sub.history, source_topk(svc.models, sub.history, svc.reranker.rules.candidates_per_source), 10, models=svc.models)
    rer = (ranks_in_topk(idx, sub.target) > 0).astype(float)
    fb = (ranks_in_topk(svc.fallback.topk(sub.history, 10)[0], sub.target) > 0).astype(float)
    d = paired_user_bootstrap(fb, rer, sub.user, resamples)
    res = {"split": "dev", "cases": int(mask.sum()), "share_of_dev_cases": round(float(mask.mean()), 4), "targets_in_vocabulary": round(float(sub.reachable.mean()), 4),
           "recall@10_reranked_as_evaluated": round(float(rer.mean()), 4), "recall@10_served_fallback": round(float(fb.mean()), 4),
           "fallback_minus_reranked": {k: round(v, 4) if isinstance(v, float) else v for k, v in d.items()},
           "effect_on_dev_recall@10": round(d["point"] * float(mask.mean()), 4)}
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fallback-check", action="store_true", help="also measure the fallback against the evaluated reranker on dev")
    ap.add_argument("--shoppers", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--splits-dir", type=Path, default=SPLITS)
    ap.add_argument("--models-dir", type=Path, default=MODELS)
    ap.add_argument("--tree", type=Path, default=RAW / RAW_FILES["tree"])
    ap.add_argument("--out", type=Path, default=ROOT / "serving")
    a = ap.parse_args(argv)
    ds = Dataset.load(a.splits_dir)
    m = export(ds, a.models_dir, load_tree(a.tree), a.out, a.shoppers, a.seed)
    if a.fallback_check:
        res = fallback_check(ds, Service.load(a.out), RESULTS / "fallback_dev.json")
        print(f"fallback on dev: {res['cases']:,} cases ({res['share_of_dev_cases']:.1%}) with no known history item; recall@10 reranked "
              f"{res['recall@10_reranked_as_evaluated']:.4f} vs fallback {res['recall@10_served_fallback']:.4f}, paired "
              f"{res['fallback_minus_reranked']['point']:+.4f} [{res['fallback_minus_reranked']['ci_low']:+.4f}, {res['fallback_minus_reranked']['ci_high']:+.4f}]")
    ok = m["checks"]["served_history_equals_evaluation_history"] and m["total_bytes"] < 120e6
    a_, b_ = (int(x) for x in m["checks"]["served_reranked_equals_batched_evaluation_top10_known_histories"].split("/"))
    return 0 if ok and a_ == b_ else 1


if __name__ == "__main__":
    raise SystemExit(main())
