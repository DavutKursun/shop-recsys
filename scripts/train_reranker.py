"""Train the LightGBM LambdaRank reranker on the rerank window [T0, T1) over the union of the sources' candidates.

    python scripts/train_reranker.py [--sources popularity,repeat,markov,item_knn,two_tower,sasrec] [--cases 25000]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# sets OMP_NUM_THREADS and checks the macOS OpenMP fix before torch / faiss / lightgbm load
import recsys  # noqa: E402,F401
from recsys.config import MODELS, RERANK, RESULTS, SOURCES, SPLITS
from recsys.data import Dataset
from recsys.pipeline import available_models, load_model
from recsys.rerank import ItemStats, Reranker


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="available", help="comma list, or 'available' = every trained model")
    ap.add_argument("--cases", type=int, default=RERANK.train_cases)
    ap.add_argument("--per-source", type=int, default=RERANK.candidates_per_source)
    ap.add_argument("--splits-dir", type=Path, default=SPLITS)
    ap.add_argument("--models-dir", type=Path, default=MODELS)
    ap.add_argument("--out", type=Path, default=RESULTS)
    a = ap.parse_args(argv)
    ds = Dataset.load(a.splits_dir)
    names = available_models(a.models_dir) if a.sources == "available" else a.sources.split(",")
    names = [n for n in SOURCES if n in names]
    models = {n: load_model(n, a.models_dir) for n in names}
    cases = ds.cases("rerank", max_cases=a.cases, seed=2)
    print(f"rerank window: {len(cases):,} cases, sources {names}")
    t = time.time()
    rules = RERANK.__class__(candidates_per_source=a.per_source, train_cases=a.cases, negatives_per_case=RERANK.negatives_per_case, rounds=RERANK.rounds)
    rr = Reranker(ItemStats.from_dataset(ds), rules, names)
    tk = {}
    chunk = 5000
    # candidates are produced chunk by chunk inside fit; here we precompute per source for the sampled cases
    for n, m in models.items():
        idx_parts, sc_parts = [], []
        for start in range(0, len(cases), chunk):
            i, s = m.topk(cases.history[start : start + chunk], a.per_source)
            idx_parts.append(i)
            sc_parts.append(s)
        import numpy as np

        tk[n] = (np.vstack(idx_parts), np.vstack(sc_parts))
        print(f"  {n:12s} candidates ready")
    rr.fit(cases, tk)
    rr.save(a.models_dir / "reranker.joblib")
    info = {**rr.train_info, "sources": names, "seconds": round(time.time() - t, 1), "feature_importance": rr.feature_importance().to_dict(orient="records")}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "train_reranker.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(rr.feature_importance().head(12).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
