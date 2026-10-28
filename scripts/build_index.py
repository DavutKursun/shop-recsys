"""FAISS over a model's item vectors: exact (flat) vs HNSW recall and latency on dev user vectors; saves the HNSW index.

    python scripts/build_index.py --model sasrec [--k 100] [--queries 5000]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# sets OMP_NUM_THREADS and checks the macOS OpenMP fix before torch / faiss / lightgbm load
import recsys  # noqa: E402,F401
from recsys.config import MODELS, RESULTS, SPLITS
from recsys.data import Dataset
from recsys.pipeline import load_model
from recsys.retrieval import ItemIndex, ann_check


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sasrec", choices=["sasrec", "two_tower"])
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--queries", type=int, default=5000)
    ap.add_argument("--ef", default="32,64,128", help="HNSW efSearch values to check, comma list")
    ap.add_argument("--splits-dir", type=Path, default=SPLITS)
    ap.add_argument("--models-dir", type=Path, default=MODELS)
    ap.add_argument("--out", type=Path, default=RESULTS)
    a = ap.parse_args(argv)
    ds = Dataset.load(a.splits_dir)
    m = load_model(a.model, a.models_dir)
    cases = ds.cases("dev", max_cases=a.queries, seed=4)
    queries = m.user_vectors(cases.history)
    report = {"model": a.model, **ann_check(m.item_vectors(), queries, k=a.k, ef_search=tuple(int(x) for x in a.ef.split(",")))}
    ItemIndex(m.item_vectors(), "hnsw").save(a.models_dir / f"index_{a.model}_hnsw.faiss")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"ann_{a.model}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
