"""The whole pipeline on the synthetic shop (known structure): raw files → dataset → baselines → SASRec →
two-tower → reranker → dev table. Sequential models must beat popularity here, or something is broken.

    python scripts/synthetic_check.py [--users 3000] [--fast] [--out eval/results]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# sets OMP_NUM_THREADS and checks the macOS OpenMP fix before torch / faiss / lightgbm load
import recsys  # noqa: E402,F401
from recsys.baselines import ItemKNN, Markov, Popularity, Repeat
from recsys.config import RESULTS, SASREC, TWO_TOWER, RerankRules, WindowRules
from recsys.data import Dataset, cache_raw, find_raw
from recsys.pipeline import evaluate_models, results_table, source_topk
from recsys.rerank import ItemStats, Reranker
from recsys.sasrec import SASRec
from recsys.synthetic import make_world, write_raw
from recsys.two_tower import TwoTower


def run(users: int, fast: bool, out: Path, seed: int = 7, log=print) -> dict:
    t0 = time.time()
    world = make_world(n_users=users, seed=seed)
    with tempfile.TemporaryDirectory() as tmp:
        raw = write_raw(world, Path(tmp) / "raw")
        assert find_raw(Path(tmp) / "raw") is not None
        facts = cache_raw(raw, Path(tmp) / "cache")
        import pandas as pd

        ds = Dataset.build(pd.read_parquet(Path(tmp) / "cache" / "events.parquet"), pd.read_parquet(Path(tmp) / "cache" / "categories.parquet"),
                           WindowRules(days=7, min_item_events=3, max_len=20))
    models = {"popularity": Popularity().fit(ds), "repeat": Repeat().fit(ds), "markov": Markov().fit(ds), "item_knn": ItemKNN(neighbors=50).fit(ds)}
    dev_sub = ds.cases("dev", max_cases=2000, seed=1)
    sas_cfg = replace(SASREC, dim=32, layers=1, heads=2, max_len=20, negatives=128, batch_size=64, lr=3e-3, epochs=3 if fast else 15)
    tt_cfg = replace(TWO_TOWER, dim=32, hidden=64, max_len=20, batch_size=256, epochs=3 if fast else 8)
    models["two_tower"] = TwoTower(tt_cfg, device="cpu").fit(ds, eval_cases=dev_sub, patience=2, log_fn=log)
    models["sasrec"] = SASRec(sas_cfg, device="cpu").fit(ds, eval_cases=dev_sub, patience=2, log_fn=log)
    rr_cases = ds.cases("rerank")
    rules = RerankRules(candidates_per_source=20, train_cases=len(rr_cases), negatives_per_case=20, rounds=100)
    rr = Reranker(ItemStats.from_dataset(ds), rules).fit(rr_cases, source_topk(models, rr_cases.history, 20), log_fn=log)
    dev = ds.cases("dev")
    res = evaluate_models(dev, models, rr, ds.n_items, per_source=20, resamples=100, log_fn=None)
    res.update({"raw_facts": facts, "summary": ds.summary(), "seconds": round(time.time() - t0, 1), "n_users": users, "fast": fast})
    out.mkdir(parents=True, exist_ok=True)
    (out / "synthetic_check.json").write_text(json.dumps(res, indent=2, default=_default), encoding="utf-8")
    return res


def _default(o):
    return int(o) if isinstance(o, np.integer) else float(o) if isinstance(o, np.floating) else o.tolist() if isinstance(o, np.ndarray) else str(o)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=3000)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", type=Path, default=RESULTS)
    a = ap.parse_args(argv)
    res = run(a.users, a.fast, a.out)
    print(results_table(res))
    pop, best = res["models"]["popularity"]["recall@10"]["point"], max(res["models"][n]["recall@10"]["point"] for n in ("sasrec", "two_tower", "markov", "item_knn"))
    print(f"popularity {pop:.4f} vs best sequential {best:.4f} ({best / max(pop, 1e-9):.1f}×); reranked {res['models']['reranked']['recall@10']['point']:.4f}; {res['seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
