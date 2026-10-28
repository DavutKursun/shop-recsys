"""Inference cost per source on dev cases: ms per 1,000 cases for every source's top-50, the reranker's feature build
and its LightGBM predict, and single-request latency (one shopper: every source + features + predict). Feeds the M5
latency budget (p50 < 50 ms per request).

    python scripts/bench_sources.py [--cases 10000] [--single 200]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# sets OMP_NUM_THREADS and checks the macOS OpenMP fix before torch / faiss / lightgbm load
import recsys  # noqa: E402,F401
from recsys.config import MODELS, RESULTS, SOURCES, SPLITS
from recsys.data import Dataset
from recsys.pipeline import available_models, load_model
from recsys.rerank import Reranker, candidate_frame


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=10_000)
    ap.add_argument("--single", type=int, default=200, help="single-request latency: this many one-case requests")
    ap.add_argument("--chunk", type=int, default=1000)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--splits-dir", type=Path, default=SPLITS)
    ap.add_argument("--models-dir", type=Path, default=MODELS)
    ap.add_argument("--out", type=Path, default=RESULTS)
    a = ap.parse_args(argv)
    ds = Dataset.load(a.splits_dir)
    names = [n for n in SOURCES if n in available_models(a.models_dir)]
    models = {n: load_model(n, a.models_dir) for n in names}
    rr = Reranker.load(a.models_dir / "reranker.joblib") if (a.models_dir / "reranker.joblib").exists() else None
    cases = ds.cases("dev", max_cases=a.cases, seed=5)
    h = cases.history
    per_1000 = {}
    tk = {}
    for n, m in models.items():
        t = time.perf_counter()
        parts_i, parts_s = [], []
        for start in range(0, len(h), a.chunk):
            i, s = m.topk(h[start : start + a.chunk], a.k)
            parts_i.append(i)
            parts_s.append(s)
        per_1000[n] = round((time.perf_counter() - t) * 1000 / len(h) * 1000, 1)
        tk[n] = (np.vstack(parts_i), np.vstack(parts_s))
        print(f"{n:12s} top-{a.k}: {per_1000[n]:8.1f} ms per 1,000 cases")
    report = {"n_cases": len(h), "k": a.k, "chunk": a.chunk, "ms_per_1000_cases": per_1000}
    if rr is not None:
        t_feat = t_pred = 0.0
        rows = 0
        for start in range(0, len(h), a.chunk):
            sl = slice(start, start + a.chunk)
            sub = {s: (i[sl], v[sl]) for s, (i, v) in tk.items()}
            t = time.perf_counter()
            X, case, item = candidate_frame(h[sl], sub, rr.stats, rr.rules.candidates_per_source, rr.sources, models)
            t_feat += time.perf_counter() - t
            t = time.perf_counter()
            rr.booster.predict(X[rr.features], num_iteration=rr.booster.best_iteration)
            t_pred += time.perf_counter() - t
            rows += len(X)
        report["reranker"] = {"features_ms_per_1000_cases": round(t_feat * 1000 / len(h) * 1000, 1), "predict_ms_per_1000_cases": round(t_pred * 1000 / len(h) * 1000, 1),
                              "candidates_per_case": round(rows / len(h), 1), "best_iteration": int(rr.booster.best_iteration)}
        print(f"reranker     features {report['reranker']['features_ms_per_1000_cases']:.1f} ms, predict {report['reranker']['predict_ms_per_1000_cases']:.1f} ms per 1,000 cases "
              f"({report['reranker']['candidates_per_case']:.0f} candidates per case)")
        # one shopper at a time: the request path of the API (every source's top-k, the features, the reranker)
        single = {n: [] for n in names}
        single["features+predict"], single["request_total"] = [], []
        for i in range(min(a.single, len(h))):
            one = h[i : i + 1]
            t_req = time.perf_counter()
            sub = {}
            for n, m in models.items():
                t = time.perf_counter()
                sub[n] = m.topk(one, a.k)
                single[n].append((time.perf_counter() - t) * 1000)
            t = time.perf_counter()
            rr.rerank(one, sub, 10, models=models)
            single["features+predict"].append((time.perf_counter() - t) * 1000)
            single["request_total"].append((time.perf_counter() - t_req) * 1000)
        report["single_request_ms"] = {k: {"p50": round(float(np.percentile(v, 50)), 2), "p90": round(float(np.percentile(v, 90)), 2)} for k, v in single.items()}
        print("single request (ms, p50 / p90): " + ", ".join(f"{k} {v['p50']} / {v['p90']}" for k, v in report["single_request_ms"].items()))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "source_cost.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
