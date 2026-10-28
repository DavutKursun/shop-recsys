"""Glue used by the scripts and the tests: load saved models, score cases in chunks, build the evaluation block."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .base import Recommender
from .config import BOOTSTRAP_RESAMPLES, EVENT_NAMES, HISTORY_BUCKETS, MODELS, PRIMARY_K, RERANK, SOURCES, TOP_K
from .data import Cases, Dataset
from .metrics import coverage, paired_user_bootstrap, per_case, popularity_quantile, ranks_in_topk, user_bootstrap
from .rerank import Reranker

MODEL_FILES = {"popularity": "popularity.joblib", "repeat": "repeat.joblib", "markov": "markov.joblib", "item_knn": "item_knn.joblib",
               "sasrec": "sasrec.pt", "two_tower": "two_tower.pt"}


def load_model(name: str, models_dir: Path = MODELS) -> Recommender:
    path = models_dir / MODEL_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; train it first")
    if name == "sasrec":
        from .sasrec import SASRec

        return SASRec.load(path)
    if name == "two_tower":
        from .two_tower import TwoTower

        return TwoTower.load(path)
    return Recommender.load(path)


def available_models(models_dir: Path = MODELS) -> list[str]:
    return [n for n, f in MODEL_FILES.items() if (models_dir / f).exists()]


def source_topk(models: dict[str, Recommender], history: np.ndarray, k: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {name: m.topk(history, k) for name, m in models.items()}


def evaluate_models(cases: Cases, models: dict[str, Recommender], reranker: Reranker | None = None, n_items: int = 0, ks=TOP_K,
                    per_source: int = RERANK.candidates_per_source, chunk: int = 20_000, resamples: int = BOOTSTRAP_RESAMPLES,
                    baseline_names=("popularity", "repeat", "markov", "item_knn"), log_fn=print) -> dict:
    """Every model (and the reranked pipeline) on the same cases: metrics with user-bootstrap CIs, coverage,
    popularity bias, breakdowns, paired differences against the best baseline, candidate-union recall."""
    k_max = max(max(ks), per_source if reranker is not None else 0)
    names = list(models) + (["reranked"] if reranker is not None else [])
    ranks = {n: np.zeros(len(cases), dtype=np.int64) for n in names}
    top10 = {n: np.zeros((len(cases), PRIMARY_K), dtype=np.int64) for n in names}
    union_hit = np.zeros(len(cases), dtype=bool)
    for start in range(0, len(cases), chunk):
        sl = slice(start, min(start + chunk, len(cases)))
        h, t = cases.history[sl], cases.target[sl]
        tk = source_topk(models, h, k_max)
        for n, (idx, _) in tk.items():
            ranks[n][sl] = ranks_in_topk(idx[:, : max(ks)], t)
            top10[n][sl] = idx[:, :PRIMARY_K]
        if reranker is not None:
            union = np.concatenate([idx[:, :per_source] for idx, _ in tk.values()], axis=1)
            union_hit[sl] = (union == t[:, None]).any(axis=1)
            idx, _ = reranker.rerank(h, tk, max(ks))
            ranks["reranked"][sl] = ranks_in_topk(idx, t)
            top10["reranked"][sl] = idx[:, :PRIMARY_K]
        if log_fn:
            log_fn(f"  scored {sl.stop:,}/{len(cases):,} cases")
    out = {"n_cases": len(cases), "n_users": len(np.unique(cases.user)), "share_target_reachable": round(float(cases.reachable.mean()), 4),
           "share_repeat": round(float(cases.repeat.mean()), 4), "share_warm": round(float(cases.warm.mean()), 4), "models": {}, "breakdowns": {}, "paired": {}}
    per = {n: per_case(ranks[n], ks) for n in names}
    for n in names:
        block = user_bootstrap(per[n], cases.user, resamples)
        block["coverage@10"] = coverage(top10[n], n_items)
        block["popularity_quantile@10"] = popularity_quantile(top10[n], n_items)
        out["models"][n] = block
    if reranker is not None:
        out["candidate_union_recall"] = round(float(union_hit.mean()), 4)
    primary = f"recall@{PRIMARY_K}"
    # breakdowns: point estimates of the primary metric on subsets
    subsets = {"warm": cases.warm, "cold": ~cases.warm, "repeat": cases.repeat, "new": ~cases.repeat}
    for lo, hi in HISTORY_BUCKETS:
        subsets[f"hist_{lo}-{hi if hi < 10_000 else 'plus'}"] = (cases.hist_len >= lo) & (cases.hist_len <= hi)
    for code, label in EVENT_NAMES.items():
        subsets[f"target_{label}"] = cases.etype == code
    for label, mask in subsets.items():
        if mask.sum() == 0:
            continue
        out["breakdowns"][label] = {"n_cases": int(mask.sum()), **{n: round(float(per[n][primary][mask].mean()), 4) for n in names}}
    # paired comparisons on the primary metric
    baselines = [n for n in baseline_names if n in names]
    if baselines:
        best_base = max(baselines, key=lambda n: out["models"][n][primary]["point"])
        out["best_baseline"] = best_base
        for n in names:
            if n != best_base:
                out["paired"][f"{n}_minus_{best_base}"] = paired_user_bootstrap(per[n][primary], per[best_base][primary], cases.user, resamples)
    if reranker is not None:
        single = [n for n in models]
        best_single = max(single, key=lambda n: out["models"][n][primary]["point"])
        out["best_single"] = best_single
        out["paired"][f"reranked_minus_{best_single}"] = paired_user_bootstrap(per["reranked"][primary], per[best_single][primary], cases.user, resamples)
    return out


def results_table(results: dict, ks=TOP_K) -> str:
    """Markdown table: one row per model, point [CI] for every metric."""
    cols = [f"recall@{k}" for k in ks] + [f"ndcg@{PRIMARY_K}", f"mrr@{PRIMARY_K}"]
    head = "| model | " + " | ".join(cols) + " | coverage@10 | pop. quantile |\n|---|" + "---|" * (len(cols) + 2)
    rows = []
    order = sorted(results["models"], key=lambda n: -results["models"][n][f"recall@{PRIMARY_K}"]["point"])
    for n in order:
        m = results["models"][n]
        cells = [f"{m[c]['point']:.4f} [{m[c]['ci_low']:.4f}, {m[c]['ci_high']:.4f}]" for c in cols]
        rows.append(f"| {n} | " + " | ".join(cells) + f" | {m['coverage@10']:.3f} | {m['popularity_quantile@10']:.3f} |")
    return head + "\n" + "\n".join(rows)


def dump_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serialisable: {type(o)}")


def dataset_or_raise(splits_dir: Path) -> Dataset:
    return Dataset.load(splits_dir)


SOURCE_ORDER = SOURCES
