"""Stage two: a LightGBM LambdaRank model over the union of every retriever's top candidates.

Candidates for a case are the union of the top-K lists of the sources (popularity, repeat, Markov, item-kNN,
two-tower, SASRec). Features are the sources' scores and ranks (missing when a source did not propose the
item), item statistics from the training window, user–item signals (seen before, how recently, category
match with the history) and the case's history length. The label is "this candidate is the next item".

The reranker is trained on the rerank window [T0, T1) with retrievers fitted on events before T0, so it
never sees a score the retrievers produced on their own training data. Its ceiling is the recall of the
candidate union, which is reported next to it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FIRST_ITEM, PAD, RERANK, SEED, SOURCES, RerankRules
from .data import DAY_MS, Cases, Dataset

SourceTopK = dict[str, tuple[np.ndarray, np.ndarray]]  # source → (idx (n, K), score (n, K))


@dataclass
class ItemStats:
    """Per-item arrays indexed by item index (PAD and UNK rows are zero)."""

    log_count: np.ndarray
    log_recent: np.ndarray
    days_since_first: np.ndarray
    days_since_last: np.ndarray
    category: np.ndarray
    n_items: int

    @classmethod
    def from_dataset(cls, ds: Dataset) -> ItemStats:
        n = ds.n_items
        idx = ds.items["iidx"].to_numpy()
        z = lambda: np.zeros(n, dtype=np.float32)
        log_count, log_recent, d_first, d_last = z(), z(), z(), z()
        cat = np.zeros(n, dtype=np.int64)
        log_count[idx] = np.log1p(ds.items["train_count"].to_numpy(dtype=float))
        log_recent[idx] = np.log1p(ds.items["recent_count"].to_numpy(dtype=float))
        d_first[idx] = (ds.windows.t0 - ds.items["first_ts"].to_numpy(dtype=float)) / DAY_MS
        d_last[idx] = (ds.windows.t0 - ds.items["last_ts"].to_numpy(dtype=float)) / DAY_MS
        cat[idx] = ds.items["category"].to_numpy()
        return cls(log_count, log_recent, d_first, d_last, cat, n)


def candidate_frame(history: np.ndarray, source_topk: SourceTopK, stats: ItemStats, per_source: int, sources=SOURCES) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Union of candidates for a chunk of cases → (features, case index per row, item per row). Rows are sorted by case."""
    n, L = history.shape
    keys = []
    for s in sources:
        if s not in source_topk:
            continue
        idx = source_topk[s][0][:, :per_source]
        keys.append(np.repeat(np.arange(n), idx.shape[1]) * stats.n_items + idx.ravel())
    pairs = np.unique(np.concatenate(keys))
    case, item = pairs // stats.n_items, pairs % stats.n_items
    feats: dict[str, np.ndarray] = {}
    n_sources = np.zeros(len(pairs), dtype=np.float32)
    for s in sources:
        if s not in source_topk:
            continue
        idx, sc = source_topk[s][0][:, :per_source], source_topk[s][1][:, :per_source]
        k = idx.shape[1]
        key = np.repeat(np.arange(n), k) * stats.n_items + idx.ravel()
        pos = np.searchsorted(pairs, key)
        score = np.full(len(pairs), np.nan, dtype=np.float32)
        rank = np.full(len(pairs), np.nan, dtype=np.float32)
        score[pos] = sc.ravel()
        rank[pos] = np.tile(np.arange(1, k + 1, dtype=np.float32), n)
        feats[f"{s}_score"], feats[f"{s}_rank"] = score, rank
        n_sources[pos] += 1
    feats["n_sources"] = n_sources
    feats["log_count"], feats["log_recent"] = stats.log_count[item], stats.log_recent[item]
    feats["days_since_first"], feats["days_since_last"] = stats.days_since_first[item], stats.days_since_last[item]
    # user–item signals from the history matrix
    h = history[case]  # (rows, L)
    valid = h != PAD
    seen = np.zeros(len(pairs), dtype=np.float32)
    last_pos = np.full(len(pairs), L, dtype=np.float32)  # positions from the end; L = never
    cat_match = np.zeros(len(pairs), dtype=np.float32)
    item_cat = stats.category[item]
    for j in range(L):
        eq = h[:, j] == item
        seen += eq
        last_pos = np.where(eq, L - 1 - j, last_pos)
        cat_match += (stats.category[h[:, j]] == item_cat) & valid[:, j]
    hist_len = valid.sum(1).astype(np.float32)
    feats["seen_count"], feats["last_seen_pos"] = seen, last_pos
    feats["cat_share"] = cat_match / np.maximum(hist_len, 1)
    feats["same_cat_as_last"] = (stats.category[h[:, -1]] == item_cat).astype(np.float32)
    feats["hist_len"] = hist_len
    feats["n_known"] = (h >= FIRST_ITEM).sum(1).astype(np.float32)
    return pd.DataFrame(feats), case, item


class Reranker:
    name = "reranked"

    def __init__(self, stats: ItemStats, rules: RerankRules = RERANK, sources=SOURCES, seed: int = SEED) -> None:
        self.stats, self.rules, self.sources, self.seed = stats, rules, tuple(sources), seed
        self.booster = None
        self.features: list[str] = []
        self.train_info: dict = {}

    # training -----------------------------------------------------------------------------------------------------
    def fit(self, cases: Cases, source_topk: SourceTopK, chunk: int = 5000, log_fn=print) -> Reranker:
        import lightgbm as lgb

        rng = np.random.default_rng(self.seed)
        frames, labels, groups = [], [], []
        n_with_positive = 0
        for start in range(0, len(cases), chunk):
            sl = slice(start, start + chunk)
            sub = {s: (i[sl], v[sl]) for s, (i, v) in source_topk.items()}
            X, case, item = candidate_frame(cases.history[sl], sub, self.stats, self.rules.candidates_per_source, self.sources)
            y = (item == cases.target[sl][case]).astype(np.int8)
            # keep cases that contain their target; downsample negatives
            has_pos = np.zeros(len(cases.target[sl]), dtype=bool)
            has_pos[case[y == 1]] = True
            keep = has_pos[case] & ((y == 1) | (rng.random(len(y)) < self.rules.negatives_per_case / max(self.rules.candidates_per_source * len(sub), 1)))
            X, y, case = X[keep].reset_index(drop=True), y[keep], case[keep]
            n_with_positive += int(has_pos.sum())
            frames.append(X)
            labels.append(y)
            groups.append(np.bincount(case)[np.bincount(case) > 0])
        X = pd.concat(frames, ignore_index=True)
        y = np.concatenate(labels)
        group = np.concatenate(groups)
        self.features = list(X.columns)
        # hold out 10% of cases for early stopping
        n_groups = len(group)
        val = rng.random(n_groups) < 0.1
        bounds = np.concatenate([[0], np.cumsum(group)])
        row_val = np.zeros(len(y), dtype=bool)
        for g in np.flatnonzero(val):
            row_val[bounds[g] : bounds[g + 1]] = True
        params = dict(objective="lambdarank", metric="ndcg", eval_at=[10], learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8,
                      bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=self.seed, num_threads=0)
        d_tr = lgb.Dataset(X[~row_val], y[~row_val], group=group[~val])
        d_va = lgb.Dataset(X[row_val], y[row_val], group=group[val], reference=d_tr)
        self.booster = lgb.train(params, d_tr, num_boost_round=self.rules.rounds, valid_sets=[d_va], callbacks=[lgb.early_stopping(30, verbose=False)])
        self.train_info = {"rows": len(y), "cases_with_target_in_candidates": n_with_positive, "cases_total": len(cases),
                           "candidate_recall": round(n_with_positive / max(len(cases), 1), 4), "best_iteration": int(self.booster.best_iteration),
                           "val_ndcg10": round(float(self.booster.best_score["valid_0"]["ndcg@10"]), 4), "features": self.features}
        if log_fn:
            log_fn(f"reranker: {self.train_info['rows']:,} rows, candidate recall {self.train_info['candidate_recall']:.3f}, "
                   f"best iteration {self.train_info['best_iteration']}, val ndcg@10 {self.train_info['val_ndcg10']:.4f}")
        return self

    # inference ----------------------------------------------------------------------------------------------------
    def rerank(self, history: np.ndarray, source_topk: SourceTopK, k: int, chunk: int = 5000, with_sources: bool = False):
        """Top-k after reranking the candidate union; (idx, score[, sources per item])."""
        idx_out, score_out, src_out = [], [], []
        for start in range(0, len(history), chunk):
            sl = slice(start, start + chunk)
            sub = {s: (i[sl], v[sl]) for s, (i, v) in source_topk.items()}
            X, case, item = candidate_frame(history[sl], sub, self.stats, self.rules.candidates_per_source, self.sources)
            pred = self.booster.predict(X[self.features], num_iteration=self.booster.best_iteration)
            n = len(history[sl])
            idx = np.full((n, k), PAD, dtype=np.int64)
            sc = np.full((n, k), -np.inf, dtype=np.float32)
            order = np.lexsort((-pred, case))  # by case, then best score first
            case_sorted, item_sorted, pred_sorted = case[order], item[order], pred[order]
            starts = np.searchsorted(case_sorted, np.arange(n))
            ends = np.searchsorted(case_sorted, np.arange(n), side="right")
            for c in range(n):
                a, b = starts[c], min(ends[c], starts[c] + k)
                idx[c, : b - a], sc[c, : b - a] = item_sorted[a:b], pred_sorted[a:b]
            idx_out.append(idx)
            score_out.append(sc)
            if with_sources:
                rank_cols = [f"{s}_rank" for s in self.sources if f"{s}_rank" in X]
                present = X[rank_cols].notna().to_numpy()
                names = [c[: -len("_rank")] for c in rank_cols]
                lookup = {(int(c), int(i)): [names[j] for j in np.flatnonzero(row)] for c, i, row in zip(case, item, present)}
                src_out.extend([[lookup.get((c, int(i)), []) for i in idx[c]] for c in range(n)])
        idx, sc = np.vstack(idx_out), np.vstack(score_out)
        return (idx, sc, src_out) if with_sources else (idx, sc)

    def feature_importance(self) -> pd.DataFrame:
        gain = self.booster.feature_importance(importance_type="gain")
        return pd.DataFrame({"feature": self.features, "gain": gain}).sort_values("gain", ascending=False).reset_index(drop=True)

    # persistence --------------------------------------------------------------------------------------------------
    def save(self, path: Path) -> None:
        import joblib

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"stats": self.stats, "rules": self.rules, "sources": self.sources, "features": self.features, "train_info": self.train_info,
                     "booster": self.booster.model_to_string()}, path)

    @classmethod
    def load(cls, path: Path) -> Reranker:
        import joblib
        import lightgbm as lgb

        blob = joblib.load(path)
        r = cls(blob["stats"], blob["rules"], blob["sources"])
        r.features, r.train_info = blob["features"], blob["train_info"]
        r.booster = lgb.Booster(model_str=blob["booster"])
        return r
