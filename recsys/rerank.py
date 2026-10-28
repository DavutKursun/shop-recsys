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


def dense_scores(model, history: np.ndarray, case: np.ndarray, item: np.ndarray, block: int = 1000) -> np.ndarray:
    """Every (case, item) pair scored by one model: the user · item dot product for the embedding models, a gather from
    `score_batch` in blocks of at most `block` rows (one block of scores is block × n_items floats) for the rest."""
    out = np.empty(len(case), dtype=np.float32)
    if hasattr(model, "user_vectors") and hasattr(model, "item_vectors"):
        u, v = model.user_vectors(history), model.item_vectors()
        out[:] = np.einsum("ij,ij->i", u[case], v[item])
        return out
    bounds = np.searchsorted(case, np.arange(len(history) + 1))  # rows are sorted by case
    for start in range(0, len(history), block):
        stop = min(start + block, len(history))
        lo, hi = bounds[start], bounds[stop]
        if hi > lo:
            s = model.score_batch(history[start:stop])
            out[lo:hi] = s[case[lo:hi] - start, item[lo:hi]]
    return out


def candidate_frame(history: np.ndarray, source_topk: SourceTopK, stats: ItemStats, per_source: int, sources=SOURCES,
                    models: dict | None = None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Union of candidates for a chunk of cases → (features, case index per row, item per row). Rows are sorted by case.
    With `models`, every candidate is scored by every source (dense `<source>_score`), not only by the sources that
    proposed it; `<source>_rank` stays NaN for a source that did not propose the item and `n_sources` counts proposers."""
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
        rank = np.full(len(pairs), np.nan, dtype=np.float32)
        rank[pos] = np.tile(np.arange(1, k + 1, dtype=np.float32), n)
        if models is not None and s in models:
            score = dense_scores(models[s], history, case, item)
        else:
            score = np.full(len(pairs), np.nan, dtype=np.float32)
            score[pos] = sc.ravel()
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
    def fit(self, cases: Cases, source_topk: SourceTopK, models: dict | None = None, chunk: int = 5000, log_fn=print) -> Reranker:
        """10% of the cases are held out (by case, never by row): LightGBM early-stops on their ndcg@10 over the full
        candidate lists, and afterwards the reranked top-10 is scored on ALL of them, misses included, which is the
        number configurations are chosen on (`heldout_recall@10`). Training rows come from the other 90%: cases whose
        target is in the candidate union, with negatives downsampled to about `negatives_per_case`."""
        import lightgbm as lgb

        from .metrics import ranks_in_topk

        rng = np.random.default_rng(self.seed)
        n = len(cases)
        val_case = rng.random(n) < 0.1
        n_sources = len(source_topk)
        neg_rate = self.rules.negatives_per_case / max(self.rules.candidates_per_source * n_sources, 1)
        tr_frames, tr_labels, tr_groups, va_frames, va_labels, va_groups = [], [], [], [], [], []
        n_with_positive = 0
        for start in range(0, n, chunk):
            sl = slice(start, min(start + chunk, n))
            sub = {s: (i[sl], v[sl]) for s, (i, v) in source_topk.items()}
            X, case, item = candidate_frame(cases.history[sl], sub, self.stats, self.rules.candidates_per_source, self.sources, models)
            y = (item == cases.target[sl][case]).astype(np.int8)
            has_pos = np.zeros(sl.stop - sl.start, dtype=bool)
            has_pos[case[y == 1]] = True
            n_with_positive += int(has_pos.sum())
            is_val = val_case[sl][case]
            keep_tr = has_pos[case] & ~is_val & ((y == 1) | (rng.random(len(y)) < neg_rate))
            keep_va = has_pos[case] & is_val  # full lists: the early-stopping metric sees what inference sees
            for keep, frames, labels, groups in ((keep_tr, tr_frames, tr_labels, tr_groups), (keep_va, va_frames, va_labels, va_groups)):
                if keep.any():
                    frames.append(X[keep].reset_index(drop=True))
                    labels.append(y[keep])
                    counts = np.bincount(case[keep])
                    groups.append(counts[counts > 0])
        X_tr, y_tr, g_tr = pd.concat(tr_frames, ignore_index=True), np.concatenate(tr_labels), np.concatenate(tr_groups)
        X_va, y_va, g_va = pd.concat(va_frames, ignore_index=True), np.concatenate(va_labels), np.concatenate(va_groups)
        self.features = list(X_tr.columns)
        params = {"objective": "lambdarank", "metric": "ndcg", "eval_at": [10], "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 50,
                  "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0, "verbose": -1, "seed": self.seed, "num_threads": 0}
        d_tr = lgb.Dataset(X_tr, y_tr, group=g_tr)
        d_va = lgb.Dataset(X_va, y_va, group=g_va, reference=d_tr)
        self.booster = lgb.train(params, d_tr, num_boost_round=self.rules.rounds, valid_sets=[d_va], callbacks=[lgb.early_stopping(30, verbose=False)])
        # the held-out cases, every one of them: what the reranked top-10 is worth once misses are counted
        val = cases.subset(val_case)
        val_tk = {s: (i[val_case], v[val_case]) for s, (i, v) in source_topk.items()}
        idx, _ = self.rerank(val.history, val_tk, 10, models=models)
        ranks = ranks_in_topk(idx, val.target)
        hit = ranks > 0
        union_hit = np.concatenate([i[:, : self.rules.candidates_per_source] for i, _ in val_tk.values()], axis=1) == val.target[:, None]
        self.train_info = {"rows": int(len(y_tr)), "val_rows": int(len(y_va)), "cases_total": n, "cases_with_target_in_candidates": n_with_positive,
                           "candidate_recall": round(n_with_positive / max(n, 1), 4), "train_cases": int(len(g_tr)), "val_cases": int(val_case.sum()),
                           "best_iteration": int(self.booster.best_iteration), "val_ndcg10": round(float(self.booster.best_score["valid_0"]["ndcg@10"]), 4),
                           "heldout_recall@10": round(float(hit.mean()), 4), "heldout_ndcg@10": round(float(np.where(hit, 1.0 / np.log2(np.maximum(ranks, 1) + 1.0), 0.0).mean()), 4),
                           "heldout_union_recall": round(float(union_hit.any(axis=1).mean()), 4), "candidates_per_source": self.rules.candidates_per_source,
                           "sources": list(source_topk), "dense_scores": models is not None, "features": self.features}
        if log_fn:
            log_fn(f"reranker: {self.train_info['rows']:,} rows, candidate recall {self.train_info['candidate_recall']:.3f}, "
                   f"best iteration {self.train_info['best_iteration']}, val ndcg@10 {self.train_info['val_ndcg10']:.4f}, "
                   f"held-out recall@10 {self.train_info['heldout_recall@10']:.4f} (union {self.train_info['heldout_union_recall']:.4f})")
        return self

    # inference ----------------------------------------------------------------------------------------------------
    def rerank(self, history: np.ndarray, source_topk: SourceTopK, k: int, models: dict | None = None, chunk: int = 5000, with_sources: bool = False):
        """Top-k after reranking the candidate union; (idx, score[, sources per item]). Pass the models that produced
        `source_topk` so every candidate carries every source's dense score, as in training."""
        idx_out, score_out, src_out = [], [], []
        for start in range(0, len(history), chunk):
            sl = slice(start, start + chunk)
            sub = {s: (i[sl], v[sl]) for s, (i, v) in source_topk.items()}
            X, case, item = candidate_frame(history[sl], sub, self.stats, self.rules.candidates_per_source, self.sources, models)
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
