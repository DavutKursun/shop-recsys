"""Ranking metrics over the FULL catalogue, per case, with user-cluster bootstrap confidence intervals.

Why full ranking: scoring the target against 100 sampled negatives (common in papers) is not a consistent
estimate of the true metric and can even reorder models (Krichene & Rendle, KDD 2020). Every model here ranks
all vocabulary items; `sampled_ranks` exists only to demonstrate that trap on the dev split.

Why cluster bootstrap: a shopper with 40 events in the window contributes 40 cases that are not independent;
resampling users, not cases, gives honest intervals.
"""

from __future__ import annotations

import numpy as np

from .config import BOOTSTRAP_RESAMPLES, PRIMARY_K, SEED, TOP_K


def ranks_in_topk(topk_idx: np.ndarray, target: np.ndarray) -> np.ndarray:
    """1-based rank of the target inside each top-K list, 0 when it is not there (or not reachable)."""
    hit = topk_idx == target[:, None]
    pos = hit.argmax(axis=1) + 1
    return np.where(hit.any(axis=1), pos, 0).astype(np.int64)


def per_case(ranks: np.ndarray, ks=TOP_K) -> dict[str, np.ndarray]:
    out = {}
    r = ranks.astype(float)
    for k in ks:
        inside = (ranks > 0) & (ranks <= k)
        out[f"recall@{k}"] = inside.astype(float)
        safe = np.maximum(r, 1.0)
        out[f"ndcg@{k}"] = np.where(inside, 1.0 / np.log2(safe + 1.0), 0.0)
        out[f"mrr@{k}"] = np.where(inside, 1.0 / safe, 0.0)
    return out


def _user_sums(values: np.ndarray, users: np.ndarray):
    uniq, inv = np.unique(users, return_inverse=True)
    sums = np.bincount(inv, weights=values, minlength=len(uniq))
    counts = np.bincount(inv, minlength=len(uniq)).astype(float)
    return sums, counts


def _resampled_means(stats: list[np.ndarray], counts: np.ndarray, n_resamples: int, seed: int) -> np.ndarray:
    """For each resample of users (with replacement), the case-weighted mean of every statistic. (resamples, len(stats))."""
    rng = np.random.default_rng(seed)
    n_users = len(counts)
    out = np.empty((n_resamples, len(stats)))
    for r in range(n_resamples):
        w = np.bincount(rng.integers(0, n_users, n_users), minlength=n_users).astype(float)
        denom = w @ counts
        out[r] = [(w @ s) / denom for s in stats]
    return out


def user_bootstrap(values: dict[str, np.ndarray], users: np.ndarray, n_resamples: int = BOOTSTRAP_RESAMPLES, seed: int = SEED) -> dict[str, dict]:
    """Point estimate (mean over cases) and 95% CI from resampling users with replacement."""
    names = list(values)
    sums, counts = [], None
    for name in names:
        s, counts = _user_sums(values[name], users)
        sums.append(s)
    boot = _resampled_means(sums, counts, n_resamples, seed)
    return {name: {"point": float(values[name].mean()), "ci_low": float(np.percentile(boot[:, i], 2.5)), "ci_high": float(np.percentile(boot[:, i], 97.5)),
                   "n_cases": len(values[name]), "n_users": len(counts)} for i, name in enumerate(names)}


def paired_user_bootstrap(a: np.ndarray, b: np.ndarray, users: np.ndarray, n_resamples: int = BOOTSTRAP_RESAMPLES, seed: int = SEED) -> dict:
    """CI of mean(a) − mean(b) on the same cases, same user resamples: the test for 'model A beats model B'."""
    sums_a, counts = _user_sums(a, users)
    sums_b, _ = _user_sums(b, users)
    boot = _resampled_means([sums_a, sums_b], counts, n_resamples, seed)
    diff = boot[:, 0] - boot[:, 1]
    lo, hi = float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))
    return {"point": float(a.mean() - b.mean()), "ci_low": lo, "ci_high": hi, "n_resamples": n_resamples, "significant": bool(lo > 0 or hi < 0)}


def coverage(topk_idx: np.ndarray, n_items: int, k: int = PRIMARY_K) -> float:
    """Share of the vocabulary that appears in at least one top-k list."""
    from .config import FIRST_ITEM

    return float(len(np.unique(topk_idx[:, :k])) / max(n_items - FIRST_ITEM, 1))


def popularity_quantile(topk_idx: np.ndarray, n_items: int, k: int = PRIMARY_K) -> float:
    """Mean popularity quantile of recommended items: 0 = always the most popular item, 0.5 = like random."""
    from .config import FIRST_ITEM

    q = (topk_idx[:, :k] - FIRST_ITEM) / max(n_items - FIRST_ITEM - 1, 1)
    return float(np.clip(q, 0, 1).mean())


def sampled_ranks(scores: np.ndarray, target: np.ndarray, n_negatives: int = 100, seed: int = SEED) -> np.ndarray:
    """The paper trap: rank of the target among itself and `n_negatives` uniformly sampled vocabulary items.
    `scores` is a (batch, n_items) block of full scores. Returns 1-based ranks."""
    from .config import FIRST_ITEM

    rng = np.random.default_rng(seed)
    n, v = scores.shape
    neg = rng.integers(FIRST_ITEM, v, size=(n, n_negatives))
    s_t = scores[np.arange(n), target][:, None]
    s_n = np.take_along_axis(scores, neg, axis=1)
    return 1 + (s_n > s_t).sum(axis=1) + 0.5 * (s_n == s_t).sum(axis=1)


def summarize(ranks: np.ndarray, users: np.ndarray, ks=TOP_K, n_resamples: int = BOOTSTRAP_RESAMPLES, seed: int = SEED) -> dict:
    """The standard block for one model on one window: every metric with its user-bootstrap CI."""
    return user_bootstrap(per_case(ranks, ks), users, n_resamples, seed)
