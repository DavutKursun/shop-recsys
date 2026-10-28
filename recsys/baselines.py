"""The classical baselines every learned model must beat: popularity, repeat, first-order Markov, session item-kNN.

They are cheap, strong on e-commerce data, and the honest yardstick. Each one scores the whole catalogue
from a left-padded history matrix (newest item in the last column).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .base import Recommender
from .config import FIRST_ITEM, PAD
from .data import Dataset


def _popularity_prior(ds: Dataset) -> np.ndarray:
    """A tiny, scale-free tie-breaker: popularity quantile in (0, 1e-6)."""
    prior = np.zeros(ds.n_items, dtype=np.float32)
    counts = ds.items["train_count"].to_numpy(dtype=float)
    prior[FIRST_ITEM:] = np.log1p(counts) / (np.log1p(counts).max() + 1e-9)
    return prior * 1e-6


class Popularity(Recommender):
    """Rank by training-window count (or by the last `recent_days` only). Same list for everyone."""

    name = "popularity"

    def __init__(self, recent: bool = False) -> None:
        super().__init__()
        self.recent = recent
        self.vector = None

    def _fit(self, ds: Dataset, **kw) -> None:
        col = "recent_count" if self.recent else "train_count"
        v = np.zeros(ds.n_items, dtype=np.float32)
        v[FIRST_ITEM:] = np.log1p(ds.items[col].to_numpy(dtype=float))
        self.vector = v + _popularity_prior(ds) * (1 if self.recent else 0)

    def score_batch(self, history: np.ndarray) -> np.ndarray:
        return np.tile(self.vector, (len(history), 1))


class Repeat(Recommender):
    """Recommend what the shopper already looked at, most recent first; popularity fills the rest."""

    name = "repeat"

    def __init__(self, decay: float = 0.8) -> None:
        super().__init__()
        self.decay = decay
        self.prior = None

    def _fit(self, ds: Dataset, **kw) -> None:
        self.prior = _popularity_prior(ds)

    def score_batch(self, history: np.ndarray) -> np.ndarray:
        b, L = history.shape
        s = np.tile(self.prior, (b, 1))
        weights = self.decay ** np.arange(L - 1, -1, -1, dtype=np.float32)  # last column = most recent = weight 1
        rows = np.repeat(np.arange(b), L)
        cols = history.ravel()
        w = np.tile(weights, b)
        keep = cols >= FIRST_ITEM
        np.maximum.at(s, (rows[keep], cols[keep]), w[keep])
        return s


class Markov(Recommender):
    """P(next | last item) from consecutive pairs inside training sessions; the second-to-last item adds a
    smaller vote. Unseen last items fall back to popularity."""

    name = "markov"

    def __init__(self, second_weight: float = 0.3) -> None:
        super().__init__()
        self.second_weight = second_weight
        self.P = None
        self.prior = None

    def _fit(self, ds: Dataset, **kw) -> None:
        a, b = [], []
        for s in ds.train_sessions():
            if len(s) >= 2:
                a.append(s[:-1])
                b.append(s[1:])
        if not a:
            self.P = sp.csr_matrix((ds.n_items, ds.n_items), dtype=np.float32)
        else:
            a, b = np.concatenate(a), np.concatenate(b)
            keep = (a >= FIRST_ITEM) & (b >= FIRST_ITEM)
            counts = sp.coo_matrix((np.ones(keep.sum(), dtype=np.float32), (a[keep], b[keep])), shape=(ds.n_items, ds.n_items)).tocsr()
            counts.sum_duplicates()
            row_sum = np.asarray(counts.sum(axis=1)).ravel()
            inv = np.where(row_sum > 0, 1.0 / np.maximum(row_sum, 1e-9), 0.0).astype(np.float32)
            self.P = sp.diags(inv) @ counts
        self.prior = _popularity_prior(ds)

    def score_batch(self, history: np.ndarray) -> np.ndarray:
        last, prev = history[:, -1], history[:, -2] if history.shape[1] > 1 else np.full(len(history), PAD)
        s = np.asarray(self.P[last].todense(), dtype=np.float32)
        if self.second_weight > 0:
            s += self.second_weight * np.asarray(self.P[prev].todense(), dtype=np.float32)
        return s + self.prior


class ItemKNN(Recommender):
    """Session-based item-item cosine similarity (items co-occurring in training sessions), top-`neighbors` per
    item; a history is scored by the recency-weighted sum of its items' similarity rows."""

    name = "item_knn"

    def __init__(self, neighbors: int = 100, decay: float = 0.7, block: int = 2000) -> None:
        super().__init__()
        self.neighbors, self.decay, self.block = neighbors, decay, block
        self.S = None
        self.prior = None

    def _fit(self, ds: Dataset, **kw) -> None:
        sessions = ds.train_sessions()
        rows, cols = [], []
        for i, s in enumerate(sessions):
            u = np.unique(s[s >= FIRST_ITEM])
            rows.append(np.full(len(u), i))
            cols.append(u)
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        U = sp.csc_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(sessions), ds.n_items))
        n = np.asarray(U.sum(axis=0)).ravel()
        norm = np.sqrt(np.maximum(n, 1e-9))
        parts = []
        for start in range(FIRST_ITEM, ds.n_items, self.block):
            stop = min(start + self.block, ds.n_items)
            C = (U[:, start:stop].T @ U).tocoo()  # co-occurrence counts for a block of items
            keep = C.col != C.row + start  # no self-similarity
            data = C.data[keep] / (norm[start + C.row[keep]] * norm[C.col[keep]])
            C = sp.csr_matrix((data.astype(np.float32), (C.row[keep], C.col[keep])), shape=(stop - start, ds.n_items))
            parts.append(_topk_rows(C, self.neighbors))
        top = sp.vstack(parts).tocsr()
        pad = sp.csr_matrix((FIRST_ITEM, ds.n_items), dtype=np.float32)
        self.S = sp.vstack([pad, top]).tocsr()
        self.prior = _popularity_prior(ds)

    def score_batch(self, history: np.ndarray) -> np.ndarray:
        b, L = history.shape
        weights = self.decay ** np.arange(L - 1, -1, -1, dtype=np.float32)
        rows = np.repeat(np.arange(b), L)
        cols = history.ravel()
        w = np.tile(weights, b)
        keep = cols >= FIRST_ITEM
        H = sp.csr_matrix((w[keep], (rows[keep], cols[keep])), shape=(b, self.n_items))
        return np.asarray((H @ self.S).todense(), dtype=np.float32) + self.prior


def _topk_rows(C: sp.csr_matrix, k: int) -> sp.csr_matrix:
    """Keep the k largest entries of every row of a CSR matrix."""
    C = C.tocsr()
    data, indices, indptr = [], [], [0]
    for r in range(C.shape[0]):
        lo, hi = C.indptr[r], C.indptr[r + 1]
        d, ix = C.data[lo:hi], C.indices[lo:hi]
        if len(d) > k:
            sel = np.argpartition(-d, k - 1)[:k]
            d, ix = d[sel], ix[sel]
        data.append(d)
        indices.append(ix)
        indptr.append(indptr[-1] + len(d))
    if not data:
        return sp.csr_matrix(C.shape, dtype=np.float32)
    return sp.csr_matrix((np.concatenate(data).astype(np.float32), np.concatenate(indices), np.array(indptr)), shape=C.shape)


class Hybrid(Recommender):
    """The honest bar: the shopper's own items first (repeat, most recent first), then the Markov continuation of the
    last items, then popularity. Repeat and the sequential baselines are complementary on this data (DESIGN.md §2);
    every learned model has to beat their union, not popularity."""

    name = "hybrid"

    def __init__(self, decay: float = 0.8, second_weight: float = 0.3) -> None:
        super().__init__()
        self.repeat, self.markov = Repeat(decay), Markov(second_weight)

    def _fit(self, ds: Dataset, **kw) -> None:
        self.repeat.fit(ds)
        self.markov.fit(ds)

    def score_batch(self, history: np.ndarray) -> np.ndarray:
        rep = self.repeat.score_batch(history)  # seen items: recency weight in [decay^(L−1), 1]; unseen: the tiny popularity prior
        floor = max(self.repeat.decay ** (history.shape[1] - 1), 2e-6)  # the smallest weight a seen item can carry (above the prior)
        mk = self.markov.score_batch(history)  # transitions ≥ 0 with the prior as a tie-breaker
        mk = 0.5 * floor * mk / (mk.max(axis=1, keepdims=True) + 1e-12)  # scaled strictly below the repeat band
        return np.where(rep >= floor, rep, mk).astype(np.float32)


BASELINES = {c.name: c for c in (Popularity, Repeat, Hybrid, Markov, ItemKNN)}
