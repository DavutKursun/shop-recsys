"""The one interface every recommender implements: fit on a Dataset, score a batch of histories over the whole
catalogue, return top-k. Scores for PAD and UNK are −inf, so they are never recommended."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import FIRST_ITEM, PAD
from .data import Dataset


class Recommender:
    name = "base"

    def __init__(self) -> None:
        self.n_items = 0
        self.fitted = False

    def fit(self, ds: Dataset, **kw) -> Recommender:
        """kw reaches the model's _fit (neural models take eval_cases=..., patience=..., log_fn=...)."""
        self.n_items = ds.n_items
        self._fit(ds, **kw)
        self.fitted = True
        return self

    def _fit(self, ds: Dataset, **kw) -> None:
        raise NotImplementedError

    def score_batch(self, history: np.ndarray) -> np.ndarray:
        """(batch, max_len) left-padded item indices → (batch, n_items) float32 scores."""
        raise NotImplementedError

    def topk(self, history: np.ndarray, k: int, batch_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
        """Top-k item indices and scores per row, best first; computed in batches to bound memory."""
        idx_out, score_out = [], []
        for start in range(0, len(history), batch_size):
            s = self.score_batch(history[start : start + batch_size])
            s[:, :FIRST_ITEM] = -np.inf
            kk = min(k, s.shape[1] - FIRST_ITEM)
            part = np.argpartition(-s, kk - 1, axis=1)[:, :kk]
            part_scores = np.take_along_axis(s, part, axis=1)
            order = np.argsort(-part_scores, axis=1, kind="stable")
            idx_out.append(np.take_along_axis(part, order, axis=1))
            score_out.append(np.take_along_axis(part_scores, order, axis=1))
        return np.vstack(idx_out), np.vstack(score_out)

    def save(self, path: Path) -> None:
        import joblib

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> Recommender:
        import joblib

        return joblib.load(path)


def history_lengths(history: np.ndarray) -> np.ndarray:
    return (history != PAD).sum(axis=1)
