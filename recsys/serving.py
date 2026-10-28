"""Serving: the exported bundle (scripts/export_serving.py → serving/) loaded once, every request answered through
`pipeline.recommend`, the code path the evaluation used. Item ids in and out are raw RetailRocket ids; a history item
outside the 69,784-item vocabulary is read as the UNK token and counted; the models read the most recent 50 items.
Nothing is stored.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .base import Recommender
from .config import FIRST_ITEM, ROOT, SOURCES, UNK, WINDOWS
from .pipeline import load_model, recommend
from .rerank import Reranker
from .retrieval import ItemIndex
from .schema import Recommendation, RecommendResponse, SimilarItem, SimilarResponse
from .torch_utils import left_pad

SERVING = ROOT / "serving"
FALLBACK = "popularity_recent"
FALLBACK_FILE = "popularity_recent.joblib"
SIMILAR_FILE = "similar_sasrec.npy"
SERVED = (*SOURCES, "reranked")  # popularity, repeat, hybrid, markov, item_knn, two_tower, sasrec, reranked


class Service:
    def __init__(self, models: dict[str, Recommender], reranker: Reranker, fallback: Recommender, items: pd.DataFrame,
                 similar: np.ndarray, shoppers: list[dict], manifest: dict) -> None:
        self.models, self.reranker, self.fallback, self.manifest, self.shoppers = models, reranker, fallback, manifest, shoppers
        self.n_items = FIRST_ITEM + len(items)
        iidx = items["iidx"].to_numpy()
        self.raw = np.zeros(self.n_items, dtype=np.int64)
        self.raw[iidx] = items["item"].to_numpy()
        self.category = np.full(self.n_items, -1, dtype=np.int64)
        self.category[iidx] = items["category"].to_numpy()
        self.popularity_rank = np.zeros(self.n_items, dtype=np.int64)
        self.popularity_rank[iidx] = items["popularity_rank"].to_numpy()
        self.path = np.full(self.n_items, "", dtype=object)
        self.path[iidx] = items["category_path"].to_numpy()
        self.index_of = dict(zip(items["item"].tolist(), iidx.tolist()))
        self.similar_vectors = np.ascontiguousarray(similar, dtype=np.float32)
        self.similar_index = ItemIndex(self.similar_vectors, "flat")  # exact search
        self.max_len = int(manifest.get("max_len", WINDOWS.max_len))
        self.version = f"{__version__}+{manifest.get('git_commit', 'dev')}"
        self.loaded_at = time.strftime("%Y-%m-%d %H:%M:%S")

    @classmethod
    def load(cls, serving_dir: Path = SERVING) -> Service:
        serving_dir = Path(serving_dir)
        if not (serving_dir / "manifest.json").exists():
            raise FileNotFoundError(f"no exported bundle in {serving_dir}; run python scripts/export_serving.py")
        manifest = json.loads((serving_dir / "manifest.json").read_text(encoding="utf-8"))
        models = {n: load_model(n, serving_dir) for n in SOURCES}
        return cls(models, Reranker.load(serving_dir / "reranker.joblib"), Recommender.load(serving_dir / FALLBACK_FILE),
                   pd.read_parquet(serving_dir / "items.parquet"), np.load(serving_dir / SIMILAR_FILE),
                   json.loads((serving_dir / "shoppers.json").read_text(encoding="utf-8")), manifest)

    # --- histories ---------------------------------------------------------------------------------------------------
    def to_index(self, raw_items) -> np.ndarray:
        return np.array([self.index_of.get(int(r), UNK) for r in raw_items], dtype=np.int64)

    def history_matrix(self, history_raw: list[int]) -> tuple[np.ndarray, dict]:
        """(1, max_len) left-padded item indices of the most recent items, exactly as an evaluation case is built, plus counts."""
        idx = self.to_index(history_raw)
        used = idx[-self.max_len :]
        known = int((idx != UNK).sum())
        meta = {"known_history": known, "unknown_history": int(len(idx) - known), "history_used": int(len(used)), "any_known": bool((used != UNK).any())}
        return left_pad([used], self.max_len), meta

    def item_info(self, raw_item: int) -> dict | None:
        i = self.index_of.get(int(raw_item))
        if i is None:
            return None
        return {"item": int(raw_item), "category": self._cat(i), "category_path": self.path[i] or None, "popularity_rank": int(self.popularity_rank[i])}

    def _cat(self, i: int) -> int | None:
        return int(self.category[i]) if self.category[i] >= 0 else None

    def _items(self, idx: np.ndarray, scores: np.ndarray, sources: list, seen: set[int]) -> list[Recommendation]:
        out = []
        for i, s, src in zip(idx, scores, sources):
            if i < FIRST_ITEM:  # PAD: the candidate union was smaller than k
                continue
            raw = int(self.raw[i])
            out.append(Recommendation(item=raw, score=round(float(s), 6), rank=len(out) + 1, category=self._cat(i), category_path=self.path[i] or None,
                                      popularity_rank=int(self.popularity_rank[i]), seen_before=raw in seen, sources=list(src)))
        return out

    # --- requests -----------------------------------------------------------------------------------------------------
    def lists(self, history_raw: list[int], k: int = 10, names: tuple[str, ...] = ("reranked",)) -> tuple[dict[str, list[Recommendation]], dict]:
        """Ranked lists of several models for one history, from one pass of every source (the demo's five lists)."""
        h, meta = self.history_matrix(history_raw)
        seen = {int(x) for x in history_raw}
        if not meta["any_known"]:
            idx, sc = self.fallback.topk(h, k)
            fb = self._items(idx[0], sc[0], [[FALLBACK]] * idx.shape[1], seen)
            note = (f"none of the {meta['history_used']} history items the models read is in the {self.n_items - FIRST_ITEM:,}-item vocabulary; "
                    "showing the most popular items of the last training week instead (no personalisation)")
            return {n: fb for n in names}, {**meta, "fallback": True, "note": note}
        res = recommend(self.models, self.reranker, h, k, tuple(names))
        lists = {n: self._items(res[n][0][0], res[n][1][0], res[n][2][0], seen) for n in names}
        note = None
        if "reranked" in lists and len(lists["reranked"]) < k:
            note = f"the reranked list holds {len(lists['reranked'])} items: the union of the sources' candidates is smaller than k = {k}"
        return lists, {**meta, "fallback": False, "note": note}

    def recommend(self, history_raw: list[int], k: int = 10, model: str = "reranked") -> RecommendResponse:
        t0 = time.perf_counter()
        lists, meta = self.lists(history_raw, k, (model,))
        return RecommendResponse(items=lists[model], model=model, known_history=meta["known_history"], unknown_history=meta["unknown_history"],
                                 history_used=meta["history_used"], fallback=meta["fallback"], note=meta["note"], version=self.version,
                                 latency_ms=round((time.perf_counter() - t0) * 1000, 2))

    def similar(self, raw_item: int, k: int = 10) -> SimilarResponse | None:
        """Nearest items by cosine similarity of the SASRec item embeddings (exact search); None outside the vocabulary."""
        t0 = time.perf_counter()
        i = self.index_of.get(int(raw_item))
        if i is None:
            return None
        ids, scores = self.similar_index.search(self.similar_vectors[i][None, :], k + 1)
        items = []
        for j, s in zip(ids[0], scores[0]):
            if j == i or j < FIRST_ITEM:
                continue
            if len(items) == k:
                break
            items.append(SimilarItem(item=int(self.raw[j]), score=round(float(s), 6), rank=len(items) + 1, category=self._cat(j),
                                     category_path=self.path[j] or None, popularity_rank=int(self.popularity_rank[j])))
        info = self.item_info(raw_item)
        return SimilarResponse(**info, items=items, version=self.version, latency_ms=round((time.perf_counter() - t0) * 1000, 2))

    def health(self) -> dict:
        import faiss
        import lightgbm
        import torch

        return {"status": "ok", "models": list(SERVED), "default_model": "reranked", "fallback": FALLBACK, "n_items": self.n_items - FIRST_ITEM,
                "history_read": self.max_len, "reranker": {"candidates_per_source": self.reranker.rules.candidates_per_source,
                                                           "best_iteration": int(self.reranker.booster.best_iteration), "features": len(self.reranker.features)},
                "similar": "exact cosine search (FAISS IndexFlatIP) over the SASRec item embeddings", "shoppers": len(self.shoppers),
                "version": self.version, "exported": self.manifest.get("created"), "loaded_at": self.loaded_at,
                "libraries": {"torch": torch.__version__, "faiss": faiss.__version__, "lightgbm": lightgbm.__version__, "numpy": np.__version__}}
