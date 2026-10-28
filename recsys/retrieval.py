"""Approximate nearest-neighbour retrieval over item vectors with FAISS, and the check that it is honest:
recall of the ANN index against exact search, and the latency of both.

Rows PAD and UNK are never indexed; returned ids are item indices (≥ FIRST_ITEM).

HNSW over maximum inner product is not a metric-space search: with unnormalised item vectors (SASRec's, whose norms
grow with popularity) the graph misses a large share of the exact top-k, and the usual repair, appending
sqrt(M² − ‖x‖²) so every vector has the same length (Bachrach et al., RecSys 2014), was tried in M3 and did not
help (recall@100 0.81 → 0.81 at ef 128). At 70k items exact search is 0.3 ms per query, which is what is served;
`ann_check` keeps the honest comparison.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .config import FIRST_ITEM


class ItemIndex:
    def __init__(self, vectors: np.ndarray, kind: str = "flat", hnsw_m: int = 32, ef_construction: int = 200, ef_search: int = 64) -> None:
        import faiss

        self.kind, self.dim = kind, int(vectors.shape[1])
        self.n_items = int(vectors.shape[0])
        v = np.ascontiguousarray(vectors[FIRST_ITEM:], dtype=np.float32)
        if kind == "flat":
            self.index = faiss.IndexFlatIP(self.dim)
        elif kind == "hnsw":
            self.index = faiss.IndexHNSWFlat(self.dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = ef_construction
            self.index.hnsw.efSearch = ef_search
        else:
            raise ValueError(f"unknown index kind {kind!r}")
        self.index.add(v)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        scores, ids = self.index.search(np.ascontiguousarray(queries, dtype=np.float32), k)
        ids = np.where(ids >= 0, ids + FIRST_ITEM, FIRST_ITEM)  # −1 = not found (tiny catalogues); never leak a padding id
        return ids, scores

    def save(self, path: Path) -> None:
        import faiss

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: Path, kind: str) -> ItemIndex:
        import faiss

        obj = cls.__new__(cls)
        obj.index = faiss.read_index(str(path))
        obj.kind, obj.dim, obj.n_items = kind, obj.index.d, obj.index.ntotal + FIRST_ITEM
        return obj


def ann_check(vectors: np.ndarray, queries: np.ndarray, k: int = 100, ef_search: tuple[int, ...] = (32, 64, 128), single_queries: int = 200) -> dict:
    """Recall@k of HNSW against exact search, plus latency per query (batched and one at a time) for both."""
    flat = ItemIndex(vectors, "flat")
    t = time.perf_counter()
    exact_ids, _ = flat.search(queries, k)
    flat_batch_ms = (time.perf_counter() - t) * 1000 / len(queries)
    flat_single_ms = _single_latency(flat, queries[:single_queries], k)
    norms = np.linalg.norm(vectors[FIRST_ITEM:], axis=1)
    out = {"n_items": int(vectors.shape[0] - FIRST_ITEM), "dim": int(vectors.shape[1]), "k": k, "n_queries": len(queries),
           "item_norm_min_max": [round(float(norms.min()), 4), round(float(norms.max()), 4)],
           "flat": {"batch_ms_per_query": round(flat_batch_ms, 4), "single_ms_per_query": round(flat_single_ms, 4)}, "hnsw": {}}
    for ef in ef_search:
        t = time.perf_counter()
        hnsw = ItemIndex(vectors, "hnsw", ef_search=ef)
        build_s = time.perf_counter() - t
        t = time.perf_counter()
        ids, _ = hnsw.search(queries, k)
        batch_ms = (time.perf_counter() - t) * 1000 / len(queries)
        recall = float(np.mean([len(np.intersect1d(a, b)) / k for a, b in zip(ids, exact_ids)]))
        out["hnsw"][str(ef)] = {"recall_at_k": round(recall, 4), "batch_ms_per_query": round(batch_ms, 4),
                                "single_ms_per_query": round(_single_latency(hnsw, queries[:single_queries], k), 4), "build_seconds": round(build_s, 2)}
    return out


def _single_latency(index: ItemIndex, queries: np.ndarray, k: int) -> float:
    times = []
    for q in queries:
        t = time.perf_counter()
        index.search(q[None, :], k)
        times.append((time.perf_counter() - t) * 1000)
    return float(np.median(times)) if times else float("nan")
