"""Metrics, baselines, SASRec, two-tower, FAISS retrieval, the reranker and the API schema — on the synthetic shop."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from recsys.config import FIRST_ITEM, PAD, UNK, RerankRules
from recsys.metrics import (
    coverage,
    paired_user_bootstrap,
    per_case,
    popularity_quantile,
    ranks_in_topk,
    sampled_ranks,
    summarize,
    user_bootstrap,
)
from recsys.pipeline import evaluate_models, results_table, source_topk
from recsys.rerank import ItemStats, Reranker, candidate_frame
from recsys.retrieval import ItemIndex, ann_check
from recsys.schema import Recommendation, RecommendRequest, RecommendResponse


def recall10(model, cases) -> float:
    idx, _ = model.topk(cases.history, 10)
    return float((ranks_in_topk(idx, cases.target) > 0).mean())


# --- metrics -----------------------------------------------------------------------------------------------------
def test_ranks_and_per_case_metrics():
    topk = np.array([[5, 3, 9], [2, 4, 6], [7, 8, 1]])
    target = np.array([3, 6, 4])
    r = ranks_in_topk(topk, target)
    assert r.tolist() == [2, 3, 0]
    m = per_case(r, ks=(2, 3))
    assert m["recall@2"].tolist() == [1.0, 0.0, 0.0] and m["recall@3"].tolist() == [1.0, 1.0, 0.0]
    assert m["ndcg@3"][0] == pytest.approx(1 / np.log2(3)) and m["mrr@3"][1] == pytest.approx(1 / 3)


def test_user_bootstrap_and_paired():
    rng = np.random.default_rng(0)
    users = np.repeat(np.arange(200), 5)
    a = (rng.random(1000) < 0.3).astype(float)
    b = (rng.random(1000) < 0.2).astype(float)
    ci = user_bootstrap({"m": a}, users, n_resamples=200)["m"]
    assert ci["ci_low"] <= ci["point"] <= ci["ci_high"] and ci["n_users"] == 200 and ci["n_cases"] == 1000
    d = paired_user_bootstrap(a, b, users, n_resamples=200)
    assert d["point"] == pytest.approx(a.mean() - b.mean()) and d["ci_low"] < d["point"] < d["ci_high"]
    same = paired_user_bootstrap(a, a, users, n_resamples=50)
    assert same["point"] == 0 and not same["significant"]
    s = summarize(np.array([1, 0, 3, 0]), np.array([0, 0, 1, 1]), ks=(3,), n_resamples=20)
    assert s["recall@3"]["point"] == 0.5


def test_coverage_popularity_and_sampled_ranks():
    topk = np.array([[2, 3], [2, 4], [2, 3]])
    assert coverage(topk, n_items=12, k=2) == pytest.approx(3 / 10)
    assert popularity_quantile(np.array([[2, 2]]), n_items=12, k=2) == 0.0  # index 2 = most popular
    scores = np.tile(np.arange(100, dtype=float), (4, 1))  # higher index = higher score
    target = np.array([99, 50, 10, 2])
    r = sampled_ranks(scores, target, n_negatives=50, seed=0)
    assert r[0] == 1 and r[1] > 1 and r[3] >= r[2]  # the top item is always rank 1; weak items drop


# --- baselines -----------------------------------------------------------------------------------------------------
def test_baselines_behave(ds, dev, baselines):
    pop = baselines["popularity"]
    idx, _ = pop.topk(dev.history[:5], 3)
    assert (idx == idx[0]).all() and idx[0, 0] == FIRST_ITEM  # the same list for everyone, most popular first
    rep = baselines["repeat"]
    idx, _ = rep.topk(dev.history[:50], 3)
    last = dev.history[:50, -1]
    assert (idx[:, 0] == last).mean() > 0.9  # the most recent item comes first
    mk = baselines["markov"]
    P = mk.P
    rows = np.flatnonzero(np.asarray(P.sum(axis=1)).ravel() > 0)[:30]
    h = np.full((len(rows), 20), PAD, dtype=np.int64)
    h[:, -1] = rows
    idx, _ = mk.topk(h, 1)
    top_true = np.array([P[r].toarray().ravel().argmax() for r in rows])
    assert (idx[:, 0] == top_true).all()
    knn = baselines["item_knn"]
    assert knn.S.shape == (ds.n_items, ds.n_items) and knn.S[:FIRST_ITEM].nnz == 0 and knn.S.diagonal().sum() == 0
    for m in baselines.values():
        s = m.score_batch(dev.history[:4])
        assert s.shape == (4, ds.n_items) and np.isfinite(s[:, FIRST_ITEM:]).all()
        idx, sc = m.topk(dev.history[:4], 10)
        assert idx.shape == (4, 10) and (idx >= FIRST_ITEM).all() and (np.diff(sc, axis=1) <= 1e-6).all()


def test_hybrid_is_repeat_then_markov(ds, dev, baselines):
    hy, rep, mk = baselines["hybrid"], baselines["repeat"], baselines["markov"]
    h = dev.history[:200]
    n_seen = np.array([len(set(row[row >= FIRST_ITEM])) for row in h])
    idx, sc = hy.topk(h, 10)
    rep_idx, _ = rep.topk(h, 10)
    for i in np.flatnonzero(n_seen >= 2)[:50]:
        k = min(n_seen[i], 10)
        assert (idx[i, :k] == rep_idx[i, :k]).all()  # the seen items come first, in repeat's recency order
        seen = set(h[i][h[i] >= FIRST_ITEM])
        assert not any(j in seen for j in idx[i, k:])  # and never again below them
    one = np.flatnonzero(n_seen == 1)[:50]  # single-item history: the item itself, then its Markov successors
    mk_idx, _ = mk.topk(h[one], 10)
    assert (idx[one, 0] == h[one, -1]).all()
    assert np.mean([(idx[i, 1] == mk_idx[j, 0]) or mk_idx[j, 0] == h[i, -1] for j, i in enumerate(one)]) > 0.9
    s = hy.score_batch(h[:4])
    assert s.shape == (4, ds.n_items) and np.isfinite(s[:, FIRST_ITEM:]).all() and (np.diff(sc, axis=1) <= 1e-9).all()
    r = {n: recall10(m, dev) for n, m in baselines.items()}
    assert r["hybrid"] > r["repeat"] and r["hybrid"] > 3 * r["popularity"], r


def test_sequential_baselines_beat_popularity_on_the_synthetic_shop(dev, baselines):
    r = {n: recall10(m, dev) for n, m in baselines.items()}
    assert r["markov"] > 1.5 * r["popularity"] and r["item_knn"] > 1.5 * r["popularity"], r
    assert r["repeat"] > r["popularity"], r


# --- neural models -----------------------------------------------------------------------------------------------
def test_sasrec_learns_and_roundtrips(ds, dev, baselines, sasrec, tmp_path):
    from recsys.sasrec import SASRec

    r_pop, r_sas = recall10(baselines["popularity"], dev), recall10(sasrec, dev)
    assert r_sas > 1.5 * r_pop, (r_sas, r_pop)
    assert sasrec.log.best_epoch >= 1 and sasrec.log.epochs[-1]["loss"] < sasrec.log.epochs[0]["loss"]
    s = sasrec.score_batch(dev.history[:8])
    assert s.shape == (8, ds.n_items) and np.isinf(s[:, :FIRST_ITEM]).all()
    idx, sc = sasrec.topk(dev.history[:8], 5)
    assert (idx == np.argsort(-s, axis=1)[:, :5]).all()
    sasrec.save(tmp_path / "s.pt")
    back = SASRec.load(tmp_path / "s.pt")
    assert np.allclose(back.score_batch(dev.history[:8]), s, atol=1e-5)
    assert sasrec.item_vectors().shape == (ds.n_items, 32) and sasrec.user_vectors(dev.history[:3]).shape == (3, 32)


def test_sasrec_full_softmax_loss_runs(ds):
    from dataclasses import replace

    import torch

    from recsys.config import SASREC
    from recsys.sasrec import SASRec, SASRecNet

    cfg = replace(SASREC, dim=16, layers=1, heads=2, max_len=10, negatives=0)
    m = SASRec(cfg, device="cpu")
    m.n_items = ds.n_items
    m.net = SASRecNet(ds.n_items, cfg)
    seq = torch.tensor([[0, 0, 5, 6, 7, 8, 9, 10, 11, 12], [0, 0, 0, 0, 0, 0, 0, 0, 3, 4]])
    tgt = torch.tensor([[0, 0, 6, 7, 8, 9, 10, 11, 12, 13], [0, 0, 0, 0, 0, 0, 0, 0, 4, UNK]])
    loss = m._loss(seq, tgt, torch.Generator().manual_seed(0), torch.device("cpu"))
    assert torch.isfinite(loss) and loss > 0


def test_two_tower_learns_and_roundtrips(ds, dev, baselines, two_tower, tmp_path):
    from recsys.two_tower import TwoTower

    r_pop, r_tt = recall10(baselines["popularity"], dev), recall10(two_tower, dev)
    assert r_tt > 1.3 * r_pop, (r_tt, r_pop)
    v = two_tower.item_vectors()
    assert v.shape == (ds.n_items, 32) and np.allclose(np.linalg.norm(v[FIRST_ITEM:], axis=1), 1.0, atol=1e-4)
    u = two_tower.user_vectors(dev.history[:5])
    assert np.allclose(np.linalg.norm(u, axis=1), 1.0, atol=1e-4)
    two_tower.save(tmp_path / "t.pt")
    back = TwoTower.load(tmp_path / "t.pt")
    assert np.allclose(back.score_batch(dev.history[:5]), two_tower.score_batch(dev.history[:5]), atol=1e-5)
    cold = two_tower.embed_new_items(np.array([1, 2]), np.array([1, 1]))
    assert cold.shape == (2, 32) and np.allclose(np.linalg.norm(cold, axis=1), 1.0, atol=1e-4)


# --- retrieval ---------------------------------------------------------------------------------------------------
def test_faiss_flat_is_exact_and_hnsw_is_close(dev, two_tower, tmp_path):
    vectors, queries = two_tower.item_vectors(), two_tower.user_vectors(dev.history[:300])
    flat = ItemIndex(vectors, "flat")
    ids, sc = flat.search(queries, 10)
    exact = np.argsort(-(queries @ vectors[FIRST_ITEM:].T), axis=1)[:, :10] + FIRST_ITEM
    assert (ids == exact).mean() > 0.99 and (ids >= FIRST_ITEM).all()
    rep = ann_check(vectors, queries, k=20, ef_search=(64,), single_queries=20)
    assert rep["hnsw"]["64"]["recall_at_k"] >= 0.9 and rep["flat"]["single_ms_per_query"] > 0
    flat.save(tmp_path / "i.faiss")
    back = ItemIndex.load(tmp_path / "i.faiss", "flat")
    assert back.n_items == flat.n_items and (back.search(queries[:3], 5)[0] == ids[:3, :5]).all()


# --- reranker ----------------------------------------------------------------------------------------------------
def test_candidate_frame_features(ds, dev, baselines):
    stats = ItemStats.from_dataset(ds)
    tk = source_topk(baselines, dev.history[:50], 10)
    X, case, item = candidate_frame(dev.history[:50], tk, stats, per_source=10, models=baselines)
    assert len(X) == len(case) == len(item) and (np.diff(case) >= 0).all() and len(np.unique(case * ds.n_items + item)) == len(X)
    assert set(X.columns) >= {"popularity_rank", "markov_score", "n_sources", "seen_count", "last_seen_pos", "cat_share", "same_cat_as_last", "hist_len", "log_count"}
    assert X["n_sources"].between(1, len(baselines)).all() and X["seen_count"].max() >= 1  # repeat proposes seen items
    rep_rows = X["repeat_rank"] == 1
    assert (X.loc[rep_rows, "seen_count"] >= 1).all() and (X.loc[rep_rows, "last_seen_pos"] == 0).all()
    # dense scores: every candidate carries every source's score; the rank is NaN only where the source did not propose it
    assert X["markov_score"].notna().all() and X["markov_rank"].isna().any() and X["repeat_rank"].isna().any()
    full = baselines["markov"].score_batch(dev.history[:50])
    assert np.allclose(X["markov_score"].to_numpy(), full[case, item], atol=1e-6)
    X2, _, _ = candidate_frame(dev.history[:50], tk, stats, per_source=10)  # without models: scores only where proposed
    assert X2["markov_score"].isna().any() and np.allclose(X2["markov_rank"].to_numpy(), X["markov_rank"].to_numpy(), equal_nan=True)


def test_reranker_fits_and_does_not_collapse(ds, dev, baselines, sasrec, two_tower):
    models = {**baselines, "two_tower": two_tower, "sasrec": sasrec}
    rr_cases = ds.cases("rerank")
    rules = RerankRules(candidates_per_source=20, train_cases=len(rr_cases), negatives_per_case=20, rounds=60)
    rr = Reranker(ItemStats.from_dataset(ds), rules).fit(rr_cases, source_topk(models, rr_cases.history, 20), models=models, log_fn=None)
    assert rr.train_info["candidate_recall"] > 0.5 and rr.booster is not None and rr.train_info["dense_scores"]
    assert 0 < rr.train_info["heldout_recall@10"] <= rr.train_info["heldout_union_recall"] <= 1 and rr.train_info["val_cases"] > 0
    tk3 = source_topk(models, dev.history[:3], 20)
    dense = rr.rerank(dev.history[:3], tk3, 5, models=models)[0]
    assert dense.shape == (3, 5) and (dense >= FIRST_ITEM).all()
    res = evaluate_models(dev, models, rr, ds.n_items, per_source=20, resamples=50, log_fn=None)
    best_single = max(models, key=lambda n: res["models"][n]["recall@10"]["point"])
    assert res["models"]["reranked"]["recall@10"]["point"] >= 0.9 * res["models"][best_single]["recall@10"]["point"]
    assert res["candidate_union_recall"] >= res["models"]["reranked"]["recall@20"]["point"]
    assert set(res["breakdowns"]) >= {"warm", "cold", "repeat", "new", "hist_1-1", "target_view"} and "best_baseline" in res
    assert "reranked" in results_table(res)
    idx, sc, src = rr.rerank(dev.history[:3], source_topk(models, dev.history[:3], 20), 5, models=models, with_sources=True)
    assert idx.shape == (3, 5) and len(src) == 3 and all(len(s) >= 1 for s in src[0])
    imp = rr.feature_importance()
    assert imp["gain"].iloc[0] > 0


# --- schema ------------------------------------------------------------------------------------------------------
def test_schema():
    r = RecommendRequest(history=[1, 2, 3])
    assert r.k == 10 and r.model == "reranked"
    with pytest.raises(ValidationError):
        RecommendRequest(history=[], k=5)
    with pytest.raises(ValidationError):
        RecommendRequest(history=[1], user_id=5)
    resp = RecommendResponse(items=[Recommendation(item=7, score=0.5, rank=1)], model="sasrec", known_history=1, version="x", latency_ms=1.0)
    assert resp.items[0].rank == 1
