"""The API and the demo through FastAPI's TestClient on a synthetic serving bundle (no network, no raw data): the export and
its checks, a valid recommendation with seen-before flags consistent with the history and non-empty sources, every model,
the shared code path, the recent-popularity fallback, schema rejections, /similar, /health, logs without item ids, the
Gradio mount."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import export_serving  # noqa: E402

from recsys.api import create_app  # noqa: E402
from recsys.config import FIRST_ITEM, RerankRules  # noqa: E402
from recsys.data import load_tree  # noqa: E402
from recsys.pipeline import MODEL_FILES, source_topk  # noqa: E402
from recsys.rerank import ItemStats, Reranker  # noqa: E402
from recsys.serving import FALLBACK, SERVED, Service  # noqa: E402


@pytest.fixture(scope="module")
def bundle(ds, baselines, sasrec, two_tower, raw_dir, tmp_path_factory):
    models = {**baselines, "two_tower": two_tower, "sasrec": sasrec}
    mdir = tmp_path_factory.mktemp("models")
    for name, m in models.items():
        m.save(mdir / MODEL_FILES[name])
    rr_cases = ds.cases("rerank")
    rules = RerankRules(candidates_per_source=10, train_cases=len(rr_cases), negatives_per_case=10, rounds=40)
    Reranker(ItemStats.from_dataset(ds), rules).fit(rr_cases, source_topk(models, rr_cases.history, 10), models=models, log_fn=None).save(mdir / "reranker.joblib")
    out = tmp_path_factory.mktemp("serving")
    manifest = export_serving.export(ds, mdir, load_tree(raw_dir / "category_tree.csv"), out, n_shoppers=120, seed=3, check=40, log=None)
    return out, manifest


@pytest.fixture(scope="module")
def svc(bundle):
    return Service.load(bundle[0])


@pytest.fixture(scope="module")
def client(svc):
    return TestClient(create_app(svc))


def known_shopper(svc):
    return next(s for s in svc.shoppers if svc.history_matrix(s["history"])[1]["any_known"] and len(s["history"]) >= 3)


def test_export_bundle_and_checks(bundle, svc):
    out, m = bundle
    names = {p.name for p in out.iterdir()}
    assert {MODEL_FILES[n] for n in SERVED if n != "reranked"} | {"reranker.joblib", "popularity_recent.joblib", "similar_sasrec.npy", "items.parquet",
                                                                   "shoppers.json", "manifest.json"} <= names
    assert m["checks"]["served_history_equals_evaluation_history"]
    a, b = (int(x) for x in m["checks"]["served_reranked_equals_batched_evaluation_top10_known_histories"].split("/"))
    assert a == b > 0
    s = svc.shoppers[0]
    assert set(s) == {"id", "history", "window", "events_before", "bucket", "next_item", "next_event", "next_in_vocabulary", "next_seen_before"}  # no visitor id, no timestamp
    assert all(len(x["history"]) == len(x["window"]) <= svc.max_len and set(x["window"]) <= {"train", "rerank", "dev"} for x in svc.shoppers)
    assert all(p == "unknown" or p.split(" › ")[-1] == str(c) for p, c in zip(svc.path[FIRST_ITEM:], svc.category[FIRST_ITEM:]))


def test_recommend_reranked(client, svc):
    s = known_shopper(svc)
    r = client.post("/recommend", json={"history": s["history"], "k": 10})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["model"] == "reranked" and not b["fallback"] and b["version"] == svc.version and b["latency_ms"] > 0
    assert b["known_history"] + b["unknown_history"] == len(s["history"]) and b["history_used"] == len(s["history"][-svc.max_len :])
    items = b["items"]
    assert 0 < len(items) <= 10 and [x["rank"] for x in items] == list(range(1, len(items) + 1)) and len({x["item"] for x in items}) == len(items)
    seen = set(s["history"])
    assert all(x["seen_before"] == (x["item"] in seen) for x in items) and any(x["seen_before"] for x in items)
    assert all(x["sources"] and set(x["sources"]) <= set(SERVED) for x in items)
    assert all(x["popularity_rank"] >= 1 and x["category_path"] for x in items)


@pytest.mark.parametrize("model", [m for m in SERVED if m != "reranked"])
def test_every_single_model(client, svc, model):
    s = known_shopper(svc)
    b = client.post("/recommend", json={"history": s["history"], "k": 7, "model": model}).json()
    assert len(b["items"]) == 7 and all(x["sources"] == [model] for x in b["items"])
    assert all(x["seen_before"] == (x["item"] in set(s["history"])) for x in b["items"])


def test_served_list_is_the_evaluation_path(client, svc):
    s = known_shopper(svc)
    h, _ = svc.history_matrix(s["history"])
    idx, _ = svc.reranker.rerank(h, source_topk(svc.models, h, svc.reranker.rules.candidates_per_source), 10, models=svc.models)
    served = [x["item"] for x in client.post("/recommend", json={"history": s["history"], "k": 10}).json()["items"]]
    assert served == [int(svc.raw[i]) for i in idx[0] if i >= FIRST_ITEM]


def test_unknown_only_history_falls_back(client):
    b = client.post("/recommend", json={"history": [-5, -6], "k": 5}).json()
    assert b["fallback"] and b["known_history"] == 0 and b["unknown_history"] == 2 and "vocabulary" in b["note"]
    assert len(b["items"]) == 5 and all(x["sources"] == [FALLBACK] and not x["seen_before"] for x in b["items"])


def test_schema_rejections(client):
    assert client.post("/recommend", json={"history": [1, 2], "user_id": 7}).status_code == 422
    assert client.post("/recommend", json={"history": [1, 2], "k": 101}).status_code == 422
    assert client.post("/recommend", json={"history": [1, 2], "k": 0}).status_code == 422
    assert client.post("/recommend", json={"history": []}).status_code == 422
    assert client.post("/recommend", json={"history": list(range(201))}).status_code == 422
    assert client.post("/recommend", json={"history": [1], "model": "gpt"}).status_code == 422


def test_similar(client, svc):
    item = int(svc.raw[FIRST_ITEM + 3])
    b = client.get(f"/similar/{item}?k=5").json()
    assert b["item"] == item and len(b["items"]) == 5 and item not in {x["item"] for x in b["items"]}
    scores = [x["score"] for x in b["items"]]
    assert scores == sorted(scores, reverse=True) and all(-1.0001 <= x <= 1.0001 for x in scores)
    assert client.get("/similar/-12345").status_code == 404
    assert client.get(f"/similar/{item}?k=101").status_code == 422


def test_health(client, svc):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["models"] == list(SERVED) and h["n_items"] == svc.n_items - FIRST_ITEM and h["version"] == svc.version
    assert set(h["libraries"]) >= {"torch", "faiss", "lightgbm"} and h["reranker"]["candidates_per_source"] == 10


def test_logs_carry_no_item_ids(client, svc, caplog):
    s = known_shopper(svc)
    caplog.set_level(logging.INFO, logger="shop_recsys.api")
    client.post("/recommend", json={"history": s["history"], "k": 10})
    client.get(f"/similar/{s['history'][-1]}?k=3")
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "recommend model=reranked" in text and "/similar/{item}" in text
    assert not any(str(x) in text for x in s["history"])


def test_gradio_ui_is_mounted(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"] == "/ui"
    assert client.get("/ui/").status_code == 200 and "gradio" in client.get("/ui/").text.lower()
    cfg = client.get("/ui/config")
    assert cfg.status_code == 200 and "components" in cfg.json()
