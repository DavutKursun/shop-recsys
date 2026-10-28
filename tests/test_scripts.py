"""The script chain on a synthetic raw export: download → build_dataset → train_baselines → train_neural (both, --fast)
→ build_index → train_reranker → evaluate (dev; test refused) → synthetic_check. Same chain as on the real files."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_dataset
import build_index
import download
import evaluate
import synthetic_check
import train_baselines
import train_neural
import train_reranker

from recsys.synthetic import make_world, write_raw


@pytest.fixture(scope="module")
def ws(tmp_path_factory):
    base = tmp_path_factory.mktemp("ws")
    write_raw(make_world(n_users=1500, seed=11), base / "raw")
    d = {"raw": base / "raw", "cache": base / "cache", "splits": base / "splits", "models": base / "models", "out": base / "results"}
    for k in ("splits", "models", "out"):
        d[k].mkdir(parents=True, exist_ok=True)
    return d


def test_download_reports_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(download, "RAW", tmp_path)
    assert download.main() == 1
    assert "kaggle" in capsys.readouterr().out.lower()


def test_build_dataset(ws):
    args = ["--raw", str(ws["raw"]), "--cache", str(ws["cache"]), "--out", str(ws["splits"]), "--days", "7", "--min-item-events", "3", "--max-len", "20"]
    assert build_dataset.main(args) == 0
    s = json.loads((ws["splits"] / "summary.json").read_text())
    assert s["vocab_items"] > 50 and s["by_window"]["test"]["events"] > 0 and (ws["cache"] / "raw_facts.json").exists()
    assert build_dataset.main(args) == 0  # second run uses the cache


def test_train_and_evaluate_chain(ws):
    common = ["--splits-dir", str(ws["splits"]), "--models-dir", str(ws["models"]), "--out", str(ws["out"])]
    assert train_baselines.main(["--neighbors", "30", *common]) == 0
    assert train_neural.main(["--model", "sasrec", "--fast", "--epochs", "2", "--max-len", "20", "--negatives", "64", "--dev-cases", "500", "--device", "cpu", *common]) == 0
    assert train_neural.main(["--model", "two_tower", "--fast", "--epochs", "2", "--max-len", "20", "--dev-cases", "500", "--device", "cpu", *common]) == 0
    assert (ws["models"] / "sasrec.pt").exists() and (ws["models"] / "two_tower.pt").exists()
    log = json.loads((ws["out"] / "train_sasrec.json").read_text())
    assert log["train_log"]["best_epoch"] >= 1
    assert build_index.main(["--model", "two_tower", "--k", "20", "--queries", "200", *common]) == 0
    assert (ws["models"] / "index_two_tower_hnsw.faiss").exists()
    assert train_reranker.main(["--cases", "1500", "--per-source", "15", *common]) == 0
    assert (ws["models"] / "reranker.joblib").exists()
    assert evaluate.main(["--split", "test", *common]) == 3  # refused without --final
    assert evaluate.main(["--split", "dev", "--resamples", "30", "--max-cases", "1500", "--sampled-demo", *common]) == 0
    res = json.loads((ws["out"] / "dev-summary.json").read_text())
    assert set(res["models"]) == {"popularity", "repeat", "hybrid", "markov", "item_knn", "two_tower", "sasrec", "reranked"}
    assert "sampled_negatives_demo" in res and res["candidate_union_recall"] > 0
    assert (ws["out"] / "dev-summary.md").exists()


def test_synthetic_check_script(ws):
    res = synthetic_check.run(users=1200, fast=True, out=ws["out"], log=None)
    assert res["models"]["sasrec"]["recall@10"]["point"] > res["models"]["popularity"]["recall@10"]["point"]
    assert (ws["out"] / "synthetic_check.json").exists()
