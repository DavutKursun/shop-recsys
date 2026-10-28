"""Loader, windows, sessions, vocabulary, evaluation cases — on the synthetic shop written in the RetailRocket layout."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recsys.config import FIRST_ITEM, PAD, UNK
from recsys.data import Dataset, dedup_consecutive, find_raw, load_categories, load_events, load_tree, make_windows, session_ids
from tests.conftest import RULES


def test_raw_files_load_like_the_real_ones(world, raw_dir):
    raw = find_raw(raw_dir)
    assert raw is not None and len(raw["properties"]) == 2
    ev = load_events(raw["events"])
    assert len(ev) == len(world["events"]) and list(ev.columns) == ["ts", "user", "item", "etype"]
    assert (ev["user"].to_numpy()[1:] >= ev["user"].to_numpy()[:-1]).all()  # sorted by user
    assert set(ev["etype"].unique()) <= {0, 1, 2}
    cat = load_categories(raw["properties"])
    truth = world["categories"].set_index("item")["category"]
    assert len(cat) == len(truth) and (cat.set_index("item")["category"].reindex(truth.index) == truth).all()  # latest value wins
    tree = load_tree(raw["tree"])
    assert all(tree[c] == p for c, p in zip(world["categories"]["category"], world["categories"]["parent"]))
    assert all(tree[p] == -1 for p in world["categories"]["parent"].unique())


def test_dedup_sessions_windows():
    df = pd.DataFrame({"ts": [0, 1, 2, 3, 100 * 60_000, 100 * 60_000 + 5], "user": [1, 1, 1, 1, 1, 2], "item": [7, 7, 8, 7, 7, 7], "etype": 0})
    d = dedup_consecutive(df)
    assert d["item"].tolist() == [7, 8, 7, 7, 7]  # the refresh is dropped, the later re-visit stays, the far re-visit stays
    s = session_ids(d, gap_minutes=30)
    assert s.tolist() == [0, 0, 0, 1, 2]
    w = make_windows(ts_max=100 * 86_400_000, rules=RULES)
    assert w.t0 < w.t1 < w.t2 < w.end and (w.t1 - w.t0) == 7 * 86_400_000
    codes = w.code(np.array([w.t0 - 1, w.t0, w.t1, w.t2, w.end - 1]))
    assert codes.tolist() == [0, 1, 2, 3, 3]


def test_dataset_vocab_and_items(ds, world):
    assert ds.n_items > FIRST_ITEM + 50
    items = ds.items
    assert (items["iidx"].to_numpy() == np.arange(FIRST_ITEM, ds.n_items)).all()
    assert (np.diff(items["train_count"].to_numpy()) <= 0).all()  # ordered by popularity
    assert (items["popularity_rank"] == items["iidx"] - 1).all()
    assert (items["train_count"] >= RULES.min_item_events).all()
    assert items["category"].min() >= 1 and items["parent"].min() >= 1  # every synthetic item has a known category
    ev = ds.events
    assert set(ev["window"].unique()) == {0, 1, 2, 3}
    train_items = ev.loc[ev["window"] == 0, "iidx"]
    assert (train_items[train_items >= FIRST_ITEM].value_counts().min()) >= RULES.min_item_events
    s = ds.summary()
    assert s["vocab_items"] == ds.n_items - FIRST_ITEM and s["by_window"]["dev"]["events"] > 0 and 0 < s["by_window"]["dev"]["share_events_in_vocab"] <= 1


def test_cases_use_only_the_past(ds, dev):
    assert len(dev) > 500
    ev = ds.events
    assert (dev.history[:, -1] != PAD).all()  # every case has a history
    assert (dev.hist_len >= 1).all() and (dev.hist_len >= (dev.history != PAD).sum(1)).all()
    assert dev.history.shape[1] == RULES.max_len
    # pick a few cases and rebuild the history by hand from the events table
    rng = np.random.default_rng(0)
    for i in rng.choice(len(dev), 20, replace=False):
        u, ts = dev.user[i], dev.ts[i]
        past = ev[(ev["uidx"] == u) & (ev["ts"] < ts)]
        assert len(past) == dev.hist_len[i]
        k = min(len(past), RULES.max_len)
        assert (dev.history[i, -k:] == past["iidx"].to_numpy()[-k:]).all()
        assert dev.warm[i] == bool((past["window"] < 2).any())
        assert dev.repeat[i] == (dev.target[i] != UNK and dev.target[i] in set(past["iidx"]))
    assert 0 < dev.warm.mean() < 1 and 0 < dev.repeat.mean() < 1
    sub = ds.cases("dev", max_cases=100, seed=1)
    assert len(sub) == 100 and (np.diff(sub.ts) != 0).any()
    rr = ds.cases("rerank")
    assert (rr.ts < ds.windows.t1).all() and (rr.ts >= ds.windows.t0).all()


def test_train_sequences_and_sessions(ds):
    seqs = ds.train_sequences()
    assert all(len(s) >= RULES.min_train_events for s in seqs) and all((s != PAD).all() for s in seqs)
    sessions = ds.train_sessions()
    assert sum(len(s) for s in sessions) == int((ds.events["window"] == 0).sum())


def test_save_load_roundtrip(ds, tmp_path):
    meta = ds.save(tmp_path / "splits")
    back = Dataset.load(tmp_path / "splits")
    assert back.n_items == ds.n_items and back.windows == ds.windows and meta["n_items"] == ds.n_items
    assert len(back.cases("dev")) == len(ds.cases("dev"))
    assert (back.raw_to_index(ds.items["item"].head(5).to_numpy()) == ds.items["iidx"].head(5).to_numpy()).all()
    assert back.raw_to_index([-12345])[0] == UNK
    with pytest.raises(FileNotFoundError):
        Dataset.load(tmp_path / "nope")
