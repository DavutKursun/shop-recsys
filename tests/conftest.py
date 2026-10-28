from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys.baselines import ItemKNN, Markov, Popularity, Repeat
from recsys.config import SASREC, TWO_TOWER, WindowRules
from recsys.data import Dataset, cache_raw
from recsys.synthetic import make_world, write_raw

RULES = WindowRules(days=7, min_item_events=3, max_len=20)


@pytest.fixture(scope="session")
def world():
    return make_world(n_users=2500, seed=5)


@pytest.fixture(scope="session")
def raw_dir(world, tmp_path_factory):
    d = tmp_path_factory.mktemp("raw")
    write_raw(world, d)
    return d


@pytest.fixture(scope="session")
def ds(world, raw_dir, tmp_path_factory):
    cache = tmp_path_factory.mktemp("cache")
    from recsys.data import find_raw

    cache_raw(find_raw(raw_dir), cache)
    return Dataset.build(pd.read_parquet(cache / "events.parquet"), pd.read_parquet(cache / "categories.parquet"), RULES)


@pytest.fixture(scope="session")
def dev(ds):
    return ds.cases("dev")


@pytest.fixture(scope="session")
def baselines(ds):
    return {"popularity": Popularity().fit(ds), "repeat": Repeat().fit(ds), "markov": Markov().fit(ds), "item_knn": ItemKNN(neighbors=50).fit(ds)}


@pytest.fixture(scope="session")
def sasrec(ds, dev):
    from recsys.sasrec import SASRec

    cfg = replace(SASREC, dim=32, layers=1, heads=2, max_len=20, negatives=128, batch_size=64, epochs=6)
    return SASRec(cfg, device="cpu").fit(ds, eval_cases=dev.subset(slice(0, 1500)), patience=2, log_fn=None)


@pytest.fixture(scope="session")
def two_tower(ds, dev):
    from recsys.two_tower import TwoTower

    cfg = replace(TWO_TOWER, dim=32, hidden=64, max_len=20, batch_size=256, epochs=6)
    return TwoTower(cfg, device="cpu").fit(ds, eval_cases=dev.subset(slice(0, 1500)), patience=2, log_fn=None)
