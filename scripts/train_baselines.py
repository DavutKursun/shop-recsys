"""Fit popularity, repeat, Markov and item-kNN on the training window and save them under models/.

    python scripts/train_baselines.py [--models popularity,repeat,markov,item_knn] [--neighbors 100]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys.baselines import BASELINES, ItemKNN
from recsys.config import MODELS, RESULTS, SPLITS
from recsys.data import Dataset
from recsys.pipeline import MODEL_FILES


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all")
    ap.add_argument("--neighbors", type=int, default=100)
    ap.add_argument("--splits-dir", type=Path, default=SPLITS)
    ap.add_argument("--models-dir", type=Path, default=MODELS)
    ap.add_argument("--out", type=Path, default=RESULTS)
    a = ap.parse_args(argv)
    names = list(BASELINES) if a.models == "all" else a.models.split(",")
    ds = Dataset.load(a.splits_dir)
    log = {}
    for name in names:
        t = time.time()
        m = ItemKNN(neighbors=a.neighbors) if name == "item_knn" else BASELINES[name]()
        m.fit(ds)
        m.save(a.models_dir / MODEL_FILES[name])
        log[name] = {"seconds": round(time.time() - t, 1)}
        print(f"{name:12s} fitted in {log[name]['seconds']}s → {a.models_dir / MODEL_FILES[name]}")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "train_baselines.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
