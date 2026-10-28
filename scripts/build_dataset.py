"""Raw files → cached parquet → vocabulary, windows, sessions → data/splits/ (parquet gitignored, summary.json committed).

    python scripts/build_dataset.py [--raw data/raw] [--days 14] [--min-item-events 5] [--force]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# sets OMP_NUM_THREADS and checks the macOS OpenMP fix before torch / faiss / lightgbm load
import recsys  # noqa: E402,F401
from recsys.config import CACHE, RAW, SPLITS, WINDOWS, WindowRules
from recsys.data import Dataset, cache_raw, find_raw


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--out", type=Path, default=SPLITS)
    ap.add_argument("--days", type=int, default=WINDOWS.days)
    ap.add_argument("--min-item-events", type=int, default=WINDOWS.min_item_events)
    ap.add_argument("--max-len", type=int, default=WINDOWS.max_len)
    ap.add_argument("--force", action="store_true", help="rebuild the parquet cache")
    a = ap.parse_args(argv)
    if a.force or not (a.cache / "events.parquet").exists():
        raw = find_raw(a.raw)
        if raw is None:
            print(f"no raw files in {a.raw}; run scripts/download.py", file=sys.stderr)
            return 1
        facts = cache_raw(raw, a.cache)
        (a.cache / "raw_facts.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
        print("raw facts:", json.dumps(facts, indent=2))
    events = pd.read_parquet(a.cache / "events.parquet")
    categories = pd.read_parquet(a.cache / "categories.parquet")
    rules = WindowRules(days=a.days, min_item_events=a.min_item_events, max_len=a.max_len)
    ds = Dataset.build(events, categories, rules)
    ds.save(a.out)
    summary = ds.summary()
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
