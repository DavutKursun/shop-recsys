"""Check that the RetailRocket files are in data/raw/ and say how to get them. No credentials are read or printed.

    python scripts/download.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# sets OMP_NUM_THREADS and checks the macOS OpenMP fix before torch / faiss / lightgbm load
import recsys  # noqa: E402,F401
from recsys.config import KAGGLE_DATASET, RAW, RAW_FILES
from recsys.data import find_raw

HOW = f"""
Raw files not found in {RAW} (need {RAW_FILES['events']}, {', '.join(RAW_FILES['properties'])}, {RAW_FILES['tree']}).

Get the RetailRocket recommender dataset (≈ 300 MB zip, ≈ 1 GB unpacked) — your Kaggle login, never shared in a chat:
  1) Browser: https://www.kaggle.com/datasets/{KAGGLE_DATASET} → Download → unzip into data/raw/
  2) CLI (needs ~/.kaggle/kaggle.json): kaggle datasets download -d {KAGGLE_DATASET} -p data/raw/ --unzip
Then: python scripts/build_dataset.py
"""


def main() -> int:
    raw = find_raw(RAW)
    if raw is None:
        print(HOW)
        return 1
    for name, p in raw.items():
        paths = p if isinstance(p, list) else [p]
        for q in paths:
            print(f"{name:12s} {q.name:32s} {q.stat().st_size / 1e6:8.1f} MB")
    print("\nAll files present. Next: python scripts/build_dataset.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
