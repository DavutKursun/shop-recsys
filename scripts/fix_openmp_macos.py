"""macOS: make torch, faiss-cpu and lightgbm share ONE OpenMP runtime. Each wheel ships its own libomp, and two of
them in one process abort, hang or segfault Python at the first parallel region (KMP_DUPLICATE_LIB_OK does not
help). Idempotent; rerun after a pip reinstall or upgrade of any of the three. Does nothing on Linux / Windows.

    python scripts/fix_openmp_macos.py [--check] [--no-verify]
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)  # recsys warns about the very thing this script fixes
    from recsys._openmp_macos import main

if __name__ == "__main__":
    raise SystemExit(main())
