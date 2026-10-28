"""shop-recsys: what will this shopper look at or buy next — retrieval, ranking, honest evaluation, serving."""

import os as _os
import sys as _sys

# One laptop-sized thread pool for every OpenMP user (torch CPU kernels, faiss, LightGBM); override it in the shell.
# It has to be set before torch / faiss / lightgbm are imported, which is why every script imports recsys first.
_os.environ.setdefault("OMP_NUM_THREADS", "4")

if _sys.platform == "darwin":
    # torch, faiss-cpu and lightgbm each ship a libomp; two in one process crash Python (see _openmp_macos.py).
    from ._openmp_macos import warn_if_two_runtimes as _warn

    _warn()

__version__ = "0.1.0"
