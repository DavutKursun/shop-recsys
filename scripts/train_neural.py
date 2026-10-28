"""Train SASRec or the two-tower model on the training window, choosing the epoch on a dev subsample.

    python scripts/train_neural.py --model sasrec   [--epochs 20 --dim 64 --layers 2 --negatives 1024 --batch-size 128]
    python scripts/train_neural.py --model two_tower [--epochs 10 --dim 64 --batch-size 1024 --no-logq]
Common: --device cpu|mps|cuda  --dev-cases 20000  --patience 3  --fast (tiny settings for smoke runs)  --tag name
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# sets OMP_NUM_THREADS and checks the macOS OpenMP fix before torch / faiss / lightgbm load
import recsys  # noqa: E402,F401
from recsys.config import MODELS, RESULTS, SASREC, SPLITS, TWO_TOWER
from recsys.data import Dataset
from recsys.pipeline import MODEL_FILES


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["sasrec", "two_tower"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--heads", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--negatives", type=int, default=None, help="SASRec: sampled negatives per batch (0 = full softmax)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--no-logq", action="store_true", help="two-tower: drop the logQ correction (ablation)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dev-cases", type=int, default=20_000, help="dev cases used to pick the epoch (0 = run all epochs)")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--tag", default=None, help="save as models/<model>-<tag>.pt instead of the default file (ablations)")
    ap.add_argument("--splits-dir", type=Path, default=SPLITS)
    ap.add_argument("--models-dir", type=Path, default=MODELS)
    ap.add_argument("--out", type=Path, default=RESULTS)
    a = ap.parse_args(argv)

    ds = Dataset.load(a.splits_dir)
    base = SASREC if a.model == "sasrec" else TWO_TOWER
    over = {k: v for k, v in dict(epochs=a.epochs, dim=a.dim, layers=a.layers, heads=a.heads, dropout=a.dropout, negatives=a.negatives, batch_size=a.batch_size,
                                  lr=a.lr, max_len=a.max_len, temperature=a.temperature).items() if v is not None and hasattr(base, k)}
    if a.no_logq:
        over["logq_correction"] = False
    if a.fast:
        fast = dict(epochs=3, dim=32, layers=1, heads=2) if a.model == "sasrec" else dict(epochs=3, dim=32, hidden=64)
        over = {**fast, **over}
        over = {k: v for k, v in over.items() if hasattr(base, k)}
    cfg = replace(base, **over)
    if a.model == "sasrec":
        from recsys.sasrec import SASRec

        model = SASRec(cfg, device=a.device)
    else:
        from recsys.two_tower import TwoTower

        model = TwoTower(cfg, device=a.device)
    dev = ds.cases("dev", max_cases=a.dev_cases, seed=1) if a.dev_cases > 0 else None
    print(f"{a.model}: {cfg}")
    t = time.time()
    model.fit(ds, eval_cases=dev, patience=a.patience)
    name = f"{a.model}-{a.tag}.pt" if a.tag else MODEL_FILES[a.model]
    model.save(a.models_dir / name)
    log = {"model": a.model, "config": cfg.__dict__, "train_log": model.log.as_dict(), "seconds": round(time.time() - t, 1), "file": name, "dev_cases_for_epoch_choice": a.dev_cases}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"train_{name.replace('.pt', '')}.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"saved {a.models_dir / name}; best epoch {model.log.best_epoch} (dev recall@10 {model.log.best_metric:.4f}) in {log['seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
