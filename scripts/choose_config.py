"""Choose a neural model's configuration ONCE on the dev subsample: the default run against its --tag ablations.

Reads eval/results/train_<model>.json (the default) and eval/results/ablations/train_<model>-<tag>.json (ablations;
tags starting with "timing" are skipped), writes eval/results/<model>_choice.json with every run's config delta, best
epoch, dev recall@10 on the epoch-choice subsample and timing, and names the winner. If an ablation wins, its weights
become models/<model>.pt (the default's go to models/<model>-default.pt) and its log becomes train_<model>.json.

    python scripts/choose_config.py --model sasrec [--margin 0.0005]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# sets OMP_NUM_THREADS and checks the macOS OpenMP fix before torch / faiss / lightgbm load
import recsys  # noqa: E402,F401
from recsys.config import MODELS, RESULTS
from recsys.pipeline import MODEL_FILES


def summarise(tag: str, log: dict, default_cfg: dict) -> dict:
    tl = log["train_log"]
    epochs = tl["epochs"]
    return {"tag": tag, "config_delta": {k: v for k, v in log["config"].items() if default_cfg.get(k) != v}, "best_epoch": tl["best_epoch"],
            "dev_recall@10": tl["best_metric"], "epochs_run": len(epochs), "seconds_per_epoch": round(sum(e["seconds"] for e in epochs) / max(len(epochs), 1), 1),
            "seconds": log["seconds"], "file": log["file"], "epoch_log": epochs}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["sasrec", "two_tower", "reranker"])
    ap.add_argument("--margin", type=float, default=0.0005, help="an ablation must beat the default by more than this to be chosen")
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--ablations", type=Path, default=RESULTS / "ablations")
    ap.add_argument("--models-dir", type=Path, default=MODELS)
    a = ap.parse_args(argv)
    if a.model == "reranker":
        return choose_reranker(a)
    default_path = a.results / f"train_{a.model}.json"
    if not default_path.exists():
        print(f"{default_path} is missing; train the default first", file=sys.stderr)
        return 1
    default = json.loads(default_path.read_text(encoding="utf-8"))
    runs = [summarise("default", default, default["config"])]
    for p in sorted(a.ablations.glob(f"train_{a.model}-*.json")):
        tag = p.stem[len(f"train_{a.model}-") :]
        if tag.startswith("timing") or tag == "default":
            continue
        runs.append(summarise(tag, json.loads(p.read_text(encoding="utf-8")), default["config"]))
    best = max(runs, key=lambda r: r["dev_recall@10"])
    chosen = best if best["tag"] != "default" and best["dev_recall@10"] > runs[0]["dev_recall@10"] + a.margin else runs[0]
    rule = f"highest dev recall@10 on the epoch-choice subsample ({default['dev_cases_for_epoch_choice']:,} cases); the default keeps ties within {a.margin}"
    if chosen["tag"] != "default":
        src = a.models_dir / f"{a.model}-{chosen['tag']}.pt"
        dst = a.models_dir / MODEL_FILES[a.model]
        if dst.exists():
            shutil.copy(dst, a.models_dir / f"{a.model}-default.pt")
        shutil.copy(src, dst)
        shutil.copy(default_path, a.ablations / f"train_{a.model}-default.json")
        shutil.copy(a.ablations / f"train_{a.model}-{chosen['tag']}.json", default_path)
        print(f"{src.name} -> {dst.name} (the default's weights kept as {a.model}-default.pt)")
    out = {"model": a.model, "chosen": chosen["tag"], "rule": rule, "dev_cases_for_epoch_choice": default["dev_cases_for_epoch_choice"], "runs": runs}
    (a.results / f"{a.model}_choice.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"| run | config change | best epoch | dev recall@10 ({default['dev_cases_for_epoch_choice'] // 1000}k) | s / epoch | total s |\n|---|---|---|---|---|---|")
    for r in runs:
        delta = ", ".join(f"{k}={v}" for k, v in r["config_delta"].items()) or "—"
        mark = " **(chosen)**" if r["tag"] == chosen["tag"] else ""
        print(f"| {r['tag']}{mark} | {delta} | {r['best_epoch']} / {r['epochs_run']} | {r['dev_recall@10']:.4f} | {r['seconds_per_epoch']} | {r['seconds']:.0f} |")
    return 0


def _reranker_run(tag: str, log: dict) -> dict:
    keys = ("heldout_recall@10", "heldout_ndcg@10", "heldout_union_recall", "candidate_recall", "rows", "val_cases", "best_iteration", "val_ndcg10", "seconds")
    return {"tag": tag, "config_delta": {"candidates_per_source": log["candidates_per_source"]}, **{k: log.get(k) for k in keys}}


def choose_reranker(a) -> int:
    """The reranker is chosen on the 10% of rerank-window cases held out inside `Reranker.fit` (reranked recall@10 with
    misses counted), never on dev. The winning weights become models/reranker.joblib."""
    default_path = a.results / "train_reranker.json"
    if not default_path.exists():
        print(f"{default_path} is missing; train the default first", file=sys.stderr)
        return 1
    default = json.loads(default_path.read_text(encoding="utf-8"))
    runs = [_reranker_run("default", default)]
    for p in sorted(a.ablations.glob("train_reranker-*.json")):
        tag = p.stem[len("train_reranker-") :]
        if tag != "default":
            runs.append(_reranker_run(tag, json.loads(p.read_text(encoding="utf-8"))))
    best = max(runs, key=lambda r: r["heldout_recall@10"])
    chosen = best if best["tag"] != "default" and best["heldout_recall@10"] > runs[0]["heldout_recall@10"] + a.margin else runs[0]
    rule = f"highest reranked recall@10 on the {runs[0]['val_cases']:,} rerank-window cases held out of training (misses counted); the default keeps ties within {a.margin}"
    if chosen["tag"] != "default":
        src, dst = a.models_dir / f"reranker-{chosen['tag']}.joblib", a.models_dir / "reranker.joblib"
        if dst.exists():
            shutil.copy(dst, a.models_dir / "reranker-default.joblib")
        shutil.copy(src, dst)
        shutil.copy(default_path, a.ablations / "train_reranker-default.json")
        shutil.copy(a.ablations / f"train_reranker-{chosen['tag']}.json", default_path)
        print(f"{src.name} -> {dst.name} (the default kept as reranker-default.joblib)")
    out = {"model": "reranker", "chosen": chosen["tag"], "rule": rule, "runs": runs}
    (a.results / "rerank_choice.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("| run | candidates per source | held-out recall@10 | held-out ndcg@10 | held-out union recall | candidate recall (all) | rows | best iter | val ndcg@10 |\n|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        mark = " **(chosen)**" if r["tag"] == chosen["tag"] else ""
        print(f"| {r['tag']}{mark} | {r['config_delta']['candidates_per_source']} | {r['heldout_recall@10']:.4f} | {r['heldout_ndcg@10']:.4f} | "
              f"{r['heldout_union_recall']:.4f} | {r['candidate_recall']:.4f} | {r['rows']:,} | {r['best_iteration']} | {r['val_ndcg10']:.4f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
