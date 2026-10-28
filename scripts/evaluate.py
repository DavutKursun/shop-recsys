"""Every saved model (and the reranked pipeline) on one window: full-catalogue metrics with user-bootstrap CIs,
coverage, popularity bias, breakdowns, paired differences. `--split test` needs `--final` and is run ONCE.

    python scripts/evaluate.py --split dev [--models all] [--no-rerank] [--max-cases 0] [--resamples 500] [--sampled-demo]
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys.config import BOOTSTRAP_RESAMPLES, MODELS, PRIMARY_K, RERANK, RESULTS, SOURCES, SPLITS
from recsys.data import Dataset
from recsys.metrics import sampled_ranks
from recsys.pipeline import available_models, dump_json, evaluate_models, load_model, results_table
from recsys.rerank import Reranker


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["rerank", "dev", "test"])
    ap.add_argument("--final", action="store_true", help="required for --split test; the one and only look")
    ap.add_argument("--models", default="all")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--max-cases", type=int, default=0, help="0 = every case in the window")
    ap.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    ap.add_argument("--chunk", type=int, default=20_000)
    ap.add_argument("--sampled-demo", action="store_true", help="also report HR@10 against 100 sampled negatives (the paper trap)")
    ap.add_argument("--tag", default="", help="suffix for the result files (ablations)")
    ap.add_argument("--splits-dir", type=Path, default=SPLITS)
    ap.add_argument("--models-dir", type=Path, default=MODELS)
    ap.add_argument("--out", type=Path, default=RESULTS)
    a = ap.parse_args(argv)
    if a.split == "test" and not a.final:
        print("refusing to touch the test split without --final (it is used once; see CLAUDE.md §2)", file=sys.stderr)
        return 3
    ds = Dataset.load(a.splits_dir)
    names = available_models(a.models_dir) if a.models == "all" else a.models.split(",")
    names = [n for n in SOURCES if n in names]
    models = {n: load_model(n, a.models_dir) for n in names}
    reranker = None
    if not a.no_rerank and (a.models_dir / "reranker.joblib").exists():
        reranker = Reranker.load(a.models_dir / "reranker.joblib")
        missing = [s for s in reranker.sources if s not in models]
        if missing:
            print(f"reranker needs sources {missing}; evaluating without it", file=sys.stderr)
            reranker = None
    cases = ds.cases(a.split, max_cases=a.max_cases or None, seed=3)
    print(f"{a.split}: {len(cases):,} cases from {len(np.unique(cases.user)):,} users; models {names}; reranker {'yes' if reranker else 'no'}")
    res = evaluate_models(cases, models, reranker, ds.n_items, per_source=RERANK.candidates_per_source, chunk=a.chunk, resamples=a.resamples)
    res.update({"split": a.split, "run_date": str(date.today()), "models_evaluated": names + (["reranked"] if reranker else [])})
    if a.sampled_demo:
        res["sampled_negatives_demo"] = sampled_demo(cases, models)
    tag = f"-{a.tag}" if a.tag else ""
    dump_json(res, a.out / f"{a.split}-summary{tag}.json")
    md = [f"# {a.split} results — {len(cases):,} cases, {res['n_users']:,} users, run {res['run_date']}", "",
          f"Targets reachable (in vocabulary): {res['share_target_reachable']:.1%} · repeat targets: {res['share_repeat']:.1%} · warm users: {res['share_warm']:.1%}", "",
          results_table(res), ""]
    if "candidate_union_recall" in res:
        md.append(f"Candidate union recall (reranker ceiling, {RERANK.candidates_per_source} per source): {res['candidate_union_recall']:.4f}\n")
    md.append("## Paired differences, recall@10 (user bootstrap)\n")
    for k, v in res["paired"].items():
        md.append(f"- {k}: {v['point']:+.4f} [{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]{' *' if v['significant'] else ''}")
    md.append("\n## Breakdowns, recall@10\n")
    for label, row in res["breakdowns"].items():
        md.append(f"- {label} (n={row['n_cases']:,}): " + ", ".join(f"{n} {row[n]:.4f}" for n in res["models_evaluated"]))
    (a.out / f"{a.split}-summary{tag}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(results_table(res))
    for k, v in res["paired"].items():
        print(f"{k}: {v['point']:+.4f} [{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]")
    return 0


def sampled_demo(cases, models: dict, n_cases: int = 5000, n_neg: int = 100) -> dict:
    """HR@10 with 100 sampled negatives next to the full-ranking HR@10, on the same cases: the two can disagree."""
    from recsys.metrics import ranks_in_topk

    sub = cases.subset(np.arange(min(n_cases, len(cases))))
    out = {"n_cases": len(sub), "n_negatives": n_neg, "models": {}}
    for n, m in models.items():
        full, samp = [], []
        for start in range(0, len(sub), 512):
            h, t = sub.history[start : start + 512], sub.target[start : start + 512]
            s = m.score_batch(h)
            idx = np.argsort(-s, axis=1)[:, :PRIMARY_K]
            full.append(ranks_in_topk(idx, t) > 0)
            samp.append(sampled_ranks(s, t, n_neg, seed=start) <= PRIMARY_K)
        out["models"][n] = {"full_hr@10": round(float(np.concatenate(full).mean()), 4), "sampled_hr@10": round(float(np.concatenate(samp).mean()), 4)}
    return out


if __name__ == "__main__":
    raise SystemExit(main())
