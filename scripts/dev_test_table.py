"""Dev vs test for every model and metric: the two point estimates with CIs, the dev → test change, which paired
conclusions survived (interval excludes 0 on both windows) and the tuning-optimism number (the dev → test change of
the models whose configuration was chosen on dev). Reads eval/results/dev-summary.json and test-summary.json.

    python scripts/dev_test_table.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# sets OMP_NUM_THREADS and checks the macOS OpenMP fix before torch / faiss / lightgbm load
import recsys  # noqa: E402,F401
from recsys.config import RESULTS

METRICS = ("recall@10", "recall@20", "ndcg@10", "mrr@10")
EXTRA = ("coverage@10", "popularity_quantile@10")
TUNED = ("reranked", "sasrec", "two_tower")  # configurations chosen on dev or the rerank window


def ci(block: dict) -> str:
    return f"{block['point']:.4f} [{block['ci_low']:.4f}, {block['ci_high']:.4f}]"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=RESULTS)
    a = ap.parse_args(argv)
    dev = json.loads((a.results / "dev-summary.json").read_text(encoding="utf-8"))
    test = json.loads((a.results / "test-summary.json").read_text(encoding="utf-8"))
    models = [m for m in test["models"] if m in dev["models"]]
    order = sorted(models, key=lambda m: -test["models"][m]["recall@10"]["point"])
    lines = [f"# dev → test — dev: {dev['n_cases']:,} cases / {dev['n_users']:,} visitors; test: {test['n_cases']:,} cases / {test['n_users']:,} visitors "
             f"(test run {test['run_date']}, once)", "",
             f"Reachable targets dev {dev['share_target_reachable']:.1%} → test {test['share_target_reachable']:.1%}; repeat targets {dev['share_repeat']:.1%} → "
             f"{test['share_repeat']:.1%}; warm visitors {dev['share_warm']:.1%} → {test['share_warm']:.1%}", ""]
    out = {"dev_cases": dev["n_cases"], "test_cases": test["n_cases"], "models": {}, "paired": {}, "tuning_optimism": {}}
    for metric in METRICS:
        lines += [f"## {metric}", "", "| model | dev | test | test − dev |", "|---|---|---|---|"]
        for m in order:
            d, t = dev["models"][m][metric], test["models"][m][metric]
            out["models"].setdefault(m, {})[metric] = {"dev": d["point"], "test": t["point"], "delta": round(t["point"] - d["point"], 4)}
            lines.append(f"| {m} | {ci(d)} | {ci(t)} | {t['point'] - d['point']:+.4f} |")
        lines.append("")
    lines += ["## coverage@10 and popularity quantile", "", "| model | coverage dev | coverage test | pop. quantile dev | pop. quantile test |", "|---|---|---|---|---|"]
    for m in order:
        d, t = dev["models"][m], test["models"][m]
        out["models"][m]["coverage@10"] = {"dev": d["coverage@10"], "test": t["coverage@10"]}
        out["models"][m]["popularity_quantile@10"] = {"dev": d["popularity_quantile@10"], "test": t["popularity_quantile@10"]}
        lines.append(f"| {m} | {d['coverage@10']:.3f} | {t['coverage@10']:.3f} | {d['popularity_quantile@10']:.3f} | {t['popularity_quantile@10']:.3f} |")
    if "candidate_union_recall" in dev and "candidate_union_recall" in test:
        lines += ["", f"Candidate-union ceiling (K = {test.get('candidates_per_source')}): dev {dev['candidate_union_recall']:.4f} → test {test['candidate_union_recall']:.4f}"]
        out["candidate_union_recall"] = {"dev": dev["candidate_union_recall"], "test": test["candidate_union_recall"]}
    lines += ["", "## Paired differences in recall@10 (visitor bootstrap): did the conclusion survive?", "",
              "| comparison | dev | test | survived (CI excludes 0 on both, same sign) |", "|---|---|---|---|"]
    for key in test["paired"]:
        if key not in dev["paired"]:
            continue
        d, t = dev["paired"][key], test["paired"][key]
        same_sign = (d["point"] > 0) == (t["point"] > 0)
        survived = bool(d["significant"] and t["significant"] and same_sign)
        out["paired"][key] = {"dev": d["point"], "test": t["point"], "dev_ci": [d["ci_low"], d["ci_high"]], "test_ci": [t["ci_low"], t["ci_high"]], "survived": survived}
        lines.append(f"| {key.replace('_minus_', ' − ')} | {d['point']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}] | "
                     f"{t['point']:+.4f} [{t['ci_low']:+.4f}, {t['ci_high']:+.4f}] | {'yes' if survived else 'NO'} |")
    lines += ["", "## Tuning optimism: dev → test change in recall@10 of the models chosen on dev / the rerank window", ""]
    for m in TUNED:
        if m in out["models"]:
            delta = out["models"][m]["recall@10"]["delta"]
            out["tuning_optimism"][m] = delta
            lines.append(f"- {m}: {dev['models'][m]['recall@10']['point']:.4f} → {test['models'][m]['recall@10']['point']:.4f} ({delta:+.4f})")
    untuned = [m for m in order if m not in TUNED]
    if untuned:
        mean_untuned = sum(out["models"][m]["recall@10"]["delta"] for m in untuned) / len(untuned)
        out["tuning_optimism"]["mean_untuned_baselines"] = round(mean_untuned, 4)
        lines.append(f"- for reference, the untuned baselines move {mean_untuned:+.4f} on average (window drift, not tuning)")
    lines += ["", "## Breakdowns, recall@10: dev → test", "", "| subset | " + " | ".join(order) + " |", "|---|" + "---|" * len(order)]
    for label in test["breakdowns"]:
        if label in dev["breakdowns"]:
            cells = [f"{dev['breakdowns'][label][m]:.3f} → {test['breakdowns'][label][m]:.3f}" for m in order]
            lines.append(f"| {label} (n {dev['breakdowns'][label]['n_cases']:,} → {test['breakdowns'][label]['n_cases']:,}) | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    (a.results / "dev_vs_test.md").write_text(text, encoding="utf-8")
    (a.results / "dev_vs_test.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
