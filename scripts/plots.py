"""Figures for DESIGN.md → docs/img/: training curves of SASRec and the two-tower model, recall@10 by history bucket
for every model, and the sampled-negatives demonstration (HR@10 vs 100 sampled negatives next to full-catalogue HR@10).

    python scripts/plots.py [--results eval/results] [--split dev] [--out docs/img]

Static light-mode PNGs. Colours follow a validated categorical palette assigned to the models in a fixed order (never
cycled); the tables in DESIGN.md are the table view for the light hues that sit below 3:1 contrast.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys.config import RESULTS, SOURCES  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]  # fixed order, light mode
COLOR = {name: SLOTS[i] for i, name in enumerate(SOURCES)}  # popularity, repeat, hybrid, markov, item_knn, two_tower, sasrec
COLOR["reranked"] = SLOTS[7]
LABEL = {"popularity": "popularity", "repeat": "repeat", "hybrid": "hybrid", "markov": "Markov", "item_knn": "item-kNN", "two_tower": "two-tower",
         "sasrec": "SASRec", "reranked": "reranked"}
BUCKETS = [("hist_1-1", "1"), ("hist_2-4", "2–4"), ("hist_5-19", "5–19"), ("hist_20-plus", "20+")]

plt.rcParams.update({"font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": AXIS, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "axes.titlecolor": INK, "legend.frameon": False})


def chrome(ax, ylabel: str | None = None, ymax: float | None = None) -> None:
    """Recessive chrome: hairline solid gridlines, one baseline, no box."""
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.grid(axis="y", color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, labelsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2)
    if ymax is not None:
        ax.set_ylim(0, ymax)


def rounded_bar(ax, x: float, height: float, width: float, color: str, radius_px: float = 4.0) -> None:
    """A column that grows from the baseline with 4px rounded data-end corners and a square base."""
    fig = ax.figure
    inv = ax.transData.inverted()
    x0, y0 = inv.transform((0, 0))
    x1, y1 = inv.transform((radius_px, radius_px))
    rx, ry = min(abs(x1 - x0), width / 2), min(abs(y1 - y0), max(height, 1e-9) / 2)
    left, right, top = x - width / 2, x + width / 2, height
    verts = [(left, 0), (right, 0), (right, top - ry), (right, top), (right - rx, top), (left + rx, top), (left, top), (left, top - ry), (left, 0)]
    codes = [MPath.MOVETO, MPath.LINETO, MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, edgecolor="none", linewidth=0))
    del fig


def training_curves(results: Path, out: Path) -> Path | None:
    logs = {}
    for name in ("sasrec", "two_tower"):
        p = results / f"train_{name}.json"
        if p.exists():
            logs[name] = json.loads(p.read_text(encoding="utf-8"))
    if not logs:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), dpi=160)
    for ax, key, title in zip(axes, ("loss", "metric"), ("Training loss per epoch", "Dev recall@10 per epoch (epoch-choice subsample)")):
        for name, log in logs.items():
            ep = log["train_log"]["epochs"]
            xs = [e["epoch"] for e in ep]
            ys = [e[key] for e in ep]
            if all(y is None for y in ys):
                continue
            ax.plot(xs, ys, color=COLOR[name], linewidth=2, solid_capstyle="round", solid_joinstyle="round", label=LABEL[name])
            best = log["train_log"]["best_epoch"]
            if key == "metric" and 1 <= best <= len(ys):
                ax.plot([best], [ys[best - 1]], marker="o", markersize=8, color=COLOR[name], markeredgecolor=SURFACE, markeredgewidth=2)
                ax.annotate(f"best {ys[best - 1]:.3f} @ {best}", (best, ys[best - 1]), textcoords="offset points", xytext=(6, 6), fontsize=8, color=INK2)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("epoch", color=INK2)
        chrome(ax)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.legend(loc="best", fontsize=9, labelcolor=INK2)
    axes[1].set_ylim(top=axes[1].get_ylim()[1] * 1.06)  # headroom for the best-epoch labels
    neg = logs.get("sasrec", {}).get("config", {}).get("negatives", "?")
    fig.text(0.01, -0.04, f"Losses are not comparable across models: SASRec's is a sampled softmax over {neg:,} uniform negatives, "
             "the two-tower's an in-batch softmax with the logQ correction.", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    path = out / "training_curves.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def recall_by_history(summary: dict, out: Path, split: str) -> Path:
    models = [m for m in list(SOURCES) + ["reranked"] if m in summary["models"]]
    fig, ax = plt.subplots(figsize=(10, 3.8), dpi=160)
    n = len(models)
    group, gap = 0.84, 0.03  # a 2px-ish surface gap between touching bars
    width = group / n
    best_per_bucket = []
    for b, (key, _) in enumerate(BUCKETS):
        row = summary["breakdowns"].get(key, {})
        vals = {m: row.get(m, 0.0) for m in models}
        best_per_bucket.append(max(vals, key=vals.get))
        for i, m in enumerate(models):
            x = b - group / 2 + width * (i + 0.5)
            rounded_bar(ax, x, vals[m], width - gap, COLOR[m])
            if m == best_per_bucket[-1]:  # selective direct labels: the best bar of each bucket only
                ax.annotate(f"{vals[m]:.3f}", (x, vals[m]), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8, color=INK2)
    ax.set_xticks(range(len(BUCKETS)))
    ax.set_xticklabels([f"{lab}\n(n={summary['breakdowns'][k]['n_cases']:,})" for k, lab in BUCKETS])
    ax.set_xlim(-0.6, len(BUCKETS) - 0.4)
    ymax = max(summary["breakdowns"][k][m] for k, _ in BUCKETS for m in models) * 1.12
    chrome(ax, "recall@10", ymax)
    ax.set_xlabel("history length before the target (events)", color=INK2)
    ax.set_title(f"Recall@10 by history length — {split} window, {summary['n_cases']:,} cases", fontsize=10, loc="left")
    handles = [plt.Line2D([], [], marker="s", linestyle="none", markersize=8, color=COLOR[m], label=LABEL[m]) for m in models]
    ax.legend(handles=handles, ncol=n, loc="upper center", bbox_to_anchor=(0.5, -0.3), fontsize=8.5, labelcolor=INK2)  # below the axis: never over a bar
    fig.tight_layout()
    path = out / "recall_by_history.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def sampled_vs_full(summary: dict, out: Path, split: str) -> Path | None:
    demo = summary.get("sampled_negatives_demo")
    if not demo:
        return None
    models = sorted(demo["models"], key=lambda m: -demo["models"][m]["full_hr@10"])
    fig, ax = plt.subplots(figsize=(10, 3.8), dpi=160)
    width, gap = 0.22, 0.03
    for i, m in enumerate(models):
        full, samp = demo["models"][m]["full_hr@10"], demo["models"][m]["sampled_hr@10"]
        rounded_bar(ax, i - width / 2, full, width - gap, SLOTS[0])
        rounded_bar(ax, i + width / 2, samp, width - gap, SLOTS[1])
        ax.annotate(f"{full:.3f}", (i - width / 2, full), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8, color=INK2)
        ax.annotate(f"{samp:.3f}", (i + width / 2, samp), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8, color=INK2)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([LABEL[m] for m in models])
    ax.set_xlim(-0.6, len(models) - 0.4)
    chrome(ax, "HR@10", max(v["sampled_hr@10"] for v in demo["models"].values()) * 1.18)
    ax.set_title(f"The sampled-negatives trap — first {demo['n_cases']:,} {split} cases, models ordered by their full-catalogue HR@10", fontsize=10, loc="left")
    handles = [plt.Line2D([], [], marker="s", linestyle="none", markersize=8, color=SLOTS[0], label="full catalogue (69,784 items)"),
               plt.Line2D([], [], marker="s", linestyle="none", markersize=8, color=SLOTS[1], label=f"target vs {demo['n_negatives']} uniformly sampled negatives")]
    ax.legend(handles=handles, loc="upper right", fontsize=8.5, labelcolor=INK2)
    fig.tight_layout()
    path = out / "sampled_vs_full.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def ann_chart(results: Path, out: Path) -> Path | None:
    reports = {}
    for name in ("sasrec", "two_tower"):
        p = results / f"ann_{name}.json"
        if p.exists():
            reports[name] = json.loads(p.read_text(encoding="utf-8"))
    if not reports:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), dpi=160)
    efs = sorted({int(ef) for r in reports.values() for ef in r["hnsw"]})
    for name, r in reports.items():
        rec = [r["hnsw"][str(ef)]["recall_at_k"] for ef in efs]
        axes[0].plot(range(len(efs)), rec, color=COLOR[name], linewidth=2, marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, label=LABEL[name])
        axes[0].annotate(f"{rec[-1]:.3f}", (len(efs) - 1, rec[-1]), textcoords="offset points", xytext=(6, -3), fontsize=8, color=INK2)
        lat = [r["flat"]["single_ms_per_query"]] + [r["hnsw"][str(ef)]["single_ms_per_query"] for ef in efs]
        axes[1].plot(range(len(lat)), lat, color=COLOR[name], linewidth=2, marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, label=LABEL[name])
        axes[1].annotate(f"{lat[0]:.2f} ms", (0, lat[0]), textcoords="offset points", xytext=(6, 2), fontsize=8, color=INK2)
    axes[0].axhline(0.95, color=AXIS, linewidth=1)
    axes[0].annotate("0.95 target", (0, 0.95), textcoords="offset points", xytext=(2, 3), fontsize=8, color=MUTED)
    axes[0].set_xticks(range(len(efs)))
    axes[0].set_xticklabels([f"ef {ef}" for ef in efs])
    axes[0].set_title(f"HNSW recall@{next(iter(reports.values()))['k']} against exact search", fontsize=10, loc="left")
    chrome(axes[0], "recall@100")
    axes[0].set_ylim(min(0.6, axes[0].get_ylim()[0]), 1.0)
    axes[1].set_xticks(range(len(efs) + 1))
    axes[1].set_xticklabels(["exact"] + [f"ef {ef}" for ef in efs])
    axes[1].set_title("Latency per single query (median, ms)", fontsize=10, loc="left")
    chrome(axes[1], "ms")
    axes[1].set_ylim(0, axes[1].get_ylim()[1] * 1.15)
    for ax in axes:
        ax.legend(loc="lower right", fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    path = out / "ann_recall_latency.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def feature_importance_chart(results: Path, out: Path, top: int = 15) -> Path | None:
    p = results / "train_reranker.json"
    if not p.exists():
        return None
    info = json.loads(p.read_text(encoding="utf-8"))
    imp = sorted(info["feature_importance"], key=lambda r: -r["gain"])
    total = sum(r["gain"] for r in imp) or 1.0
    rows = imp[:top][::-1]
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(rows) + 1.2), dpi=160)
    for i, r in enumerate(rows):
        share = r["gain"] / total
        ax.barh(i, share, height=0.62, color=COLOR["reranked"], linewidth=0)
        ax.annotate(f"{share:.1%}", (share, i), textcoords="offset points", xytext=(4, 0), va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["feature"] for r in rows], fontsize=9)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0)
    ax.set_xlim(0, max(r["gain"] / total for r in rows) * 1.18)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("share of total split gain", color=INK2)
    ax.set_title(f"Reranker: top {len(rows)} of {len(imp)} features by gain (LightGBM LambdaRank, {info['best_iteration']} rounds)", fontsize=10, loc="left")
    fig.tight_layout()
    path = out / "reranker_features.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def reranked_by_history(summary: dict, out: Path, split: str) -> Path | None:
    models = [m for m in ("hybrid", "sasrec", "reranked") if m in summary["models"]]
    if "reranked" not in models:
        return None
    fig, ax = plt.subplots(figsize=(10, 3.6), dpi=160)
    group, gap = 0.6, 0.03
    width = group / len(models)
    for b, (key, _) in enumerate(BUCKETS):
        row = summary["breakdowns"][key]
        for i, m in enumerate(models):
            x = b - group / 2 + width * (i + 0.5)
            rounded_bar(ax, x, row[m], width - gap, COLOR[m])
            ax.annotate(f"{row[m]:.3f}", (x, row[m]), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8, color=INK2)
    ax.set_xticks(range(len(BUCKETS)))
    ax.set_xticklabels([f"{lab}\n(n={summary['breakdowns'][k]['n_cases']:,})" for k, lab in BUCKETS])
    ax.set_xlim(-0.6, len(BUCKETS) - 0.4)
    chrome(ax, "recall@10", max(summary["breakdowns"][k][m] for k, _ in BUCKETS for m in models) * 1.15)
    ax.set_xlabel("history length before the target (events)", color=INK2)
    ax.set_title(f"Reranked vs the hybrid bar vs SASRec by history length — {split} window", fontsize=10, loc="left")
    handles = [plt.Line2D([], [], marker="s", linestyle="none", markersize=8, color=COLOR[m], label=LABEL[m]) for m in models]
    ax.legend(handles=handles, ncol=len(models), loc="upper center", bbox_to_anchor=(0.5, -0.3), fontsize=8.5, labelcolor=INK2)
    fig.tight_layout()
    path = out / "reranked_by_history.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "img")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    made = [training_curves(a.results, a.out), ann_chart(a.results, a.out), feature_importance_chart(a.results, a.out)]
    summary_path = a.results / f"{a.split}-summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        made += [recall_by_history(summary, a.out, a.split), sampled_vs_full(summary, a.out, a.split), reranked_by_history(summary, a.out, a.split)]
    for p in made:
        if p is not None:
            print(f"wrote {p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
