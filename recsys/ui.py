"""The demo UI (Gradio, mounted at /ui): Shopper · Item · Evidence · About.

Every list comes from the same `Service` the API uses (one pass of the seven sources, then the reranker). The shoppers are
2,000 anonymous dev-window visitors (raw item ids and the window of each event; no visitor id, no timestamp). The Evidence
tab reads only committed files: the test results (read once, on 2026-10-28) under eval/results and the figures under
docs/img. Nothing typed here is stored.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

import gradio as gr
import pandas as pd
from PIL import Image

from .config import RESULTS, ROOT
from .serving import Service

IMG = ROOT / "docs" / "img"
TEST_LABEL = "test window, read once on 2026-10-28"
LISTS = (("hybrid", "Hybrid"), ("sasrec", "SASRec"), ("two_tower", "Two-tower"), ("reranked", "Reranked"), ("repeat", "Repeat"))
NAME = {"popularity": "popularity", "repeat": "repeat", "hybrid": "hybrid", "markov": "Markov", "item_knn": "item-kNN", "two_tower": "two-tower",
        "sasrec": "SASRec", "reranked": "reranked", "popularity_recent": "recent popularity"}
ABBR = {"popularity": "P", "repeat": "R", "hybrid": "H", "markov": "M", "item_knn": "K", "two_tower": "T", "sasrec": "S", "popularity_recent": "P7"}
CSS = """
.list-table th, .list-table td { font-size: 0.82rem !important; padding: 3px 6px !important; }
.list-table th { font-family: inherit !important; }
"""

ABOUT = """
## About this demo

**Data.** RetailRocket e-commerce events (Kaggle `retailrocket/ecommerce-dataset`): 2.76M views, add-to-carts and purchases by
1.41M anonymous visitors of one online shop, May to September **2015**, released under **CC BY-NC-SA 4.0** (non-commercial,
share-alike). Items carry a category id and hashed properties: **there are no product names, texts or images**, so an item is
shown as its id, its category path (category ids from the root of the shop's category tree) and its popularity rank.

**What is served.** Seven retrieval sources (popularity, repeat, the repeat-then-Markov hybrid, Markov, session item-kNN, a
two-tower model and SASRec, the last two written from scratch in PyTorch) and a LightGBM LambdaRank reranker over the union of
every source's **top-20** candidates. The models read the **50 most recent** history items; an id outside the 69,784-item
vocabulary is read as an unknown token and counted. A history with no known item gets the last training week's most
popular items, and the response says so. The embedding models are searched **exactly** (FAISS flat index, 0.3 ms per query):
the HNSW index was not faster where it was accurate at this catalogue size, so it is not served.

**How it was measured.** A global time split: the retrievers learned from events before 2015-08-07, the reranker from the
next fortnight, every choice was made on the dev fortnight, and the test fortnight was read once. Recall is over the full
catalogue, with visitor-bootstrap confidence intervals; the Evidence tab has the numbers.

**Limits.** The reranker optimises clicks and finds fewer purchases than the repeat lookup; a new product is reachable through
its category but almost never found; a quarter of test targets are outside the vocabulary and count as misses.

**The shopper shown first** is the first one (by id) whose reranked list finds a next item the hybrid misses, chosen to show the difference; *Random shopper* shows typical cases (the reranked list finds the next item for about a third of these 2,000).

**Historical data; nothing is stored.** The server logs the route, status, latency and counts of each call, never an item id.

Repository: https://github.com/DavutKursun/shop-recsys · design notes `docs/DESIGN.md` · model card `docs/model_card.md`
"""


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _img(path: Path):
    return Image.open(path) if path.exists() else None


def _ci(b: dict) -> str:
    return f"{b['point']:.3f} [{b['ci_low']:.3f}, {b['ci_high']:.3f}]"


def _test_table(test: dict) -> pd.DataFrame:
    rows = []
    for m in sorted(test.get("models", {}), key=lambda n: -test["models"][n]["recall@10"]["point"]):
        b = test["models"][m]
        rows.append({"model": NAME.get(m, m), "recall@10 [95% CI]": _ci(b["recall@10"]), "recall@20": f"{b['recall@20']['point']:.3f}",
                     "ndcg@10 [95% CI]": _ci(b["ndcg@10"]), "mrr@10": f"{b['mrr@10']['point']:.3f}", "coverage@10": f"{b['coverage@10']:.2f}",
                     "popularity quantile": f"{b['popularity_quantile@10']:.2f}"})
    return pd.DataFrame(rows)


def _paired_table(test: dict) -> pd.DataFrame:
    rows = []
    order = sorted(test.get("paired", {}), key=lambda k: (not k.startswith("reranked"), not k.startswith(("sasrec", "two_tower")), k))
    for key in order:
        d = test["paired"][key]
        a, b = key.split("_minus_")
        rows.append({"difference in recall@10": f"{NAME.get(a, a)} − {NAME.get(b, b)}", "point": f"{d['point']:+.4f}",
                     "95% CI": f"[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]", "excludes 0": "yes" if d["significant"] else "no"})
    return pd.DataFrame(rows)


def _breakdown_table(test: dict) -> pd.DataFrame:
    labels = {"warm": "warm visitors", "cold": "cold visitors", "repeat": "repeat targets", "new": "new targets", "hist_1-1": "history 1 (first click)",
              "hist_2-4": "history 2–4", "hist_5-19": "history 5–19", "hist_20-plus": "history 20+", "target_view": "target = view",
              "target_addtocart": "target = add-to-cart", "target_transaction": "target = purchase"}
    cols = [m for m in ("reranked", "hybrid", "sasrec", "two_tower", "repeat", "markov", "item_knn", "popularity") if m in test.get("models", {})]
    rows = []
    for key, label in labels.items():
        b = test.get("breakdowns", {}).get(key)
        if b:
            rows.append({"subset": label, "cases": f"{b['n_cases']:,}", **{NAME[m]: f"{b[m]:.3f}" for m in cols}})
    return pd.DataFrame(rows)


def _dev_test_table(dvt: dict) -> pd.DataFrame:
    rows = []
    for m, v in sorted(dvt.get("models", {}).items(), key=lambda kv: -kv[1]["recall@10"]["test"]):
        r = v["recall@10"]
        rows.append({"model": NAME.get(m, m), "dev recall@10": f"{r['dev']:.4f}", "test recall@10": f"{r['test']:.4f}", "change": f"{r['delta']:+.4f}"})
    return pd.DataFrame(rows)


def _ann_table(results: Path) -> pd.DataFrame:
    rows = []
    for m in ("sasrec", "two_tower"):
        r = _json(results / f"ann_{m}.json")
        if not r:
            continue
        rows.append({"model": NAME[m], "index": "exact (served)", "recall@100": "1", "ms per query, batched": f"{r['flat']['batch_ms_per_query']:.3f}",
                     "ms per query, single": f"{r['flat']['single_ms_per_query']:.3f}"})
        for ef, h in r["hnsw"].items():
            rows.append({"model": NAME[m], "index": f"HNSW ef {ef}", "recall@100": f"{h['recall_at_k']:.3f}", "ms per query, batched": f"{h['batch_ms_per_query']:.3f}",
                         "ms per query, single": f"{h['single_ms_per_query']:.3f}"})
    return pd.DataFrame(rows)


def _parse_ids(text: str) -> list[int]:
    return [int(x) for x in re.findall(r"-?\d+", text or "")][:200]


def build(service: Service, results: Path = RESULTS, img: Path = IMG) -> gr.Blocks:
    test, dvt = _json(results / "test-summary.json"), _json(results / "dev_vs_test.json")
    r10 = {m: test["models"][m]["recall@10"]["point"] for m in test.get("models", {})}
    shoppers = service.shoppers
    labels = [f"{s['id']:04d} · {s['events_before']} earlier event{'s' if s['events_before'] != 1 else ''} · next: {s['next_event']}" for s in shoppers]
    by_label = dict(zip(labels, shoppers))
    demo_id = service.manifest.get("demo_shopper")
    first = labels[demo_id] if isinstance(demo_id, int) and 0 <= demo_id < len(labels) else (labels[0] if labels else None)
    names = tuple(m for m, _ in LISTS)

    def header(m: str, title: str) -> str:
        return f"{title} · test recall@10 {r10[m]:.3f}" if m in r10 else title

    def history_frame(history: list[int], windows: list[str] | None) -> pd.DataFrame:
        rows = []
        for j, x in enumerate(history[-service.max_len :]):
            info = service.item_info(x)
            rows.append({"#": j + 1, "item": x, "category path": info["category_path"] if info else "—",
                         "popularity rank": f"{info['popularity_rank']:,}" if info else "unknown item",
                         "window": (windows[-service.max_len :][j] if windows else "typed")})
        return pd.DataFrame(rows)

    def list_frame(recs, model: str, next_item: int | None) -> pd.DataFrame:
        rows = []
        for r in recs:
            mark = " ".join(x for x in ("★ next" if next_item is not None and r.item == next_item else "", "seen" if r.seen_before else "") if x)
            row = {"#": r.rank, "item": r.item, "mark": mark}
            if model == "reranked":
                row["from"] = "".join(ABBR.get(s, s) for s in r.sources)
            else:
                row["cat"] = r.category if r.category is not None else "—"
            rows.append(row)
        return pd.DataFrame(rows, columns=["#", "item", "mark", "from" if model == "reranked" else "cat"])

    def why(lists: dict) -> str:
        hy, re_ = lists["hybrid"], lists["reranked"]
        h_items = {r.item for r in hy}
        kept = [r for r in re_ if r.item in h_items]
        new = [r for r in re_ if r.item not in h_items]
        seen_h, seen_r = sum(r.seen_before for r in hy), sum(r.seen_before for r in re_)
        if not new:
            return (f"**Why the reranked list differs from the hybrid:** here it does not — the reranker keeps the hybrid's {len(hy)} items "
                    f"({seen_r} already opened by this shopper) and only reorders them.")
        src = Counter(s for r in new for s in r.sources if s not in ("hybrid", "repeat", "popularity"))
        top = ", ".join(f"{NAME[s]} for {n}" for s, n in src.most_common(3)) or "the other sources"
        lift = f" On test this lifted recall@10 from {r10['hybrid']:.3f} to {r10['reranked']:.3f}, all of it on items the shopper had not opened." if {"hybrid", "reranked"} <= set(r10) else ""
        return (f"**Why the reranked list differs from the hybrid:** it keeps {len(kept)} of the hybrid's {len(hy)} items ({seen_r} already opened by "
                f"this shopper, against {seen_h} in the hybrid) and brings in {len(new)} the hybrid does not rank (proposed by {top}). The hybrid "
                f"is a fixed rule — seen items by recency, then Markov successors; the reranker is a LambdaRank model trained on a separate "
                f"fortnight to weigh all seven sources.{lift}")

    def show(history: list[int], windows: list[str] | None = None, shopper: dict | None = None):
        if not history:
            empty = pd.DataFrame()
            return ("*Type at least one item id, or pick a shopper.*", empty, *[empty] * len(LISTS), "")
        lists, meta = service.lists(history, 10, names)
        nxt = shopper["next_item"] if shopper else None
        if shopper:
            head = (f"**Shopper {shopper['id']:04d}** · {shopper['events_before']:,} earlier event{'s' if shopper['events_before'] != 1 else ''} "
                    f"(the models read the last {meta['history_used']}: {meta['history_used'] - sum(service.item_info(x) is None for x in history[-service.max_len:])} "
                    f"known, {sum(service.item_info(x) is None for x in history[-service.max_len:])} outside the vocabulary)")
            found = [f"{NAME[m]} (#{next(r.rank for r in lists[m] if r.item == nxt)})" for m in names if any(r.item == nxt for r in lists[m])]
            missed = [NAME[m] for m in names if not any(r.item == nxt for r in lists[m])]
            where = ("outside the vocabulary: no id-based model can rank it" if not shopper["next_in_vocabulary"]
                     else ("an item they had opened before" if shopper["next_seen_before"] else "an item they had not opened before"))
            head += (f"\n\n**What they opened next** (dev window): item **{nxt}** ({shopper['next_event']}; {where}) — ★ in the lists. "
                     f"Found in the top 10 by: {', '.join(found) or 'none of the five'}" + (f"; missed by: {', '.join(missed)}." if found and missed else "."))
        else:
            head = f"**Typed history** · {len(history)} items, the models read the last {meta['history_used']}: {meta['known_history']} known, {meta['unknown_history']} outside the vocabulary"
        if meta["fallback"]:
            head += f"\n\n⚠️ **Fallback:** {meta['note']}."
        frames = [list_frame(lists[m], m, nxt) for m, _ in LISTS]
        return head, history_frame(history, windows), *frames, ("" if meta["fallback"] else why(lists))

    def show_shopper(label: str):
        s = by_label.get(label)
        return show(s["history"], s["window"], s) if s else show([])

    def show_typed(text: str):
        return show(_parse_ids(text))

    def similar_view(item_text, k):
        ids = _parse_ids(str(item_text))
        if not ids:
            return "*Type an item id.*", pd.DataFrame()
        r = service.similar(ids[0], int(k))
        if r is None:
            return (f"Item **{ids[0]}** is not in the {service.n_items - 2:,}-item vocabulary (fewer than 5 events before the training cut-off), "
                    "so it has no embedding to search from."), pd.DataFrame()
        md = (f"**Item {r.item}** · category path {r.category_path or 'unknown'} · popularity rank {r.popularity_rank:,} of {service.n_items - 2:,}\n\n"
              f"Nearest items by cosine similarity of the SASRec item embeddings (exact search over all {service.n_items - 2:,} items, {r.latency_ms:.1f} ms). "
              "Click a row to move to that item.")
        same = r.category
        df = pd.DataFrame([{"#": x.rank, "item": x.item, "cosine": f"{x.score:.3f}", "category path": x.category_path or "unknown",
                            "same category": "yes" if same is not None and x.category == same else "", "popularity rank": f"{x.popularity_rank:,}"} for x in r.items])
        return md, df

    def open_item(evt: gr.SelectData, k):
        row = evt.row_value or []
        item = str(row[1]) if len(row) > 1 else ""
        md, df = similar_view(item, k)
        return item, md, df, gr.Tabs(selected="item")

    with gr.Blocks(title="shop-recsys", fill_width=True) as demo:
        line = (f"Test recall@10 over the full catalogue (read once): **reranked {r10['reranked']:.3f}**, hybrid {r10['hybrid']:.3f}, SASRec {r10['sasrec']:.3f}, "
                f"popularity {r10['popularity']:.3f}. " if {"reranked", "hybrid", "sasrec", "popularity"} <= set(r10) else "")
        gr.Markdown(f"# shop-recsys — what will this shopper look at or buy next?\nFive recommenders side by side for one shopper's clicks, "
                    f"RetailRocket 2015 (no product names: items are ids with a category path). {line}Version {service.version}.")
        with gr.Tabs(selected="shopper") as tabs:
            with gr.Tab("Shopper", id="shopper"):
                with gr.Row():
                    pick = gr.Dropdown(choices=labels, value=first, label="Dev shopper (2,000 anonymous visitors of the dev window; type to search)",
                                       filterable=True, scale=4, elem_id="shopper-pick")
                    rnd = gr.Button("Random shopper", scale=1)
                with gr.Row():
                    typed = gr.Textbox(label="…or type item ids, oldest first (comma or space separated)", placeholder="e.g. 461686, 5411, 257040", scale=4)
                    go = gr.Button("Recommend for these items", variant="primary", scale=1)
                summary = gr.Markdown()
                hist = gr.Dataframe(label="History the models read (oldest first, at most the last 50 events); click an item to see its neighbours",
                                    interactive=False, max_height=230, elem_id="history")
                why_md = gr.Markdown()
                with gr.Row(equal_height=False):
                    frames = [gr.Dataframe(label=header(m, t), interactive=False, elem_id=f"list-{m}", elem_classes=["list-table"], buttons=[], min_width=200,
                                           scale=4 if m == "reranked" else 3, column_widths=["14%", "31%", "27%", "28%"]) for m, t in LISTS]
                gr.Markdown("**seen** = the shopper opened this item before · **★ next** = what they actually opened next · *cat* = category id · *from* (reranked "
                            "list): the sources that proposed the item — R repeat, H hybrid, M Markov, K item-kNN, T two-tower, S SASRec, P popularity. "
                            "Click any item to open it in the Item tab.")
            with gr.Tab("Item", id="item"):
                with gr.Row():
                    item_box = gr.Textbox(label="Item id", scale=3, elem_id="item-id")
                    k_slider = gr.Slider(5, 50, value=10, step=1, label="How many neighbours", scale=3)
                    item_btn = gr.Button("Similar items", variant="primary", scale=1)
                item_md = gr.Markdown("*Type an item id, or click an item in the Shopper tab.*")
                sim = gr.Dataframe(label="Nearest items (SASRec embedding cosine, exact search)", interactive=False, elem_id="similar")
            with gr.Tab("Evidence", id="evidence"):
                n_cases = f"{test['n_cases']:,} cases from {test['n_users']:,} visitors, " if test else ""
                ceiling = (f" Candidate-union ceiling of the reranker (top-{test.get('candidates_per_source', 20)} of every source): "
                           f"{test['candidate_union_recall']:.3f}." if "candidate_union_recall" in test else "")
                gr.Markdown(f"## The test window — {n_cases}read once on 2026-10-28\nNext-item recall over the full catalogue of 69,784 items with "
                            f"visitor-bootstrap 95% intervals.{ceiling}")
                gr.Dataframe(value=_test_table(test), label=f"Every model on test ({TEST_LABEL})", interactive=False, wrap=True)
                gr.Dataframe(value=_paired_table(test), label="Paired differences in recall@10 (same visitor resamples)", interactive=False)
                gr.Image(value=_img(img / "reranked_by_history_test.png"), label=f"Reranked vs hybrid vs SASRec by history length ({TEST_LABEL})", interactive=False, show_label=False)
                with gr.Accordion("Every model by history length", open=False):
                    gr.Image(value=_img(img / "recall_by_history_test.png"), label=f"Every model by history length ({TEST_LABEL})", interactive=False, show_label=False)
                gr.Dataframe(value=_breakdown_table(test), label="Recall@10 by subset (test): visitors, repeat vs new targets, history length, event type",
                             interactive=False, wrap=True)
                gr.Markdown("### Dev → test\nEvery conclusion drawn on dev held on test with the same sign; every model lost about 0.03 recall@10 "
                            "because the later fortnight has more products the vocabulary never saw.")
                gr.Dataframe(value=_dev_test_table(dvt), label="Recall@10, dev → test", interactive=False)
                gr.Markdown("### The sampled-negatives trap\nRanked against 100 random items instead of the whole catalogue, the embedding models "
                            "seem to beat the hybrid and popularity reaches 0.32; over the full catalogue the order is the one above.")
                gr.Image(value=_img(img / "sampled_vs_full_test.png"), label=f"HR@10, full catalogue vs 100 sampled negatives ({TEST_LABEL})", interactive=False, show_label=False)
                gr.Markdown("### Exact search vs HNSW (dev queries)\nThe served index is exact: HNSW is not faster where it is accurate at 70k items.")
                gr.Dataframe(value=_ann_table(results), label="Recall@100 against exact search, latency per query", interactive=False)
                gr.Image(value=_img(img / "reranker_features.png"), label="What the reranker learned: features by gain", interactive=False, show_label=False)
            with gr.Tab("About", id="about"):
                gr.Markdown(ABOUT)

        outs = [summary, hist, *frames, why_md]
        pick.change(show_shopper, inputs=[pick], outputs=outs)
        rnd.click(lambda: random.choice(labels) if labels else None, outputs=[pick])
        go.click(show_typed, inputs=[typed], outputs=outs)
        typed.submit(show_typed, inputs=[typed], outputs=outs)
        item_btn.click(similar_view, inputs=[item_box, k_slider], outputs=[item_md, sim])
        item_box.submit(similar_view, inputs=[item_box, k_slider], outputs=[item_md, sim])
        for table in (hist, *frames, sim):
            table.select(open_item, inputs=[k_slider], outputs=[item_box, item_md, sim, tabs])
        demo.load(show_shopper, inputs=[pick], outputs=outs)
    return demo
