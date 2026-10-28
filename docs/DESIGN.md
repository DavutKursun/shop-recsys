# Design notes

Filled milestone by milestone; each section says what was measured, on which window, and what it changes.

## 1. The data and the split (M1)

Everything here comes from `python scripts/build_dataset.py` (`data/splits/raw_facts.json`, `data/splits/summary.json`).
Build time from the four raw CSVs: 11.8 s, peak resident memory 0.81 GB (macOS, Python 3.11, pandas 3.1); 5.9 s from
the parquet cache. The two property files (20.3M rows, 893 MB) are streamed in 2M-row chunks and only their
`categoryid` rows are kept, latest value per item.

### 1.1 Raw facts against the dataset page

| Quantity | Dataset page | Measured |
|---|---|---|
| Events | ≈ 2.76M: ≈ 2.66M views, ≈ 69k add-to-carts, ≈ 22k transactions | 2,756,101: 2,664,312 views (96.7%), 69,332 add-to-carts (2.5%), 22,457 transactions (0.8%) |
| Visitors / items in events | ≈ 1.41M / ≈ 235k | 1,407,580 / 235,061 |
| Period (UTC) | 2015-05-03 → 2015-09-18 | 2015-05-03 03:00 → 2015-09-18 02:59 (138 days) |
| Category tree | ≈ 1.7k categories | 1,669 rows; 417,053 items carry a `categoryid` (many never appear in events), 1,180 distinct categories among them, 940 among vocabulary items |
| Behaviour | most visitors have a single event; heavy repeat viewing | 76.5% of training-window visitors have one event; 25.5% of dev targets are items the visitor already opened (§1.5) |

### 1.2 Cleaning, sessions, windows

- A view that repeats the previous event's item for the same visitor within 30 minutes (a refresh or a back-click) is
  dropped: 2,756,101 → 2,475,306 events (−10.2%). A later re-visit of the same item is kept; it is a real next-item
  target in e-commerce.
- A gap longer than 30 minutes starts a new session: 1,763,140 sessions over all windows. Markov transitions and
  item-kNN co-occurrence use training-window sessions only.
- Global time split, 14-day windows counted back from the last event (UTC):

| Window | From | To | Events | Visitors | Items | Events whose item is in the vocabulary |
|---|---|---|---|---|---|---|
| train | 2015-05-03 03:00 | 2015-08-07 02:59 (96 days) | 1,809,085 | 1,023,737 | 204,788 | 86.7% |
| rerank | 2015-08-07 02:59 | 2015-08-21 02:59 | 221,590 | 142,514 | 73,312 | 79.7% |
| dev | 2015-08-21 02:59 | 2015-09-04 02:59 | 221,473 | 141,764 | 72,827 | 77.6% |
| test | 2015-09-04 02:59 | 2015-09-18 02:59 | opened once, in M4 | | | |

### 1.3 Vocabulary

Items with at least 5 events before T0: **69,784**, which is 34.1% of the 204,788 items seen in training but 86.7% of
training events. Index = popularity rank (0 = PAD, 1 = UNK). The share of events the vocabulary covers falls to 79.7%
in the rerank window and 77.6% in dev as new products arrive; a target outside the vocabulary is a miss for every
id-based model and is counted, never dropped ("reachable" below). The cold-item path through the category (M3) is
the only way to reach those.

### 1.4 Training sequences

240,351 visitors have at least two training events (23.5% of the 1,023,737 training visitors) and contribute
1,025,699 events (56.7% of the training window). Sequence length: median 2, p95 10, 0.3% longer than the model
history of 50. Single-event visitors still count: their one item feeds popularity, sessions and the item vocabulary.

### 1.5 What a case is, and what the cases look like

A case is one event in the window whose visitor has at least one earlier event anywhere; the history is that
visitor's earlier events (last 50), the target is the event's item. Predictions use only events before that moment,
including earlier events inside the same window. *Warm* = the visitor had an event before the window; a *cold*
visitor was first seen inside it and still has ≥ 1 in-window event as history. *Repeat* = the target is in the
history. Metrics are per case; confidence intervals resample visitors, so a visitor with 40 cases is one cluster.

| | rerank window | dev window |
|---|---|---|
| cases (events with ≥ 1 earlier event) | 92,536 of 221,590 events (41.8%) | 93,937 of 221,473 events (42.4%) |
| visitors behind them | 36,527 | 38,012 |
| history length p25 / p50 / p75 / p90 / p95 / p99 | 1 / 3 / 9 / 62 / 399 / 2,753 | 1 / 3 / 9 / 78 / 520 / 1,808 |
| history buckets 1 / 2–4 / 5–19 / 20+ | 32.1% / 29.6% / 21.6% / 16.8% | 32.9% / 29.9% / 20.4% / 16.8% |
| warm visitors | 38.3% | 39.1% |
| repeat targets | 26.0% | 25.5% |
| reachable targets (in the vocabulary) | 81.9% | 79.2% |
| repeat share among reachable targets | 31.8% | 32.2% |
| target type view / add-to-cart / transaction | 90,032 / 1,778 / 726 | 91,480 / 1,715 / 742 |

Two things shape everything that follows. A third of the cases have a single-item history, so "the last item" is all a
model gets. And the tail is heavy: 1% of cases come from visitors with more than 1,800 earlier events, which look like
crawlers rather than shoppers; truncating the history to 50 and clustering the bootstrap by visitor keeps them from
dominating either the models or the intervals.

## 2. Baselines on dev (M1)

`python scripts/train_baselines.py` (popularity, repeat, Markov, item-kNN fitted on the training window in 15 s) and
`python scripts/evaluate.py --split dev --no-rerank --sampled-demo` (93,937 dev cases from 38,012 visitors, full
catalogue of 69,784 items, 500 visitor-bootstrap resamples; 6.5 min on CPU, peak 1.3 GB). Results:
`eval/results/dev-summary.md` / `.json`. Point estimate [95% CI].

| model | recall@10 | recall@20 | ndcg@10 | mrr@10 | coverage@10 | pop. quantile |
|---|---|---|---|---|---|---|
| repeat | 0.2416 [0.2336, 0.2493] | 0.2481 [0.2404, 0.2556] | 0.1906 [0.1835, 0.1976] | 0.1736 [0.1666, 0.1802] | 0.434 | 0.083 |
| markov | 0.1396 [0.1355, 0.1435] | 0.1653 [0.1607, 0.1698] | 0.0859 [0.0833, 0.0885] | 0.0693 [0.0669, 0.0714] | 0.488 | 0.097 |
| item_knn | 0.1364 [0.1320, 0.1408] | 0.1790 [0.1733, 0.1846] | 0.0782 [0.0758, 0.0807] | 0.0603 [0.0583, 0.0622] | 0.609 | 0.315 |
| popularity | 0.0030 [0.0024, 0.0036] | 0.0072 [0.0063, 0.0082] | 0.0014 [0.0011, 0.0017] | 0.0009 [0.0007, 0.0011] | 0.000 | 0.000 |

Paired visitor-bootstrap differences in recall@10 against the best baseline (repeat); every interval excludes 0:

| comparison | difference [95% CI] |
|---|---|
| markov − repeat | −0.1020 [−0.1090, −0.0945] |
| item_knn − repeat | −0.1052 [−0.1124, −0.0977] |
| popularity − repeat | −0.2386 [−0.2464, −0.2306] |

Recall@10 by subset (point estimates; n = cases):

| subset | n | popularity | repeat | markov | item_knn |
|---|---|---|---|---|---|
| warm visitors | 36,736 | 0.0033 | 0.2300 | 0.0817 | 0.0791 |
| cold visitors | 57,201 | 0.0028 | 0.2490 | 0.1768 | 0.1733 |
| repeat targets | 23,929 | 0.0076 | 0.9448 | 0.1563 | 0.1423 |
| new targets | 70,008 | 0.0015 | 0.0012 | 0.1339 | 0.1345 |
| history 1 | 30,878 | 0.0028 | 0.1436 | 0.1411 | 0.1383 |
| history 2–4 | 28,114 | 0.0038 | 0.2820 | 0.1669 | 0.1587 |
| history 5–19 | 19,158 | 0.0028 | 0.3508 | 0.1397 | 0.1412 |
| history 20+ | 15,787 | 0.0024 | 0.2289 | 0.0880 | 0.0875 |
| target = view | 91,480 | 0.0030 | 0.2380 | 0.1398 | 0.1366 |
| target = add-to-cart | 1,715 | 0.0035 | 0.2315 | 0.1539 | 0.1574 |
| target = transaction | 742 | 0.0054 | 0.7089 | 0.0863 | 0.0741 |

**Repeat wins, and the breakdown says why.** A quarter of the dev targets (25.5%) are items the visitor already
opened, and on those the repeat baseline scores 0.945: 0.255 × 0.945 ≈ 0.241, which is its entire recall. On new
targets it is 0.001 (it can only offer seen items; the popularity prior fills the list), while Markov and item-kNN
reach 0.134. The sequential baselines barely see the repeat signal (0.156 and 0.142 on repeat targets): session
transitions rarely point back to the same item once refreshes are removed, and training sequences are short (median
2). The business slice is the clearest case: repeat finds 70.9% of purchases in the top 10 (so at least that share of
purchases is of an item already in the history) and Markov 8.6%. With a single-item history the three non-trivial baselines tie at ≈ 0.14:
the one item is both the repeat candidate and the Markov context. Heavy visitors (20+ events) are hardest for
everyone, Markov and item-kNN dropping to 0.09; many of them are the crawler-like tail of §1.5.

**Popularity is not a bar here.** With 69,784 vocabulary items and a flat head, the ten most viewed products are
the next item 0.3% of the time (coverage@10 is those ten items; popularity quantile 0.000). "Beats popularity" means
nothing on this catalogue; the bars that matter are repeat (0.242) and the sequential pair (≈ 0.14).

**Markov vs item-kNN.** Markov is ahead at @10 (0.140 vs 0.136, a gap inside the intervals) and item-kNN at @20
(0.179 vs 0.165); item-kNN recommends from a wider and less popular slice of the catalogue (coverage 0.609 vs 0.488,
popularity quantile 0.315 vs 0.097), which is what the reranker will later be able to exploit.

**What this sets up.** Repeat and the sequential baselines are complementary: repeat covers 25.5% of cases at 0.945,
the sequential models cover the other 74.5% at 0.134, so a perfect choice between them would reach ≈ 0.34 recall@10.
That is the room for SASRec and the two-tower model (M2), which score seen and unseen items in one list, and for the
reranker (M3), whose `seen_count` and `last_seen_pos` features encode exactly this split.

**The sampled-negatives trap, first look.** On the first 5,000 dev cases, HR@10 against 100 uniformly sampled
negatives (the protocol of many papers) next to the full-catalogue HR@10 on the same cases:

| model | full-catalogue HR@10 | HR@10 vs 100 sampled negatives |
|---|---|---|
| item_knn | 0.132 | 0.489 |
| repeat | 0.249 | 0.474 |
| markov | 0.132 | 0.406 |
| popularity | 0.001 | 0.335 |

Sampling reorders the models (item-kNN overtakes repeat) and inflates popularity 280×, because a target only has to
beat 100 random tail items instead of 69,784. §3 repeats the demonstration with the learned models.

## 3. SASRec and the two-tower model (M2)
Losses, hyperparameters chosen once on dev, training curves, ablations (loss, max_len, logQ, event weights),
the sampled-negatives demonstration.

## 4. Retrieval and reranking (M3)
FAISS exact vs HNSW (recall, latency), candidate-union ceiling, reranker features and importance, the
cold-start paths (new users, new items via the item tower), what the reranker fixes and what it cannot.

## 5. Test results (M4) — run once on _(date)_

## 6. What I would do differently
