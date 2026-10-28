# Model card — shop-recsys

A two-stage next-item recommender for an e-commerce click stream: candidate retrieval from seven sources (four
classical baselines, a hybrid of two of them, a SASRec transformer and a two-tower model, the last two written from
scratch in PyTorch) and a LightGBM LambdaRank reranker over the union of their candidates. Version 0.1, 2026-10-28. Live demo and API: https://davutkursun-shop-recsys.hf.space (version 0.1.0+5d6f6d9).
Design notes with every table and the choices behind them: [DESIGN.md](DESIGN.md).

## Data and licence

- **RetailRocket recommender system dataset** (Kaggle `retailrocket/ecommerce-dataset`, licence CC BY-NC-SA 4.0):
  2,756,101 events by 1,407,580 anonymous visitors on 235,061 items, 2015-05-03 to 2015-09-18 (UTC); views 96.7%,
  add-to-carts 2.5%, transactions 0.8%. Items carry a category id and a hashed set of properties; there are no
  product names, texts or images, so the demo shows an item id, its category path and its popularity rank.
- Nothing is collected or stored by the models or the demo; the only identifiers are the dataset's numeric visitor
  and item ids. The code is MIT-licensed; the data keeps its own terms (non-commercial, share-alike).

## The split

Global time split, 14-day windows counted back from the last event. Retrieval models learn from events before
**T0 = 2015-08-07 02:59**; the reranker from the rerank window [T0, 2015-08-21); every choice was made on the dev
window [2015-08-21, 2015-09-04); the test window [2015-09-04, 2015-09-18) was read **once**, on 2026-10-28, with
`python scripts/evaluate.py --split test --final --sampled-demo`. Vocabulary: 69,784 items with at least five
training events; a target outside it is a miss for every id-based model and is counted (74.5% of test targets are
reachable). A case is one event with at least one earlier event by the same visitor; the history is that visitor's
earlier events (last 50). Metrics rank the full catalogue; confidence intervals resample visitors (500 resamples).

## Models, sizes, chosen configurations

| model | file | size | configuration (chosen on dev / the rerank window, once) |
|---|---|---|---|
| popularity | `models/popularity.joblib` | 0.3 MB | training-window counts |
| repeat | `models/repeat.joblib` | 0.3 MB | the visitor's own items, recency weight 0.8 per step |
| hybrid | `models/hybrid.joblib` | 3.3 MB | repeat first, then Markov, then popularity |
| markov | `models/markov.joblib` | 3.0 MB | P(next \| last) from training sessions, + 0.3 × P(next \| second-to-last) |
| item_knn | `models/item_knn.joblib` | 15.2 MB | session co-occurrence cosine, 100 neighbours per item, decay 0.7 |
| two_tower | `models/two_tower.pt` | 37.3 MB | dim 64, hidden 128, temperature 0.05, in-batch negatives (1,024) with the logQ correction, best epoch 5 |
| sasrec | `models/sasrec.pt` | 17.4 MB | dim 64, 2 layers, 2 heads, dropout 0.2, max_len 50, sampled softmax with 4,096 uniform negatives, best epoch 11 |
| reranked | `models/reranker.joblib` | 2.1 MB | LightGBM LambdaRank, 25 features, 146 rounds, union of every source's top-20, trained on 40,000 rerank-window cases |

The embedding models are served through exact inner-product search (FAISS flat; 0.3 ms per query over 69,784
items). HNSW indexes exist but are not served: they are not faster where they are accurate (DESIGN.md §4.1).

## Intended use

A demonstration, for a portfolio, of how a two-stage industrial recommender is built and how it should be
evaluated: time split, full-catalogue ranking, strong baselines, confidence intervals, paired tests, cold-start
rows, a single look at the test window. It is not a production system and must not be used as one: the data is
from 2015, the models were fitted once, and nothing in the serving path updates with new events.

## Results on the test window (once)

95,905 cases from 38,804 visitors. Point [95% visitor-bootstrap CI].

| model | recall@10 | ndcg@10 | mrr@10 | coverage@10 | popularity quantile |
|---|---|---|---|---|---|
| **reranked** | **0.324 [0.315, 0.334]** | 0.235 [0.228, 0.243] | 0.207 [0.201, 0.214] | 0.71 | 0.19 |
| hybrid (best baseline) | 0.295 [0.286, 0.304] | 0.206 [0.199, 0.213] | 0.177 [0.171, 0.184] | 0.60 | 0.14 |
| sasrec | 0.270 [0.261, 0.280] | 0.183 [0.176, 0.190] | 0.156 [0.150, 0.162] | 0.74 | 0.23 |
| repeat | 0.219 [0.211, 0.226] | 0.172 [0.165, 0.178] | 0.156 [0.150, 0.162] | 0.43 | 0.09 |
| two_tower | 0.209 [0.201, 0.217] | 0.135 [0.129, 0.140] | 0.112 [0.107, 0.116] | 0.75 | 0.21 |
| markov | 0.126 [0.122, 0.131] | 0.078 [0.075, 0.081] | 0.063 [0.061, 0.066] | 0.48 | 0.09 |
| item_knn | 0.125 [0.121, 0.130] | 0.071 [0.069, 0.074] | 0.055 [0.053, 0.057] | 0.61 | 0.31 |
| popularity | 0.004 [0.003, 0.005] | 0.002 [0.002, 0.002] | 0.001 [0.001, 0.002] | 0.00 | 0.00 |

Paired differences in recall@10: reranked − hybrid **+0.029 [+0.028, +0.031]**; reranked − sasrec +0.054 [+0.052,
+0.057]; sasrec − hybrid −0.025 [−0.028, −0.022]. All nine paired conclusions made on dev replicated on test with
the same sign and intervals excluding zero; every model lost about 0.03 recall@10 from dev to test, the untuned
baselines included, because the test fortnight has more targets the vocabulary never saw (DESIGN.md §5.3).

Where the models stand on the slices that matter (recall@10, test): first-click visitors (31,415 cases) reranked
0.292, SASRec 0.274, hybrid 0.253; new targets (73,595) reranked 0.140, SASRec 0.125, hybrid 0.101; repeat targets
(22,310) hybrid 0.936, reranked 0.932; purchases (635) hybrid 0.627, reranked 0.584, SASRec 0.288.

## Limitations

- **No product names.** The data has none; the product and the demo show item ids, category paths and popularity
  ranks and say so.
- **2015 data, one shop.** Nothing here transfers to another catalogue without refitting, and the dev → test drop of
  0.03 in one month of drift says a deployed version would need retraining on a schedule.
- **Out-of-vocabulary targets.** 25.5% of test targets are items with fewer than five training events; no id-based
  model can rank them and they count as misses in every number above.
- **Purchases.** The reranker optimises clicks (97% of targets) and finds 58% of purchases against the repeat
  lookup's 63%; a purchase-weighted objective is the first thing to change before a shop would use this.
- **Cold items.** The two-tower's category path makes 97% of targets reachable but finds 0.04% of the newly
  reachable ones: 355,310 items share 996 category vectors. New-item cold start needs item content this data lacks.
- **Cold visitors.** With zero clicks only the popular list exists, and it is the next item 0.4% of the time; the
  API requires at least one item.
- **Crawler tail.** 1% of cases come from visitors with more than 1,800 earlier events; histories are cut at 50 and
  intervals cluster by visitor, but the 20+ history rows are not representative of shoppers.
- **Latency.** On the Space (cpu-basic, 2 vCPUs) a reranked `/recommend` takes 34 ms server-side at the median and 54 ms at p95
  (30 calls; 11 ms on a laptop); a single model 2–5 ms. The network round trip comes on top.
- **Fallback.** A history with no vocabulary item (9.3% of dev cases) is served the last training week's popular items instead of
  the reranker; on those cases both are near zero (recall@10 0.0017 vs 0.0025, paired -0.0008 [-0.0019, +0.0003]).

## Reproduce

```bash
pip install -r requirements-dev.txt && python scripts/fix_openmp_macos.py && pytest -q
python scripts/download.py && python scripts/build_dataset.py
python scripts/train_baselines.py
python scripts/train_neural.py --model sasrec --negatives 4096
python scripts/train_neural.py --model two_tower
python scripts/build_index.py --model sasrec && python scripts/build_index.py --model two_tower
python scripts/train_reranker.py --per-source 20
python scripts/evaluate.py --split dev --sampled-demo            # the dev tables (DESIGN.md §2–4)
python scripts/evaluate.py --split test --final --sampled-demo   # already run once; do not run again
```
