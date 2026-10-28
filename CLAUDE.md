# shop-recsys — what will this shopper look at or buy next?

This file is the build spec. Read it fully before writing code. Work milestone by
milestone; after each milestone stop, show how to run it, and wait for Davut's OK.
Do not spawn review or refuter sub-agents; verify with the scripts and tests and
report directly. Never print any token or environment variable value. Section 0 says where we are.

## 0. Status (updated 2026-10-28)

**Milestone 4 is done: the test window was opened ONCE, on 2026-10-28, with `python scripts/evaluate.py --split test --final
--sampled-demo` (plus `python scripts/cold_items.py --split test --final` for the cold-item line). Test recall@10: reranked
0.324 [0.315, 0.334] vs hybrid 0.295 (paired +0.029 [+0.028, +0.031]) and SASRec 0.270 (−0.025 vs hybrid); all nine paired dev
conclusions replicated; every model dropped ≈ 0.03 dev → test (window drift: reachable targets 79.2% → 74.5%), the reranker's
margin over the hybrid shrank by 0.004. DESIGN.md §5–6, docs/model_card.md, README results. Never run on test again.
Start at Milestone 5: serve and ship.**

Implemented in the starter (`recsys/`):
- `config.py` (paths, window rules, model configs, seeds), `schema.py` (RecommendRequest / RecommendResponse, Pydantic)
- `data.py`: chunked loaders for `events.csv` and the two property files (only `categoryid` rows are kept, latest
  value per item wins), `category_tree.csv`; consecutive-duplicate removal within a session gap; sessions (30-min
  gap); **global time windows** T0 < T1 < T2 < end (train / rerank / dev / test, 14 days each); the item
  vocabulary from events before T0 (index 0 = PAD, 1 = UNK, then items ordered by training count, so index − 1 is
  the popularity rank); the item table (counts, recent counts, first/last seen, category and parent as dense
  indices); `train_sequences()`, `train_sessions()`, and `cases(window)` = one evaluation case per event in the
  window given everything that user did before it (left-padded history, target, user, event type, history length,
  warm / repeat flags)
- `synthetic.py`: a fake shop with known structure (favourite categories, item successors, repeats, sessions over
  60 days) written in the RetailRocket file layout, so the real loader is what the tests exercise
- `baselines.py`: popularity (all-time or recent), repeat (recency-weighted history), first-order Markov from
  training sessions, session item-kNN (cosine over session co-occurrence, top-100 neighbours per item)
- `sasrec.py`: SASRec from scratch (causal self-attention blocks, position embeddings, tied output embeddings,
  sampled-softmax loss with uniform negatives or full softmax), early stopping on dev recall@10
- `two_tower.py`: user tower (mean-pooled history + position-from-end + last item) and item tower (id + category
  + parent) with in-batch negatives, temperature and the logQ correction; `embed_new_items()` for items outside
  the id vocabulary (category only — the cold-start path)
- `retrieval.py`: FAISS flat (exact) and HNSW indexes over item vectors; `ann_check()` = recall of HNSW against
  exact search and latency per query for both
- `rerank.py`: candidate union of every source's top-K with per-source score/rank features, item statistics,
  user–item signals (seen count, last seen position, category share, same category as last item), history length;
  LightGBM LambdaRank with early stopping on held-out cases; `rerank()` streams in chunks
- `metrics.py`: full-catalogue ranks, recall / NDCG / MRR @K per case, **user-cluster bootstrap** CIs, paired
  user-bootstrap differences, coverage, popularity quantile, `sampled_ranks` (the 100-negatives trap, for the demo)
- `pipeline.py`: load models, `evaluate_models()` (everything above plus breakdowns by warm/cold, repeat/new,
  history-length bucket and target type, candidate-union recall), the markdown table
- scripts: `download.py`, `build_dataset.py`, `train_baselines.py`, `train_neural.py --model sasrec|two_tower`,
  `build_index.py`, `train_reranker.py`, `evaluate.py` (refuses `--split test` without `--final`), `synthetic_check.py`
- NOT yet: `app.py` (FastAPI + Gradio), Dockerfile / Space, DESIGN.md content, model card, GIF, plots.

No LLM anywhere; no API key; CPU (or Apple MPS) is enough. Raw files (≈ 1 GB) and parquet are gitignored.

**Milestone 1, step 0 — done (venv, 22 tests, synthetic check: reranked beats every single model, learned models ≥ 9×
popularity; data fetched with the Kaggle CLI). For the record:** create the venv, `pip install torch` (on macOS this is the CPU/MPS build),
`pip install -r requirements-dev.txt`, `pytest -q` (expect 22 passed, ~1 min). Then `python scripts/synthetic_check.py`
and report its table: on the synthetic shop the reranked pipeline must beat every single model and every learned
model must beat popularity by a wide margin. Then ask Davut to download the RetailRocket files into `data/raw/`
(§3.1) and run `python scripts/download.py`. Stop.

macOS notes: Python 3.11 venv; 8 GB RAM — never load the property files whole (the loader streams them);
`pip install torch` picks the right wheel; use `--device mps` on Apple Silicon if it is faster than cpu (try both on
one epoch); `brew install libomp` if LightGBM complains. After every `pip install`, run `python scripts/fix_openmp_macos.py` once: torch, faiss-cpu and
lightgbm each bundle a libomp, and two in one process abort, hang or segfault Python (`KMP_DUPLICATE_LIB_OK` does not help);
the script makes all three load torch's copy.

## 1. Goal

An online shop wants to show each visitor the products they are most likely to open or buy next, from their
click history alone. This project builds the standard two-stage industrial recommender — candidate retrieval
with learned embeddings, then a feature-rich reranker — writes the two retrieval models from scratch in PyTorch,
and measures everything the way a shop would feel it: on a time split, over the whole catalogue, against
baselines that are embarrassingly strong, with confidence intervals and cold-start numbers.

The README answers: *How much better than "show the popular items" (and than "show what they just looked at")
is a learned recommender here, with a confidence interval?* *How much does reranking add on top of the best
retriever?* *What happens for a first-time visitor and for a brand-new product?* *How fast is one request?*

Portfolio purpose: deep learning from scratch (attention, sampled softmax, in-batch negatives, logQ), embeddings
and approximate nearest neighbours, learning-to-rank, and the evaluation pitfalls that most recsys portfolios fall
into (random splits, sampled metrics, no popularity baseline, no CIs). Interviewers ask about exactly these.

## 2. Non-negotiables

- **Time split, never random.** Models learn only from events before T0; the reranker from [T0, T1); choices on
  dev [T1, T2); test [T2, end) once. A case is predicted with the user's events before that moment only.
- **Full-catalogue ranking.** Every metric ranks all vocabulary items. Targets outside the vocabulary count as
  misses and their share is reported. Sampled metrics appear once, as a demonstration of why not (§5).
- **One look at the test window.** `scripts/evaluate.py --split test` needs `--final`, runs once in M4, and the
  date goes into this file. Done on 2026-10-28: `python scripts/evaluate.py --split test --final --sampled-demo` and
  `python scripts/cold_items.py --split test --final`, each once; results in `eval/results/test-summary.*` and
  `cold_items-test.json`. Not to be run again.
- **The baselines stay in every table.** Popularity, repeat, Markov and item-kNN are reported wherever a learned
  model is; a learned model that does not beat them is reported as such.
- **Uncertainty is a number.** User-cluster bootstrap CIs (≥ 500 resamples) on every metric; paired
  user-bootstrap differences for every "A beats B" sentence; cases are never treated as independent.
- **Ground truth where it exists.** Every model runs on the synthetic shop first (`scripts/synthetic_check.py`);
  a model that fails there is not trusted on real data.
- **Cold start is measured, not mentioned.** Warm vs cold users and vocabulary vs new-item targets are separate
  rows in every results file.
- No secrets exist in this project; Kaggle credentials stay in `~/.kaggle/`.

## 3. Data

### 3.1 Source

RetailRocket recommender system dataset (Kaggle `retailrocket/ecommerce-dataset`, ≈ 300 MB zip): `events.csv`
(timestamp ms, visitorid, event ∈ {view, addtocart, transaction}, itemid, transactionid),
`item_properties_part1.csv` + `part2.csv` (≈ 20M rows; the `categoryid` property is the only one used),
`category_tree.csv` (categoryid, parentid). Davut downloads it once and unzips into `data/raw/`;
`scripts/download.py` checks the files and prints both routes. Items have no names: the product shows an item
id, its category path and its popularity rank, and says so.

### 3.2 Expected facts (from the dataset page; verify in M1)

| Quantity | Expected |
|---|---|
| Events | ≈ 2.76M: ≈ 2.66M views, ≈ 69k add-to-carts, ≈ 22k transactions |
| Visitors / items | ≈ 1.41M / ≈ 235k |
| Period | 2015-05-03 → 2015-09-18 (≈ 4.5 months) |
| Category tree | ≈ 1.7k categories |
| Behaviour | most visitors have a single event; heavy repeat viewing |

### 3.3 Windows, vocabulary, cases (`recsys/data.py`, defaults in `WindowRules`)

- Windows: T0 = end − 42 days, T1 = end − 28 days, T2 = end − 14 days. Retrieval models train on events before
  T0 (≈ 96 days); the reranker trains on [T0, T1); dev = [T1, T2); test = [T2, end).
- Vocabulary: items with ≥ 5 events before T0. Everything else is UNK: an input token (its embedding is trained)
  but never a target; a case whose target is UNK is a miss for every id-based model (share reported).
- Sessions: a gap > 30 minutes starts a new session; Markov transitions and item-kNN co-occurrence are
  session-based. Consecutive repeats of the same item within a session are dropped; later re-visits are kept.
- A case = one event in the window whose user has ≥ 1 earlier event (any window); history = that user's earlier
  events (last 50); `warm` = the user had an event before the window; `repeat` = the target is in the history.
  Metrics are per case; CIs resample users.
- Summary numbers to report in M1: window sizes, vocabulary size and share of events it covers per window, users
  with ≥ 2 training events, history length quantiles, repeat share, warm share, share of targets reachable.

## 4. Models (one interface: `fit(ds)`, `score_batch(history)`, `topk(history, k)`)

| Name | Idea | Why it is here |
|---|---|---|
| popularity | training-window counts (or the last 7 days) | the baseline most recommenders secretly lose to |
| repeat | the user's own recent items, recency-weighted | e-commerce shoppers come back to the same product |
| markov | P(next \| last item) from session transitions, plus the second-to-last | the strongest cheap sequential signal |
| item_knn | session co-occurrence cosine, top-100 neighbours, recency-weighted sum | the classic that still wins papers |
| two_tower | user tower ⟂ item tower, in-batch negatives, logQ correction, FAISS retrieval | industrial retrieval; cold items via the category |
| sasrec | causal transformer over the sequence, next-item at every position, sampled softmax | the sequential deep model everyone asks about |
| reranked | LambdaRank over the union of all sources' top-50 | stage two; the product |

Hyperparameters (dim, layers, negatives, lr, epochs, temperature, max_len) are chosen once on dev recall@10
for `sasrec` and `two_tower`; the epoch is chosen by early stopping on a 20k-case dev subsample.

## 5. Evaluation (`recsys/metrics.py`, `pipeline.py`, `scripts/evaluate.py`)

- Recall@10 / @20, NDCG@10, MRR@10 over the full catalogue, per case, with user-bootstrap 95% CIs.
- Coverage@10 (share of the vocabulary ever recommended) and the mean popularity quantile of recommendations
  (0 = always the top item): the popularity-bias line.
- Breakdowns of recall@10: warm vs cold users, repeat vs new targets, history length 1 / 2–4 / 5–19 / 20+,
  target type (view / add-to-cart / transaction — the business line is the last two).
- Paired user-bootstrap differences: every model vs the best baseline; reranked vs the best single model.
- Candidate-union recall (the reranker's ceiling) next to the reranked numbers.
- The sampled-negatives demonstration (`--sampled-demo`): HR@10 against 100 random negatives next to full-ranking
  HR@10 for every model on the same cases; show where the ordering changes (Krichene & Rendle, KDD 2020).
- ANN honesty (`scripts/build_index.py`): HNSW recall@100 vs exact and latency per query at ef 32/64/128.
- Synthetic ground truth (`scripts/synthetic_check.py`): the full pipeline on the fake shop.
DESIGN.md: data facts, the dev tables with CIs, training curves, ablations (SASRec negatives 0 vs 1024, max_len
20 vs 50, two-tower with/without logQ, event-weighted training), the ANN table, reranker feature importance, the
cold-start section, the sampled-metrics demonstration, five worked shoppers.

## 6. Serving (M5)

- FastAPI: `POST /recommend` with `{history: [raw item ids], k, model}` → ranked items with score, category,
  popularity rank, seen-before flag and (for `reranked`) the sources that proposed them; `GET /similar/{item}`
  (nearest items by embedding); `GET /health`. Default model `reranked`; others selectable.
- Gradio at /ui: tab "Shopper" (pick one of 2,000 committed dev shoppers or type item ids; the history and the
  lists of popularity / Markov / SASRec / two-tower / reranked side by side, with the dev recall of each model);
  tab "Item" (similar items and the category path); tab "Evidence" (the test table with CIs, the breakdowns, the
  sampled-vs-full chart, the ANN table, feature importance); tab "About" (data, limits: no product names, 2015 data,
  nothing stored).
- Hugging Face Space, Docker SDK, cpu-basic; the SASRec and two-tower weights, the HNSW index and the reranker are
  committed under `models/` (a few MB each); latency target p50 < 50 ms per request for retrieve + rerank.

## 7. Repository layout

```
shop-recsys/
  app.py                       # FastAPI + Gradio (M5)
  recsys/ config.py, schema.py, data.py, synthetic.py, base.py, baselines.py, sasrec.py, two_tower.py,
          torch_utils.py, retrieval.py, rerank.py, metrics.py, pipeline.py
  scripts/ download.py, build_dataset.py, train_baselines.py, train_neural.py, build_index.py, train_reranker.py,
           evaluate.py, synthetic_check.py, (M2+) plots.py, ablations.py, (M5) bench_latency.py, demo_gif.py
  data/raw/ data/cache/ (gitignored)  data/splits/ (parquet gitignored, summary.json committed)
  models/ (weights gitignored until M5), eval/results/, tests/, docs/DESIGN.md, docs/model_card.md, assets/demo.gif,
  README.md, requirements*.txt, Makefile, LICENSE, .github/workflows/ci.yml
```

## 8. Milestones

**M1 — data and baselines.** Step 0; raw facts vs §3.2; `build_dataset.py` (report §3.3 numbers and RAM/time);
`train_baselines.py`; `evaluate.py --split dev --no-rerank` for the four baselines with CIs and breakdowns;
DESIGN.md §1–2. Say which baseline wins and why.
**M2 — the PyTorch models.** SASRec and two-tower on the training window; one epoch timed on cpu and mps; the
dev table with CIs; hyperparameters chosen once on dev; the four ablations; training curves; the sampled-negatives
demonstration; DESIGN.md §3.
**M3 — retrieval and reranking.** `build_index.py` for both models (ANN table); `train_reranker.py`; the dev table
with the reranked row, candidate-union ceiling, feature importance; cold-start: new users (cold rows), new items
through the two-tower item tower (report how many dev targets become reachable and what recall they get); DESIGN.md §4.
**M4 — final.** Test once for every table; DESIGN.md §5; model card; README results.
**M5 — serve and ship.** FastAPI + Gradio, latency, Space, GIF, README.
**M6 — optional.** BERT4Rec-style masked training; GRU4Rec; H&M data with product names and images (an image
tower); online bandit re-ranking; multi-objective (view vs purchase) ranking.

## 9. Acceptance criteria

- Synthetic shop: every learned model beats popularity on recall@10 by ≥ 3×; reranked beats every single model.
- Dev and test: the best learned model beats the best baseline on recall@10 with a paired user-bootstrap 95% CI
  that excludes 0 — or the README says it does not, with the CI.
- Reranked beats the best single model with a paired CI excluding 0 — or the README says so.
- The sampled-negatives demonstration is in DESIGN.md with at least one pair of models whose order flips or whose
  gap changes by more than its CI.
- HNSW recall@100 ≥ 0.95 against exact search at the served ef; both latencies reported.
- Cold rows (cold users, new-item targets) are in the test table with their own numbers.
- `/recommend` p50 < 50 ms on the Space CPU; every response carries the sources and the seen-before flag.
- `pytest` green without the raw data; CI green; README on one screen before setup.

## 10. Out of scope

Product text/images (not in the data), real-time feature updates, multi-objective optimisation, explore/exploit.
Listed as next steps.

## 11. Pitfalls to avoid

- Any random split, leave-one-out split, or filtering that uses events after T0 to decide what is in the training set.
- Evaluating with sampled negatives or on the vocabulary only (targets outside it exist and are reported as misses).
- Comparing models on different cases (always the same `cases(window)` object for every model).
- Treating cases as independent in CIs: a heavy user's 40 cases are one cluster.
- Letting the reranker see scores from retrievers trained on the same events it trains on.
- Loading the property files whole (≈ 1 GB CSV) — stream them.
- Reading the test window for anything before M4, including "just to check the size".
