# Design notes

Filled milestone by milestone; each section says what was measured, on which window, and what it changes.

## 1. The data and the split (M1)
Raw facts vs the published ones, vocabulary rule and the share of events it covers, the four windows, sessions,
history-length distribution, repeat share, warm/cold share, what "a case" is.

## 2. Baselines on dev (M1)
Popularity, repeat, Markov, item-kNN: the table with CIs, breakdowns, why repeat is strong on this data.

## 3. SASRec and the two-tower model (M2)
Losses, hyperparameters chosen once on dev, training curves, ablations (loss, max_len, logQ, event weights),
the sampled-negatives demonstration.

## 4. Retrieval and reranking (M3)
FAISS exact vs HNSW (recall, latency), candidate-union ceiling, reranker features and importance, the
cold-start paths (new users, new items via the item tower), what the reranker fixes and what it cannot.

## 5. Test results (M4) — run once on _(date)_

## 6. What I would do differently
