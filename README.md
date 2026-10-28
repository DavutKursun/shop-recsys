# shop-recsys

**Recommends what a shopper will look at or buy next, from clicks alone — and measures it the way a shop would feel it.**

_(demo GIF and live link arrive in Milestone 5)_

## The problem
A shop with hundreds of thousands of products cannot show them all. The products a shopper sees next decide
whether they buy. Recommenders are easy to build and easy to evaluate wrongly: random splits leak the future,
sampled metrics flatter weak models, and popularity alone is a hard baseline nobody reports.

## What this does
- Two retrieval models written from scratch in PyTorch: a sequential transformer (SASRec) and a two-tower model
  with in-batch negatives and the logQ correction; candidates served through a FAISS index
- A LightGBM LambdaRank reranker over the union of candidates from six sources, with user–item and item features
- Four classical baselines that must be beaten: popularity, repeat, first-order Markov, session item-kNN
- Evaluation on a global time split, over the full catalogue, with user-bootstrap confidence intervals, paired
  differences, cold-start and repeat breakdowns, and a demonstration of the sampled-negatives trap

## Results
_(filled at Milestone 4 — test window, used once)_

| Metric (next item, full catalogue) | Best baseline | SASRec | Reranked pipeline |
|---|---|---|---|
| Recall@10 (95% CI) | | | |
| NDCG@10 | | | |
| Paired difference vs best baseline, Recall@10 | | | |
| Coverage@10 / popularity bias | | | |

## Data
RetailRocket e-commerce events (2.76M views, add-to-carts and purchases by 1.4M visitors over 4.5 months;
item categories from the property files). Global time split: train → 14-day rerank window → 14-day dev → 14-day test.

## Run it yourself
```bash
pip install -r requirements-dev.txt && pytest -q      # no data needed
python scripts/download.py                            # where to put the raw files
make dataset baselines sasrec two-tower index reranker eval-dev
```

## Stack
Python · PyTorch · FAISS · LightGBM · pandas/pyarrow · FastAPI · Gradio

## License
MIT (code). The RetailRocket dataset has its own terms on Kaggle.
