---
title: shop-recsys
emoji: 🛒
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
license: mit
pinned: false
short_description: Next item for a shopper — SASRec, two-tower, LambdaRank
---

# shop-recsys — what will this shopper look at or buy next?

A two-stage recommender on the RetailRocket click stream: seven retrieval sources (popularity, repeat, a repeat-then-Markov
hybrid, Markov, session item-kNN, and a two-tower model and SASRec written from scratch in PyTorch) and a LightGBM LambdaRank
reranker over the union of their top-20 candidates. Tested once, on a fortnight it never saw (95,905 cases, full catalogue
of 69,784 items): recall@10 **0.324 [0.315, 0.334]** for the reranked list against 0.295 for the hybrid baseline, a paired
difference of +0.029 [+0.028, +0.031]; popularity scores 0.004.

Open **/ui** for the demo (pick one of 2,000 anonymous dev shoppers and see five recommenders side by side) and **/docs** for
the API: `POST /recommend`, `GET /similar/{item}`, `GET /health`.

The code is MIT-licensed; **the data is RetailRocket's, CC BY-NC-SA 4.0 (non-commercial, share-alike)**. Items have no names
(the shop published ids, categories and hashed properties), so an item is shown as its id, category path and popularity
rank. Historical data from 2015, not a live shop; nothing you type is stored. Repository:
https://github.com/DavutKursun/shop-recsys
