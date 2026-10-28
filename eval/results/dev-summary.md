# dev results — 93,937 cases, 38,012 users, run 2026-10-28

Targets reachable (in vocabulary): 79.2% · repeat targets: 25.5% · warm users: 39.1%

| model | recall@10 | recall@20 | ndcg@10 | mrr@10 | coverage@10 | pop. quantile |
|---|---|---|---|---|---|---|
| repeat | 0.2416 [0.2336, 0.2493] | 0.2481 [0.2404, 0.2556] | 0.1906 [0.1835, 0.1976] | 0.1736 [0.1666, 0.1802] | 0.434 | 0.083 |
| markov | 0.1396 [0.1355, 0.1435] | 0.1653 [0.1607, 0.1698] | 0.0859 [0.0833, 0.0885] | 0.0693 [0.0669, 0.0714] | 0.488 | 0.097 |
| item_knn | 0.1364 [0.1320, 0.1408] | 0.1790 [0.1733, 0.1846] | 0.0782 [0.0758, 0.0807] | 0.0603 [0.0583, 0.0622] | 0.609 | 0.315 |
| popularity | 0.0030 [0.0024, 0.0036] | 0.0072 [0.0063, 0.0082] | 0.0014 [0.0011, 0.0017] | 0.0009 [0.0007, 0.0011] | 0.000 | 0.000 |

## Paired differences, recall@10 (user bootstrap)

- popularity_minus_repeat: -0.2386 [-0.2464, -0.2306] *
- markov_minus_repeat: -0.1020 [-0.1090, -0.0945] *
- item_knn_minus_repeat: -0.1052 [-0.1124, -0.0977] *

## Breakdowns, recall@10

- warm (n=36,736): popularity 0.0033, repeat 0.2300, markov 0.0817, item_knn 0.0791
- cold (n=57,201): popularity 0.0028, repeat 0.2490, markov 0.1768, item_knn 0.1733
- repeat (n=23,929): popularity 0.0076, repeat 0.9448, markov 0.1563, item_knn 0.1423
- new (n=70,008): popularity 0.0015, repeat 0.0012, markov 0.1339, item_knn 0.1345
- hist_1-1 (n=30,878): popularity 0.0028, repeat 0.1436, markov 0.1411, item_knn 0.1383
- hist_2-4 (n=28,114): popularity 0.0038, repeat 0.2820, markov 0.1669, item_knn 0.1587
- hist_5-19 (n=19,158): popularity 0.0028, repeat 0.3508, markov 0.1397, item_knn 0.1412
- hist_20-plus (n=15,787): popularity 0.0024, repeat 0.2289, markov 0.0880, item_knn 0.0875
- target_view (n=91,480): popularity 0.0030, repeat 0.2380, markov 0.1398, item_knn 0.1366
- target_addtocart (n=1,715): popularity 0.0035, repeat 0.2315, markov 0.1539, item_knn 0.1574
- target_transaction (n=742): popularity 0.0054, repeat 0.7089, markov 0.0863, item_knn 0.0741
