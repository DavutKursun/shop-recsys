# dev results — 93,937 cases, 38,012 users, run 2026-10-28

Targets reachable (in vocabulary): 79.2% · repeat targets: 25.5% · warm users: 39.1%

| model | recall@10 | recall@20 | ndcg@10 | mrr@10 | coverage@10 | pop. quantile |
|---|---|---|---|---|---|---|
| hybrid | 0.3254 [0.3149, 0.3347] | 0.3520 [0.3407, 0.3619] | 0.2279 [0.2199, 0.2355] | 0.1965 [0.1890, 0.2035] | 0.602 | 0.141 |
| sasrec | 0.3018 [0.2912, 0.3115] | 0.3521 [0.3410, 0.3626] | 0.2050 [0.1971, 0.2128] | 0.1748 [0.1676, 0.1816] | 0.740 | 0.222 |
| repeat | 0.2416 [0.2336, 0.2493] | 0.2481 [0.2404, 0.2556] | 0.1906 [0.1835, 0.1976] | 0.1736 [0.1666, 0.1802] | 0.434 | 0.083 |
| two_tower | 0.2336 [0.2255, 0.2416] | 0.2816 [0.2720, 0.2907] | 0.1514 [0.1458, 0.1573] | 0.1259 [0.1210, 0.1312] | 0.751 | 0.211 |
| markov | 0.1396 [0.1355, 0.1435] | 0.1653 [0.1607, 0.1698] | 0.0859 [0.0833, 0.0885] | 0.0693 [0.0669, 0.0714] | 0.488 | 0.097 |
| item_knn | 0.1364 [0.1320, 0.1408] | 0.1790 [0.1733, 0.1846] | 0.0782 [0.0758, 0.0807] | 0.0603 [0.0583, 0.0622] | 0.609 | 0.315 |
| popularity | 0.0030 [0.0024, 0.0036] | 0.0072 [0.0063, 0.0082] | 0.0014 [0.0011, 0.0017] | 0.0009 [0.0007, 0.0011] | 0.000 | 0.000 |

## Paired differences, recall@10 (user bootstrap)

- popularity_minus_hybrid: -0.3223 [-0.3318, -0.3119] *
- repeat_minus_hybrid: -0.0838 [-0.0870, -0.0800] *
- markov_minus_hybrid: -0.1857 [-0.1937, -0.1764] *
- item_knn_minus_hybrid: -0.1889 [-0.1975, -0.1798] *
- two_tower_minus_hybrid: -0.0918 [-0.0960, -0.0870] *
- sasrec_minus_hybrid: -0.0235 [-0.0271, -0.0196] *

## Breakdowns, recall@10

- warm (n=36,736): popularity 0.0033, repeat 0.2300, hybrid 0.2577, markov 0.0817, item_knn 0.0791, two_tower 0.1798, sasrec 0.2226
- cold (n=57,201): popularity 0.0028, repeat 0.2490, hybrid 0.3688, markov 0.1768, item_knn 0.1733, two_tower 0.2681, sasrec 0.3527
- repeat (n=23,929): popularity 0.0076, repeat 0.9448, hybrid 0.9449, markov 0.1563, item_knn 0.1423, two_tower 0.5536, sasrec 0.7661
- new (n=70,008): popularity 0.0015, repeat 0.0012, hybrid 0.1136, markov 0.1339, item_knn 0.1345, two_tower 0.1242, sasrec 0.1431
- hist_1-1 (n=30,878): popularity 0.0028, repeat 0.1436, hybrid 0.2801, markov 0.1411, item_knn 0.1383, two_tower 0.2421, sasrec 0.3052
- hist_2-4 (n=28,114): popularity 0.0038, repeat 0.2820, hybrid 0.3806, markov 0.1669, item_knn 0.1587, two_tower 0.2688, sasrec 0.3620
- hist_5-19 (n=19,158): popularity 0.0028, repeat 0.3508, hybrid 0.3960, markov 0.1397, item_knn 0.1412, two_tower 0.2427, sasrec 0.3149
- hist_20-plus (n=15,787): popularity 0.0024, repeat 0.2289, hybrid 0.2297, markov 0.0880, item_knn 0.0875, two_tower 0.1432, sasrec 0.1720
- target_view (n=91,480): popularity 0.0030, repeat 0.2380, hybrid 0.3224, markov 0.1398, item_knn 0.1366, two_tower 0.2348, sasrec 0.3032
- target_addtocart (n=1,715): popularity 0.0035, repeat 0.2315, hybrid 0.3149, markov 0.1539, item_knn 0.1574, two_tower 0.1971, sasrec 0.2554
- target_transaction (n=742): popularity 0.0054, repeat 0.7089, hybrid 0.7143, markov 0.0863, item_knn 0.0741, two_tower 0.1698, sasrec 0.2372
