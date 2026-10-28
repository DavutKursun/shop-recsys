# test results — 95,905 cases, 38,804 users, run 2026-10-28

Targets reachable (in vocabulary): 74.5% · repeat targets: 23.3% · warm users: 39.1%

| model | recall@10 | recall@20 | ndcg@10 | mrr@10 | coverage@10 | pop. quantile |
|---|---|---|---|---|---|---|
| reranked | 0.3244 [0.3150, 0.3340] | 0.3669 [0.3577, 0.3770] | 0.2352 [0.2281, 0.2428] | 0.2072 [0.2006, 0.2143] | 0.711 | 0.191 |
| hybrid | 0.2953 [0.2858, 0.3044] | 0.3198 [0.3101, 0.3296] | 0.2059 [0.1988, 0.2132] | 0.1771 [0.1706, 0.1838] | 0.599 | 0.141 |
| sasrec | 0.2700 [0.2606, 0.2796] | 0.3153 [0.3048, 0.3260] | 0.1829 [0.1758, 0.1903] | 0.1556 [0.1495, 0.1623] | 0.740 | 0.227 |
| repeat | 0.2188 [0.2114, 0.2258] | 0.2241 [0.2165, 0.2312] | 0.1716 [0.1654, 0.1778] | 0.1558 [0.1500, 0.1616] | 0.428 | 0.085 |
| two_tower | 0.2091 [0.2012, 0.2173] | 0.2530 [0.2441, 0.2618] | 0.1346 [0.1292, 0.1401] | 0.1115 [0.1068, 0.1163] | 0.751 | 0.214 |
| markov | 0.1264 [0.1218, 0.1307] | 0.1504 [0.1455, 0.1550] | 0.0783 [0.0754, 0.0813] | 0.0634 [0.0609, 0.0660] | 0.479 | 0.092 |
| item_knn | 0.1253 [0.1206, 0.1300] | 0.1641 [0.1584, 0.1696] | 0.0713 [0.0685, 0.0742] | 0.0547 [0.0525, 0.0571] | 0.607 | 0.313 |
| popularity | 0.0040 [0.0034, 0.0049] | 0.0071 [0.0062, 0.0080] | 0.0019 [0.0016, 0.0023] | 0.0013 [0.0010, 0.0015] | 0.000 | 0.000 |

Candidate union recall (reranker ceiling, 20 per source): 0.3981

## Paired differences, recall@10 (user bootstrap)

- popularity_minus_hybrid: -0.2913 [-0.3006, -0.2821] *
- repeat_minus_hybrid: -0.0765 [-0.0800, -0.0730] *
- markov_minus_hybrid: -0.1689 [-0.1759, -0.1621] *
- item_knn_minus_hybrid: -0.1700 [-0.1772, -0.1630] *
- two_tower_minus_hybrid: -0.0862 [-0.0896, -0.0827] *
- sasrec_minus_hybrid: -0.0253 [-0.0284, -0.0220] *
- reranked_minus_hybrid: +0.0291 [+0.0275, +0.0309] *
- reranked_minus_sasrec: +0.0544 [+0.0516, +0.0574] *
- reranked_minus_two_tower: +0.1152 [+0.1118, +0.1189] *

## Breakdowns, recall@10

- warm (n=37,461): popularity 0.0037, repeat 0.2100, hybrid 0.2342, markov 0.0726, item_knn 0.0734, two_tower 0.1518, sasrec 0.1981, reranked 0.2533
- cold (n=58,444): popularity 0.0042, repeat 0.2244, hybrid 0.3345, markov 0.1609, item_knn 0.1586, two_tower 0.2459, sasrec 0.3161, reranked 0.3700
- repeat (n=22,310): popularity 0.0120, repeat 0.9361, hybrid 0.9362, markov 0.1554, item_knn 0.1465, two_tower 0.5346, sasrec 0.7493, reranked 0.9317
- new (n=73,595): popularity 0.0016, repeat 0.0013, hybrid 0.1010, markov 0.1176, item_knn 0.1189, two_tower 0.1105, sasrec 0.1248, reranked 0.1403
- hist_1-1 (n=31,415): popularity 0.0034, repeat 0.1267, hybrid 0.2528, markov 0.1310, item_knn 0.1290, two_tower 0.2219, sasrec 0.2735, reranked 0.2915
- hist_2-4 (n=28,661): popularity 0.0042, repeat 0.2607, hybrid 0.3501, markov 0.1486, item_knn 0.1413, two_tower 0.2439, sasrec 0.3320, reranked 0.3761
- hist_5-19 (n=19,603): popularity 0.0047, repeat 0.3085, hybrid 0.3490, markov 0.1268, item_knn 0.1332, two_tower 0.2138, sasrec 0.2756, reranked 0.3687
- hist_20-plus (n=16,226): popularity 0.0041, repeat 0.2149, hybrid 0.2159, markov 0.0775, item_knn 0.0803, two_tower 0.1173, sasrec 0.1470, reranked 0.2432
- target_view (n=93,551): popularity 0.0039, repeat 0.2165, hybrid 0.2936, markov 0.1265, item_knn 0.1257, two_tower 0.2104, sasrec 0.2714, reranked 0.3234
- target_addtocart (n=1,719): popularity 0.0041, repeat 0.1949, hybrid 0.2653, markov 0.1239, item_knn 0.1123, two_tower 0.1454, sasrec 0.1885, reranked 0.2833
- target_transaction (n=635): popularity 0.0189, repeat 0.6173, hybrid 0.6268, markov 0.1134, item_knn 0.1039, two_tower 0.1921, sasrec 0.2882, reranked 0.5843
