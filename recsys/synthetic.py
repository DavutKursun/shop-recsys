"""A fake shop with KNOWN structure, so every model can be checked before the real data arrives.

Shoppers have two favourite categories; items have a popularity (Zipf within category) and three "successor"
items that tend to follow them (a Markov chain, mostly inside the category); shoppers also re-visit items
they have seen. So: popularity is a real but weak signal, the last item is a strong one (Markov, item-kNN,
SASRec), the whole history adds a little (two-tower, SASRec), and repeats matter. Timestamps span `days`
days with sessions, so the time windows, sessions and the rerank/dev/test split all exercise the real code path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import EVENT_NAMES


def make_world(n_users: int = 3000, n_items: int = 400, n_categories: int = 20, days: int = 60, seed: int = 7,
               p_repeat: float = 0.15, p_follow: float = 0.55, mean_events: float = 12.0) -> dict:
    rng = np.random.default_rng(seed)
    item_ids = rng.permutation(np.arange(10_000, 10_000 + 3 * n_items))[:n_items]  # sparse raw ids like the real file
    category = rng.integers(0, n_categories, n_items)
    parent = category // 5
    # popularity: Zipf within category
    pop = np.zeros(n_items)
    for c in range(n_categories):
        members = np.flatnonzero(category == c)
        pop[members] = 1.0 / (1 + np.arange(len(members))) ** 1.1
    pop /= pop.sum()
    # successors: 3 per item, 80% inside the category
    successors = np.zeros((n_items, 3), dtype=np.int64)
    for i in range(n_items):
        same = np.flatnonzero(category == category[i])
        for j in range(3):
            pool = same if rng.random() < 0.8 and len(same) > 1 else np.arange(n_items)
            choice = i
            while choice == i:
                choice = rng.choice(pool)
            successors[i, j] = choice
    succ_p = np.array([0.5, 0.3, 0.2])

    rows = []
    start_ms = 1_430_000_000_000  # about May 2015, like the real file
    user_ids = rng.permutation(np.arange(1, 4 * n_users))[:n_users]
    for u in user_ids:
        interests = rng.choice(n_categories, 2, replace=False)
        n_ev = int(min(1 + rng.geometric(1.0 / mean_events), 80))
        t = start_ms + rng.uniform(0, days) * 86_400_000
        seen: list[int] = []
        for k in range(n_ev):
            if k == 0 or rng.random() >= p_repeat + p_follow:
                c = interests[0] if rng.random() < 0.7 else interests[1]
                members = np.flatnonzero(category == c)
                item = int(rng.choice(members, p=pop[members] / pop[members].sum()))
            elif seen and rng.random() < p_repeat / (p_repeat + p_follow):
                item = int(seen[rng.integers(len(seen))])
            else:
                item = int(rng.choice(successors[seen[-1]], p=succ_p))
            seen.append(item)
            r = rng.random()
            etype = 2 if r < 0.02 else (1 if r < 0.07 else 0)
            rows.append((int(t), int(u), int(item), etype))
            gap = rng.exponential(4 * 60_000) if rng.random() < 0.8 else rng.exponential(2.5 * 86_400_000)  # within / between sessions
            t += gap
    ev = pd.DataFrame(rows, columns=["ts", "user", "item_pos", "etype"])
    ev["item"] = item_ids[ev["item_pos"].to_numpy()]
    ev = ev[ev["ts"] < start_ms + days * 86_400_000].drop(columns="item_pos").sort_values(["user", "ts"]).reset_index(drop=True)
    cats = pd.DataFrame({"item": item_ids, "category": 100 + category, "parent": 1000 + parent})
    return {"events": ev, "categories": cats, "item_ids": item_ids, "successors": successors, "category": category, "popularity": pop}


def write_raw(world: dict, raw_dir: Path) -> dict[str, Path]:
    """The RetailRocket file layout, so the real loader is what the tests exercise."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    ev = world["events"]
    events = pd.DataFrame({"timestamp": ev["ts"], "visitorid": ev["user"], "event": ev["etype"].map(EVENT_NAMES), "itemid": ev["item"],
                           "transactionid": np.where(ev["etype"] == 2, np.arange(len(ev)), np.nan)})
    events.sample(frac=1.0, random_state=0).to_csv(raw_dir / "events.csv", index=False)  # the real file is not sorted
    cats = world["categories"]
    t0 = int(ev["ts"].min())
    stale = cats["category"].to_numpy().copy()
    stale[:5] += 1  # part 1 carries an outdated category for five items; part 2 (later timestamp) corrects it
    p1 = pd.DataFrame({"timestamp": t0, "itemid": cats["item"], "property": "categoryid", "value": stale.astype(str)})
    junk = pd.DataFrame({"timestamp": t0, "itemid": cats["item"], "property": "available", "value": "1"})
    hashed = pd.DataFrame({"timestamp": t0, "itemid": cats["item"].head(50), "property": "888", "value": "n277.200 1234"})
    pd.concat([p1, junk, hashed]).sample(frac=1.0, random_state=1).to_csv(raw_dir / "item_properties_part1.csv", index=False)
    later = cats.head(5)
    p2 = pd.DataFrame({"timestamp": t0 + 10, "itemid": later["item"], "property": "categoryid", "value": later["category"].astype(str)})
    p2.to_csv(raw_dir / "item_properties_part2.csv", index=False)
    tree = cats[["category", "parent"]].drop_duplicates().rename(columns={"category": "categoryid", "parent": "parentid"})
    roots = pd.DataFrame({"categoryid": tree["parentid"].unique(), "parentid": np.nan})
    pd.concat([tree, roots]).to_csv(raw_dir / "category_tree.csv", index=False)
    return {"events": raw_dir / "events.csv", "tree": raw_dir / "category_tree.csv", "properties": [raw_dir / "item_properties_part1.csv", raw_dir / "item_properties_part2.csv"]}
