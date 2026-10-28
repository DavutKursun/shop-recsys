"""RetailRocket files → one events parquet (chunked) → vocabulary, time windows, sessions, sequences, evaluation cases.

Honesty rules built in here:
  * global time split: T0 < T1 < T2 < end; models learn from events before T0 only, the reranker from [T0, T1),
    model choice on dev [T1, T2), one look at test [T2, end)
  * the item vocabulary is fixed on events before T0; later items are `UNK` for id-based models and are counted
    as misses (their share is reported, never hidden)
  * an evaluation case is one event in a window given everything that user did before it — including events
    earlier in the same window — so every prediction is made with information available at that moment
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CACHE, EVENT_TYPES, FIRST_ITEM, HISTORY_BUCKETS, PAD, RAW, RAW_FILES, SPLITS, UNK, WINDOWS, WindowRules

EVENT_COLUMNS = ["ts", "user", "item", "etype"]
WINDOW_NAMES = ("train", "rerank", "dev", "test")
WINDOW_CODE = {n: i for i, n in enumerate(WINDOW_NAMES)}
DAY_MS = 86_400_000


# --- raw files ------------------------------------------------------------------------------------------------------
def find_raw(raw_dir: Path = RAW) -> dict[str, Path] | None:
    events, tree = raw_dir / RAW_FILES["events"], raw_dir / RAW_FILES["tree"]
    props = [raw_dir / p for p in RAW_FILES["properties"] if (raw_dir / p).exists()]
    if not events.exists() or not tree.exists() or not props:
        return None
    return {"events": events, "tree": tree, "properties": props}


def load_events(path: Path, chunksize: int = 1_000_000) -> pd.DataFrame:
    """events.csv → ts (ms), user, item, etype (0 view, 1 addtocart, 2 transaction); sorted by (user, ts)."""
    parts = []
    for chunk in pd.read_csv(path, usecols=["timestamp", "visitorid", "event", "itemid"], chunksize=chunksize,
                             dtype={"timestamp": "int64", "visitorid": "int64", "itemid": "int64", "event": "category"}):
        parts.append(pd.DataFrame({"ts": chunk["timestamp"].to_numpy(), "user": chunk["visitorid"].to_numpy(), "item": chunk["itemid"].to_numpy(),
                                   "etype": chunk["event"].map(EVENT_TYPES).astype("int8").to_numpy()}))
    df = pd.concat(parts, ignore_index=True)
    return df.sort_values(["user", "ts"], kind="stable").reset_index(drop=True)


def load_categories(paths: list[Path], chunksize: int = 2_000_000) -> pd.DataFrame:
    """The property files hold ~20M rows; only the `categoryid` rows matter. Latest value per item wins."""
    parts = []
    for p in paths:
        for chunk in pd.read_csv(p, usecols=["timestamp", "itemid", "property", "value"], chunksize=chunksize, dtype={"property": "string", "value": "string"}):
            c = chunk[chunk["property"] == "categoryid"]
            if len(c):
                parts.append(pd.DataFrame({"ts": c["timestamp"].astype("int64").to_numpy(), "item": c["itemid"].astype("int64").to_numpy(),
                                           "category": pd.to_numeric(c["value"], errors="coerce").to_numpy()}))
    cat = pd.concat(parts, ignore_index=True).dropna()
    cat["category"] = cat["category"].astype("int64")
    cat = cat.sort_values(["item", "ts"]).drop_duplicates("item", keep="last")
    return cat[["item", "category"]].reset_index(drop=True)


def load_tree(path: Path) -> dict[int, int]:
    """category → parent category (−1 for a root)."""
    t = pd.read_csv(path)
    parent = t["parentid"].fillna(-1).astype("int64")
    return dict(zip(t["categoryid"].astype("int64"), parent))


def cache_raw(raw: dict[str, Path], cache_dir: Path = CACHE) -> dict:
    """Write events.parquet and categories.parquet once; later runs start from the cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    ev = load_events(raw["events"])
    ev.to_parquet(cache_dir / "events.parquet", index=False, compression="zstd")
    cat = load_categories(raw["properties"])
    tree = load_tree(raw["tree"])
    cat["parent"] = cat["category"].map(tree).fillna(-1).astype("int64")
    cat.to_parquet(cache_dir / "categories.parquet", index=False)
    return {"events": len(ev), "users": int(ev["user"].nunique()), "items": int(ev["item"].nunique()), "items_with_category": len(cat),
            "categories": int(cat["category"].nunique()), "ts_min": int(ev["ts"].min()), "ts_max": int(ev["ts"].max()),
            "by_type": {k: int((ev["etype"] == v).sum()) for k, v in EVENT_TYPES.items()}}


# --- windows, sessions, vocabulary ------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Windows:
    t0: int
    t1: int
    t2: int
    end: int

    def code(self, ts: np.ndarray) -> np.ndarray:
        """0 train (< t0), 1 rerank, 2 dev, 3 test (≤ end)."""
        return np.searchsorted(np.array([self.t0, self.t1, self.t2]), ts, side="right").astype(np.int8)

    def as_dict(self) -> dict:
        iso = lambda v: pd.Timestamp(v, unit="ms").strftime("%Y-%m-%d %H:%M")
        return {"t0": self.t0, "t1": self.t1, "t2": self.t2, "end": self.end, "t0_iso": iso(self.t0), "t1_iso": iso(self.t1), "t2_iso": iso(self.t2), "end_iso": iso(self.end)}


def make_windows(ts_max: int, rules: WindowRules = WINDOWS) -> Windows:
    d = rules.days * DAY_MS
    end = int(ts_max) + 1
    return Windows(t0=end - 3 * d, t1=end - 2 * d, t2=end - d, end=end)


def dedup_consecutive(df: pd.DataFrame, gap_minutes: int = WINDOWS.session_gap_minutes) -> pd.DataFrame:
    """Drop an event that repeats the previous event's item for the same user within `gap_minutes` (a refresh or
    a back-click carries no signal). A later re-visit of the same item is kept: it is a real, common next-item
    target in e-commerce."""
    user, item, ts = df["user"].to_numpy(), df["item"].to_numpy(), df["ts"].to_numpy()
    same = (user[1:] == user[:-1]) & (item[1:] == item[:-1]) & (ts[1:] - ts[:-1] <= gap_minutes * 60_000)
    keep = np.concatenate([[True], ~same])
    return df[keep].reset_index(drop=True)


def session_ids(df: pd.DataFrame, gap_minutes: int) -> np.ndarray:
    """A new session starts at a new user or after a gap longer than `gap_minutes`. Requires (user, ts) order."""
    user, ts = df["user"].to_numpy(), df["ts"].to_numpy()
    new = np.concatenate([[True], (user[1:] != user[:-1]) | (ts[1:] - ts[:-1] > gap_minutes * 60_000)])
    return np.cumsum(new).astype(np.int64) - 1


def build_vocab(train_items: np.ndarray, min_events: int) -> pd.Series:
    """raw item → index ≥ FIRST_ITEM, ordered by training count (index − 1 is the popularity rank)."""
    counts = pd.Series(train_items).value_counts()
    counts = counts[counts >= min_events]
    return pd.Series(np.arange(FIRST_ITEM, FIRST_ITEM + len(counts), dtype=np.int64), index=counts.index)


# --- the dataset ------------------------------------------------------------------------------------------------------
@dataclass
class Cases:
    """Evaluation cases of one window, left-padded histories (PAD = 0) of the last `max_len` items."""

    history: np.ndarray  # (n, max_len) int64 item indices, oldest → newest, left-padded
    target: np.ndarray  # (n,) item index (UNK if outside the vocabulary)
    user: np.ndarray  # (n,) dense user index — the bootstrap cluster
    etype: np.ndarray  # (n,) event type of the target
    ts: np.ndarray  # (n,) timestamp of the target
    hist_len: np.ndarray  # (n,) true history length before truncation
    warm: np.ndarray  # (n,) bool: the user had at least one event before the window
    repeat: np.ndarray  # (n,) bool: the target item is in the (full) history

    def __len__(self) -> int:
        return len(self.target)

    @property
    def reachable(self) -> np.ndarray:
        return self.target != UNK

    def subset(self, mask: np.ndarray) -> Cases:
        return Cases(self.history[mask], self.target[mask], self.user[mask], self.etype[mask], self.ts[mask], self.hist_len[mask], self.warm[mask], self.repeat[mask])

    def last_items(self) -> np.ndarray:
        return self.history[:, -1]


class Dataset:
    """Events with indices and window codes, the item table, and everything derived from them."""

    def __init__(self, events: pd.DataFrame, items: pd.DataFrame, windows: Windows, rules: WindowRules = WINDOWS) -> None:
        self.events, self.items, self.windows, self.rules = events, items, windows, rules
        self.n_items = FIRST_ITEM + len(items)  # indices 0..n_items−1 (PAD, UNK, then the vocabulary)
        self.n_categories = int(items["category"].max()) + 1 if len(items) else 1
        self.n_parents = int(items["parent"].max()) + 1 if len(items) else 1
        self._user_bounds = None

    # construction -------------------------------------------------------------------------------------------------
    @classmethod
    def build(cls, events: pd.DataFrame, categories: pd.DataFrame, rules: WindowRules = WINDOWS) -> Dataset:
        ev = dedup_consecutive(events.sort_values(["user", "ts"], kind="stable").reset_index(drop=True), rules.session_gap_minutes)
        w = make_windows(int(ev["ts"].max()), rules)
        ev["window"] = w.code(ev["ts"].to_numpy())
        train = ev[ev["window"] == 0]
        vocab = build_vocab(train["item"].to_numpy(), rules.min_item_events)
        ev["iidx"] = ev["item"].map(vocab).fillna(UNK).astype(np.int64)
        ev["uidx"] = pd.factorize(ev["user"], sort=True)[0].astype(np.int64)
        ev["session"] = session_ids(ev, rules.session_gap_minutes)
        items = cls._item_table(train, vocab, categories, w, rules)
        return cls(ev, items, w, rules)

    @staticmethod
    def _item_table(train: pd.DataFrame, vocab: pd.Series, categories: pd.DataFrame, w: Windows, rules: WindowRules) -> pd.DataFrame:
        t = train[train["item"].isin(vocab.index)]
        g = t.groupby("item")["ts"]
        recent = t[t["ts"] >= w.t0 - rules.recent_days * DAY_MS]["item"].value_counts()
        items = pd.DataFrame({"iidx": vocab.to_numpy(), "item": vocab.index.to_numpy()})
        items["train_count"] = items["item"].map(g.size()).fillna(0).astype(np.int64)
        items["recent_count"] = items["item"].map(recent).fillna(0).astype(np.int64)
        items["first_ts"] = items["item"].map(g.min()).astype(np.int64)
        items["last_ts"] = items["item"].map(g.max()).astype(np.int64)
        items["popularity_rank"] = items["iidx"] - 1
        cat = categories.set_index("item") if len(categories) else pd.DataFrame(columns=["category", "parent"])
        items["category_raw"] = items["item"].map(cat["category"]).fillna(-1).astype(np.int64)
        items["parent_raw"] = items["item"].map(cat["parent"]).fillna(-1).astype(np.int64) if "parent" in cat else -1
        # dense category / parent indices, 0 = unknown, ordered by frequency
        for col in ("category", "parent"):
            raw = items[f"{col}_raw"]
            order = raw[raw >= 0].value_counts().index
            mapping = pd.Series(np.arange(1, len(order) + 1, dtype=np.int64), index=order)
            items[col] = raw.map(mapping).fillna(0).astype(np.int64)
        return items.set_index("iidx", drop=False).sort_index()

    # persistence --------------------------------------------------------------------------------------------------
    def save(self, out_dir: Path = SPLITS) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.events.to_parquet(out_dir / "events.parquet", index=False, compression="zstd")
        self.items.to_parquet(out_dir / "items.parquet", index=False)
        meta = {"windows": self.windows.as_dict(), "rules": self.rules.__dict__, "n_items": self.n_items, "n_categories": self.n_categories, "n_parents": self.n_parents}
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    @classmethod
    def load(cls, in_dir: Path = SPLITS) -> Dataset:
        if not (in_dir / "events.parquet").exists():
            raise FileNotFoundError(f"no dataset under {in_dir}; run scripts/build_dataset.py")
        meta = json.loads((in_dir / "meta.json").read_text(encoding="utf-8"))
        w = meta["windows"]
        items = pd.read_parquet(in_dir / "items.parquet").set_index("iidx", drop=False)
        rules = WindowRules(**meta["rules"])
        return cls(pd.read_parquet(in_dir / "events.parquet"), items, Windows(w["t0"], w["t1"], w["t2"], w["end"]), rules)

    # derived views ------------------------------------------------------------------------------------------------
    def _bounds(self):
        """Start / end row of every user in the (user, ts)-sorted events table."""
        if self._user_bounds is None:
            u = self.events["uidx"].to_numpy()
            starts = np.flatnonzero(np.concatenate([[True], u[1:] != u[:-1]]))
            self._user_bounds = (starts, np.append(starts[1:], len(u)))
        return self._user_bounds

    def train_sequences(self, min_events: int | None = None) -> list[np.ndarray]:
        """Per-user item-index sequences from the training window (UNK kept as a token; models never predict it)."""
        min_events = min_events or self.rules.min_train_events
        tr = self.events[self.events["window"] == 0]
        out = []
        for _, idx in tr.groupby("uidx", sort=False)["iidx"]:
            if len(idx) >= min_events:
                out.append(idx.to_numpy(dtype=np.int64))
        return out

    def train_sessions(self) -> list[np.ndarray]:
        tr = self.events[self.events["window"] == 0]
        return [s.to_numpy(dtype=np.int64) for _, s in tr.groupby("session", sort=False)["iidx"] if len(s) >= 1]

    def cases(self, window: str, max_len: int | None = None, max_cases: int | None = None, seed: int = 0) -> Cases:
        """One case per event in the window that has at least one earlier event of the same user."""
        max_len = max_len or self.rules.max_len
        code = WINDOW_CODE[window]
        ev = self.events
        iidx, uidx, win, ts, et = ev["iidx"].to_numpy(), ev["uidx"].to_numpy(), ev["window"].to_numpy(), ev["ts"].to_numpy(), ev["etype"].to_numpy()
        starts, ends = self._bounds()
        rows = np.flatnonzero(win == code)
        first_row = starts[uidx[rows]]
        rows = rows[rows > first_row]  # at least one earlier event
        if max_cases is not None and len(rows) > max_cases:
            rows = np.sort(np.random.default_rng(seed).choice(rows, max_cases, replace=False))
        n = len(rows)
        hist = np.full((n, max_len), PAD, dtype=np.int64)
        hist_len = np.empty(n, dtype=np.int64)
        warm = np.empty(n, dtype=bool)
        repeat = np.empty(n, dtype=bool)
        for i, r in enumerate(rows):
            s = starts[uidx[r]]
            h = iidx[s:r]
            hist_len[i] = len(h)
            k = min(len(h), max_len)
            hist[i, max_len - k :] = h[-k:]
            warm[i] = win[s] < code
            repeat[i] = iidx[r] != UNK and bool((h == iidx[r]).any())
        return Cases(hist, iidx[rows], uidx[rows], et[rows], ts[rows], hist_len, warm, repeat)

    def window_frame(self, window: str) -> pd.DataFrame:
        return self.events[self.events["window"] == WINDOW_CODE[window]]

    def raw_to_index(self, raw_items) -> np.ndarray:
        m = pd.Series(self.items["iidx"].to_numpy(), index=self.items["item"].to_numpy())
        return m.reindex(np.asarray(raw_items, dtype=np.int64)).fillna(UNK).to_numpy(dtype=np.int64)

    def summary(self, case_windows: tuple[str, ...] = ("rerank", "dev")) -> dict:
        """Dataset facts for DESIGN.md. `case_windows` never includes "test" before M4 (CLAUDE.md §2)."""
        ev = self.events
        out = {"windows": self.windows.as_dict(), "vocab_items": int(self.n_items - FIRST_ITEM), "events_after_dedup": len(ev), "users": int(ev["uidx"].max()) + 1,
               "sessions": int(ev["session"].max()) + 1, "categories": self.n_categories - 1, "by_window": {}}
        for name, code in WINDOW_CODE.items():
            part = ev[ev["window"] == code]
            out["by_window"][name] = {"events": len(part), "users": int(part["uidx"].nunique()), "items": int(part["item"].nunique()),
                                      "share_events_in_vocab": round(float((part["iidx"] != UNK).mean()), 4) if len(part) else None,
                                      "by_type": {k: int((part["etype"] == v).sum()) for k, v in EVENT_TYPES.items()}}
        seqs = self.train_sequences()
        lens = np.array([len(s) for s in seqs]) if seqs else np.array([0])
        out["train_sequences"] = {"users": len(seqs), "events": int(lens.sum()), "median_len": float(np.median(lens)), "p95_len": float(np.percentile(lens, 95)),
                                  "share_longer_than_max_len": round(float((lens > self.rules.max_len).mean()), 4)}
        out["cases"] = {w: self._cases_summary(w) for w in case_windows}
        return out

    def _cases_summary(self, window: str) -> dict:
        """What the evaluation cases of one window look like: history length, warm / repeat / reachable shares."""
        c = self.cases(window)
        n = max(len(c), 1)
        pct = (25, 50, 75, 90, 95, 99)
        q = np.percentile(c.hist_len, pct) if len(c) else np.zeros(len(pct))
        buckets = {f"hist_{lo}-{hi if hi < 10_000 else 'plus'}": int(((c.hist_len >= lo) & (c.hist_len <= hi)).sum()) for lo, hi in HISTORY_BUCKETS}
        return {"cases": len(c), "users": int(len(np.unique(c.user))), "events_in_window": int((self.events["window"] == WINDOW_CODE[window]).sum()),
                "hist_len_quantiles": {f"p{p}": float(v) for p, v in zip(pct, q)}, "hist_len_buckets": buckets,
                "share_warm": round(float(c.warm.sum() / n), 4), "share_repeat": round(float(c.repeat.sum() / n), 4),
                "share_target_reachable": round(float(c.reachable.sum() / n), 4),
                "share_repeat_among_reachable": round(float(c.repeat[c.reachable].mean()), 4) if c.reachable.any() else None,
                "by_target_type": {name: int((c.etype == code).sum()) for name, code in EVENT_TYPES.items()}}
