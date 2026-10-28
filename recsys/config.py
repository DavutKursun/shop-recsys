"""Every knob: paths, time windows, filtering rules, model sizes, seeds. No environment variables, no secrets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
CACHE = DATA / "cache"
SPLITS = DATA / "splits"
MODELS = ROOT / "models"
RESULTS = ROOT / "eval" / "results"

# RetailRocket (Kaggle: retailrocket/ecommerce-dataset). The loader needs events.csv and the two property files
# (only the `categoryid` rows are kept) and category_tree.csv.
RAW_FILES = {"events": "events.csv", "properties": ("item_properties_part1.csv", "item_properties_part2.csv"), "tree": "category_tree.csv"}
KAGGLE_DATASET = "retailrocket/ecommerce-dataset"

SEED = 7
EVENT_TYPES = {"view": 0, "addtocart": 1, "transaction": 2}
EVENT_NAMES = {v: k for k, v in EVENT_TYPES.items()}
PAD, UNK = 0, 1  # item index 0 pads sequences; 1 is any item outside the training vocabulary
FIRST_ITEM = 2


@dataclass(frozen=True)
class WindowRules:
    """Global time split. Everything before T0 trains the retrieval models; the next `days` train the reranker;
    then dev; then test. T0 = end − 3·days, T1 = end − 2·days, T2 = end − days."""

    days: int = 14
    session_gap_minutes: int = 30  # a gap longer than this starts a new session (item-kNN / Markov statistics)
    min_item_events: int = 5  # an item enters the vocabulary with at least this many events before T0
    min_train_events: int = 2  # a user contributes a training sequence with at least this many events before T0
    max_len: int = 50  # history length seen by the models
    recent_days: int = 7  # the "recent popularity" window at the end of the training period


@dataclass(frozen=True)
class SASRecConfig:
    dim: int = 64
    layers: int = 2
    heads: int = 2
    dropout: float = 0.2
    max_len: int = 50
    negatives: int = 1024  # uniform sampled-softmax negatives shared by the batch; 0 = full softmax
    event_weights: tuple[float, float, float] | None = None  # loss weight of a (view, addtocart, transaction) target; None = all 1
    batch_size: int = 128
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 0.0


@dataclass(frozen=True)
class TwoTowerConfig:
    dim: int = 64
    hidden: int = 128
    dropout: float = 0.1
    max_len: int = 50
    temperature: float = 0.05
    batch_size: int = 1024  # in-batch negatives: every other positive in the batch
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    logq_correction: bool = True  # subtract log of the sampling probability (Yi et al. 2019)
    max_examples: int = 2_000_000  # (history, next item) pairs per training run; subsampled above this


@dataclass(frozen=True)
class RerankRules:
    candidates_per_source: int = 50
    train_cases: int = 40_000  # cases sampled from the rerank window to train the reranker
    negatives_per_case: int = 40  # negative candidates kept per training case (all positives kept)
    rounds: int = 300


WINDOWS = WindowRules()
SASREC = SASRecConfig()
TWO_TOWER = TwoTowerConfig()
RERANK = RerankRules()
TOP_K = (10, 20)
PRIMARY_K = 10
BOOTSTRAP_RESAMPLES = 500
SOURCES = ("popularity", "repeat", "hybrid", "markov", "item_knn", "two_tower", "sasrec")  # candidate sources, in this order
HISTORY_BUCKETS = ((1, 1), (2, 4), (5, 19), (20, 10_000))
