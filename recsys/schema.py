"""API schema: a request is a shopper's recent history (raw item ids, oldest first); the answer is a ranked list."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ModelName = Literal["popularity", "repeat", "hybrid", "markov", "item_knn", "two_tower", "sasrec", "reranked"]


class RecommendRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    history: list[int] = Field(min_length=1, max_length=200, description="raw item ids the shopper interacted with, oldest first")
    k: int = Field(default=10, ge=1, le=100)
    model: ModelName = "reranked"


class Recommendation(BaseModel):
    item: int
    score: float
    rank: int = Field(ge=1)
    category: int | None = None
    category_path: str | None = Field(default=None, description="category ids from the root to the item's category (category_tree.csv)")
    popularity_rank: int | None = Field(default=None, description="1 = most interacted-with item in the training window")
    seen_before: bool = False
    sources: list[str] = Field(default_factory=list, description="the retrievers that proposed the item (reranked); a single model is its own source")


class RecommendResponse(BaseModel):
    items: list[Recommendation]
    model: str
    known_history: int = Field(description="how many history items are in the training vocabulary")
    unknown_history: int = Field(default=0, description="history items outside the vocabulary, read as the UNK token")
    history_used: int = Field(default=0, description="history items the models read (the most recent 50)")
    fallback: bool = Field(default=False, description="true when no history item the models read is known: recent popularity is served instead")
    note: str | None = None
    version: str
    latency_ms: float


class SimilarItem(BaseModel):
    item: int
    score: float = Field(description="cosine similarity of the SASRec item embeddings")
    rank: int = Field(ge=1)
    category: int | None = None
    category_path: str | None = None
    popularity_rank: int | None = None


class SimilarResponse(BaseModel):
    item: int
    category: int | None = None
    category_path: str | None = None
    popularity_rank: int | None = None
    items: list[SimilarItem]
    version: str
    latency_ms: float
