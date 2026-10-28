"""API schema: a request is a shopper's recent history (raw item ids, oldest first); the answer is a ranked list."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RecommendRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    history: list[int] = Field(min_length=1, max_length=200, description="raw item ids the shopper interacted with, oldest first")
    k: int = Field(default=10, ge=1, le=100)
    model: Literal["popularity", "repeat", "markov", "item_knn", "two_tower", "sasrec", "reranked"] = "reranked"


class Recommendation(BaseModel):
    item: int
    score: float
    rank: int = Field(ge=1)
    category: int | None = None
    popularity_rank: int | None = Field(default=None, description="1 = most interacted-with item in the training window")
    seen_before: bool = False
    sources: list[str] = Field(default_factory=list, description="which retrievers proposed the item (reranked model only)")


class RecommendResponse(BaseModel):
    items: list[Recommendation]
    model: str
    known_history: int = Field(description="how many history items are in the training vocabulary")
    version: str
    latency_ms: float
