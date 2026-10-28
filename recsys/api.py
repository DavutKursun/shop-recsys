"""The HTTP layer: a FastAPI app around one `Service`, with the Gradio demo mounted at /ui.

    POST /recommend           {"history": [raw item ids, oldest first], "k": 10, "model": "reranked"} → RecommendResponse
    GET  /similar/{item}?k=10 nearest items by SASRec embedding cosine, exact search; 404 outside the vocabulary
    GET  /health              models loaded, versions, item count

Requests are logged with the route template, status, latency and counts, never an item id (run uvicorn with
--no-access-log: its access log would print /similar/<item>). Nothing is stored.
"""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import RedirectResponse

from .config import FIRST_ITEM
from .schema import RecommendRequest, RecommendResponse, SimilarResponse
from .serving import Service

log = logging.getLogger("shop_recsys.api")
DESCRIPTION = ("What will this shopper look at or buy next: a two-stage recommender (seven retrieval sources, a LightGBM LambdaRank "
               "reranker) trained on the RetailRocket click stream of 2015. Item ids are raw RetailRocket ids; there are no product names.")


def _route(path: str) -> str:
    return "/similar/{item}" if path.startswith("/similar/") else path


def create_app(service: Service, ui: bool = True) -> FastAPI:
    app = FastAPI(title="shop-recsys", version=service.version, description=DESCRIPTION)
    app.state.service = service

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        if not request.url.path.startswith("/ui"):
            log.info("%s %s -> %s in %.1f ms", request.method, _route(request.url.path), response.status_code, (time.perf_counter() - t0) * 1000)
        return response

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/ui")

    @app.get("/health")
    def health():
        return service.health()

    @app.post("/recommend", response_model=RecommendResponse)
    def recommend(req: RecommendRequest) -> RecommendResponse:
        r = service.recommend(req.history, req.k, req.model)
        log.info("recommend model=%s k=%d history=%d known=%d unknown=%d fallback=%s items=%d latency_ms=%.1f", r.model, req.k, len(req.history),
                 r.known_history, r.unknown_history, r.fallback, len(r.items), r.latency_ms)
        return r

    @app.get("/similar/{item}", response_model=SimilarResponse)
    def similar(item: int, k: int = Query(default=10, ge=1, le=100)) -> SimilarResponse:
        r = service.similar(item, k)
        if r is None:
            log.info("similar k=%d -> item outside the vocabulary", k)
            raise HTTPException(status_code=404, detail=f"item {item} is not in the {service.n_items - FIRST_ITEM:,}-item vocabulary "
                                                        "(fewer than 5 events before the training cut-off)")
        log.info("similar k=%d items=%d latency_ms=%.1f", k, len(r.items), r.latency_ms)
        return r

    if ui:
        try:
            import gradio as gr

            from .ui import CSS, build

            app = gr.mount_gradio_app(app, build(service), path="/ui", css=CSS, footer_links=["api"])
        except ImportError:  # the API works without the UI
            log.warning("gradio is not installed; /ui is disabled")
    return app
