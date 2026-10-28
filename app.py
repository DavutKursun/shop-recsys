"""shop-recsys API + UI: FastAPI on port 7860 with Gradio mounted at /ui.

POST /recommend           {"history": [raw item ids, oldest first], "k": 10, "model": "reranked"} → ranked items with score,
                          rank, category, category_path, popularity_rank, seen_before, sources; known / unknown history counts,
                          fallback flag, version, latency_ms
GET  /similar/{item}?k=10 nearest items by SASRec embedding cosine (exact search); 404 outside the 69,784-item vocabulary
GET  /health              models loaded, versions, item count
UI   /ui                  Shopper · Item · Evidence · About

Run: python scripts/export_serving.py (once), then uvicorn app:app --port 7860 --no-access-log
Requests are logged without item ids. Nothing is stored.
"""

from __future__ import annotations

import logging
import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # no telemetry from the UI

import recsys  # noqa: E402,F401  (OMP_NUM_THREADS; the macOS OpenMP check)
from recsys.api import create_app  # noqa: E402
from recsys.serving import SERVING, Service  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
service = Service.load(SERVING)
app = create_app(service)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=7860, access_log=False)
