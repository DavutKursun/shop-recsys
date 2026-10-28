# shop-recsys: API + demo UI. Runs as a non-root user (uid 1000) on port 7860, as Hugging Face Docker Spaces expect.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    OMP_NUM_THREADS=2

# libgomp: LightGBM's OpenMP runtime (torch and faiss-cpu bundle their own; on Linux they coexist)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 user && mkdir -p /app && chown user:user /app
WORKDIR /app

COPY --chown=user requirements-space.txt ./
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-space.txt

COPY --chown=user . .
USER user

# torch, faiss and LightGBM doing real work in one process: the build fails here if their OpenMP runtimes clash
RUN python -c "from recsys._openmp_macos import PROBE; exec(PROBE)"

EXPOSE 7860
# --no-access-log: uvicorn's access log would print /similar/<item id>; the app logs route templates instead
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--no-access-log"]
