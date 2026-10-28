.PHONY: setup setup-dev openmp-fix test dataset baselines sasrec two-tower index reranker eval-dev synth smoke serving bench app deploy

PY ?= python

setup:
	$(PY) -m pip install -r requirements.txt && $(PY) scripts/fix_openmp_macos.py
setup-dev:
	$(PY) -m pip install -r requirements-dev.txt && $(PY) scripts/fix_openmp_macos.py
openmp-fix:       ## macOS: one OpenMP runtime for torch, faiss-cpu and lightgbm (no-op elsewhere; rerun after upgrading them)
	$(PY) scripts/fix_openmp_macos.py
test:
	$(PY) -m pytest -q
dataset:          ## raw files → parquet cache → vocabulary, windows, sessions → data/splits/
	$(PY) scripts/download.py && $(PY) scripts/build_dataset.py
baselines:        ## popularity, repeat, Markov, item-kNN → models/
	$(PY) scripts/train_baselines.py
sasrec:
	$(PY) scripts/train_neural.py --model sasrec
two-tower:
	$(PY) scripts/train_neural.py --model two_tower
index:            ## FAISS exact vs HNSW on the SASRec item vectors
	$(PY) scripts/build_index.py --model sasrec
reranker:
	$(PY) scripts/train_reranker.py
eval-dev:
	$(PY) scripts/evaluate.py --split dev --sampled-demo
synth:            ## the whole pipeline on the synthetic shop (no real data)
	$(PY) scripts/synthetic_check.py
smoke:            ## the script chain on a synthetic export
	$(PY) -m pytest -q tests/test_scripts.py
serving:          ## the served bundle → serving/ (eight models, items, 2,000 dev shoppers, checks against the evaluation path)
	$(PY) scripts/export_serving.py --fallback-check
bench:            ## /recommend and /similar latency, in-process and over HTTP → eval/results/latency_local.json
	$(PY) scripts/bench_latency.py
app:              ## API + UI on :7860 (no access log: it would print item ids)
	$(PY) -m uvicorn app:app --port 7860 --no-access-log
deploy:           ## the Hugging Face Space (needs `hf auth login`) → eval/results/latency_space.json
	$(PY) scripts/deploy_space.py
