.PHONY: setup setup-dev test dataset baselines sasrec two-tower index reranker eval-dev synth smoke app

PY ?= python

setup:
	$(PY) -m pip install -r requirements.txt
setup-dev:
	$(PY) -m pip install -r requirements-dev.txt
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
app:
	$(PY) -m uvicorn app:app --port 7860
