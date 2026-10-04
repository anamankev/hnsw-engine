# Reproduction targets for hnsw-engine. PY defaults to the shared workspace environment.
PY ?= ../.venv/bin/python
# Keep the machine awake during benchmarks (macOS); empty elsewhere.
CAFFEINATE := $(if $(shell command -v caffeinate 2>/dev/null),caffeinate -i,)
# The long pure-Python run goes last so it does not heat the machine before the others.
INDEXES = faiss-hnsw hnswlib hnsw-engine-numba faiss-flat faiss-ivf hnsw-engine-reference
RUN = $(CAFFEINATE) $(PY) benchmarks/run_ann.py --n-test 1000 --indexes $(INDEXES) --label main
REPORT = $(PY) benchmarks/report_ann.py --inject docs/BENCHMARKS.md

.PHONY: test lint typecheck check data verify-data bench-quick bench-sift bench-fmnist \
	bench-glove bench bench-sift1m report report-sift1m profile studies seeds filtered \
	quantization persistence persistence-cache tier3 reproduce

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests benchmarks
	$(PY) -m ruff format --check src tests benchmarks

typecheck:
	$(PY) -m mypy src/hnsw_engine benchmarks

check: lint typecheck test

data:
	$(PY) -m hnsw_engine.datasets download sift-128-euclidean --data-dir data
	$(PY) -m hnsw_engine.datasets download fashion-mnist-784-euclidean --data-dir data
	$(PY) -m hnsw_engine.datasets download glove-100-angular --data-dir data

verify-data: data
	$(PY) benchmarks/verify_datasets.py --queries 1000

# A few minutes: 10K SIFT vectors, 200 queries, one repetition.
bench-quick:
	$(PY) benchmarks/run_ann.py --dataset sift-128-euclidean --quick --indexes $(INDEXES)

bench-sift:
	$(RUN) --dataset sift-128-euclidean --n-train 100000

bench-fmnist:
	$(RUN) --dataset fashion-mnist-784-euclidean --n-train 60000

bench-glove:
	$(RUN) --dataset glove-100-angular --n-train 100000

bench: bench-sift bench-fmnist bench-glove

# Tier 3: all 1M SIFT vectors; the pure-Python engine is left out (its build would exceed the
# 20-minute time box; it builds the same graph as the compiled engine).
bench-sift1m:
	$(CAFFEINATE) $(PY) benchmarks/run_ann.py --n-test 1000 --label main \
		--indexes faiss-hnsw hnswlib hnsw-engine-numba faiss-flat faiss-ivf \
		--dataset sift-128-euclidean

report:
	$(REPORT) results/ann/sift-128-euclidean-n100000-main --name sift100k \
		--title "SIFT 100K (128-d, L2), one thread" --inject README.md
	$(REPORT) results/ann/fashion-mnist-784-euclidean-n60000-main --name fmnist60k \
		--title "Fashion-MNIST 60K (784-d, L2), one thread"
	$(REPORT) results/ann/glove-100-angular-n100000-main --name glove100k \
		--title "GloVe 100K (100-d, cosine), one thread"
	$(PY) benchmarks/check_targets.py \
		results/ann/sift-128-euclidean-n100000-main/hnsw-engine-numba-M16-ef_construction200.json \
		--inject docs/PERFORMANCE.md
	$(PY) benchmarks/derive_stats.py

profile:
	$(PY) benchmarks/profile_engine.py --engine reference --n 10000 --queries 500
	$(PY) benchmarks/profile_engine.py --engine numba --n 10000 --queries 500
	$(PY) benchmarks/query_overhead.py --n 100000 --ef 10 40 160

report-sift1m:
	$(REPORT) results/ann/sift-128-euclidean-nfull-main --name sift1m \
		--title "SIFT 1M (128-d, L2), one thread"
	$(PY) benchmarks/derive_stats.py

studies:
	$(PY) benchmarks/run_studies.py

# Recall spread over five seeds on SIFT 100K: the yardstick for recall gaps between libraries.
seeds:
	$(PY) benchmarks/run_seed_variation.py --inject docs/BENCHMARKS.md

# Tier 3 studies (docs/DESIGN.md section 9); each script caps every library at one thread.
filtered:
	$(CAFFEINATE) $(PY) benchmarks/run_filtered.py --inject docs/STUDIES.md

quantization:
	$(CAFFEINATE) $(PY) benchmarks/run_quantization.py --inject docs/STUDIES.md

# Builds the SIFT 1M index (about 6 minutes), then times opening it in fresh processes.
persistence:
	$(CAFFEINATE) $(PY) benchmarks/run_persistence.py --inject docs/PERFORMANCE.md

# The same with the index files evicted from the page cache before half of the runs (no root).
persistence-cache:
	$(CAFFEINATE) $(PY) benchmarks/run_persistence.py --cache both --inject docs/PERFORMANCE.md

# About 50 minutes: SIFT 1M benchmark and report, then the Tier 3 studies.
tier3: bench-sift1m report-sift1m seeds filtered quantization persistence persistence-cache

# Everything, about 1.5 hours on an Apple M5 (the pure-Python engine dominates).
reproduce: verify-data bench report profile studies
