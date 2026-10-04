# Technical report: HNSW from first principles, measured against FAISS

## Abstract

We implement Hierarchical Navigable Small World graphs (Malkov and Yashunin, arXiv:1603.09320)
twice in Python: a readable reference engine that follows the paper's algorithms line by line,
and a compiled Numba engine that we prove builds the bit-identical graph (43 equivalence tests,
including property-based ones). Correctness rests on an exact brute-force oracle that remains
exact under float32 rounding and agrees with the published ground truth of three ANN-Benchmarks
datasets. A single-threaded, process-isolated benchmark against FAISS and hnswlib on SIFT,
Fashion-MNIST and GloVe shows the same recall at the same parameters (within 0.01 of FAISS
everywhere) with 0.8-9.0% fewer distance computations. The compiled engine reaches recall 0.97 at
0.076 ms per query on 100K SIFT vectors and runs within 1.1-1.5x of hnswlib and 1.4-3.5x of FAISS
at recall 0.95; the pure-Python engine is 4.5-6.6x slower still. Controlled studies reproduce the
paper's case for its neighbour-selection heuristic (recall 0.95 versus 0.50 on clustered data) and
expose a failure mode of its strict wording on duplicate-heavy data. On the full SIFT 1M the engine
serves 4,447 queries/s at recall 0.95 on one thread, 39.2x the throughput of exact search. Three
extensions are measured: filtered search, int8 scalar quantization and memory-mapped index files.

## 1. Problem

Find the k vectors most similar to a query among n stored vectors. Exact search costs O(n·d) per
query. Approximate nearest neighbour (ANN) search accepts missing a few true neighbours (measured
by recall) in exchange for sublinear work. HNSW is the dominant graph-based method and the default
index type of most vector databases, which makes it a good subject for an implementation that
must be understood, not just used.

## 2. Method

* **Layers.** Each vector receives a top layer `floor(-ln(U) * mL)`, `U ~ Uniform(0, 1]`,
  `mL = 1/ln(M)`, so each layer holds about 1/M of the one below (verified statistically in
  `tests/test_levels.py`).
* **Search (Algorithms 2 and 5).** Greedy descent with a beam of 1 through the upper layers, then a
  best-first beam search of width `max(efSearch, k)` on layer 0, with a candidate min-heap and a
  result max-heap; the search stops when the closest unexpanded candidate is farther than the
  worst kept result.
* **Insertion (Algorithm 1).** Beam search of width efConstruction on each layer the new node
  joins; select M neighbours; link both ways; re-select the links of any neighbour that exceeds its
  cap (M above layer 0, 2M on layer 0) with the same strategy; pass the whole candidate list down.
* **Neighbour selection (Algorithms 3 and 4).** "Simple" keeps the M closest. The "heuristic"
  keeps a candidate only if it is strictly closer to the base node than to every neighbour already
  kept, with the paper's two options (`extend_candidates`, `keep_pruned_connections`).
* **Canonical semantics.** Every tie-break and comparison is fixed (`DESIGN.md`, section 5):
  float32 storage, float64 accumulation, `(distance, id)` ordering in every heap. This is what
  makes two independent engines comparable bit for bit.

## 3. Implementation

| Component | Where | Notes |
|---|---|---|
| Exact oracle | `oracle.py` | float32 GEMM filter with a derived error margin, float64 re-ranking, ties to the lower id |
| Reference engine | `reference.py` | dictionaries of Python lists; vectorised distance calls per expansion |
| Compiled engine | `numba_engine.py` | flat int32 link tables, epoch-stamped visited array, whole loops compiled |
| Facade | `index.py` | validation, metric preparation, seeded levels, persistence, statistics |
| Persistence | `persistence.py` | versioned `.npz`, no pickle, generator state saved |
| Invariants | `invariants.py` | structural checks and two reachability measures |
| Harness | `bench/` | adapters, isolated workers, sleep detection, power and thermal capture |

The quality bar: 336 tests (invariants after every build, an exactness regime that must equal the
oracle, property-based tests, filtered search, quantization, persistence and corrupted files,
equivalence, harness), ruff and strict mypy clean, CI.

## 4. Experimental setup

Datasets: SIFT-128 (first 100,000 vectors), Fashion-MNIST-784 (all 60,000) and GloVe-100 angular
(first 100,000), 1,000 queries each, with exact ground truth from the oracle (`DATASETS.md`).
Indexes: both engines, FAISS `IndexHNSWFlat`, hnswlib (M = 16, efConstruction = 200, efSearch
10-800), FAISS exact and IVF baselines. One thread per library, one fresh process per index,
30-second cool-downs, warm-up, three timed passes, tie-aware recall@10. Machine: Apple M5 laptop,
16 GB, macOS 26.6, battery power, no thermal warnings recorded (`BENCHMARKS.md`, Methodology).

## 5. Results

**Correctness.** The oracle agrees with the published ground truth: all 1,000 SIFT and
Fashion-MNIST top-10 lists and 999 of 1,000 GloVe lists are valid exact answers under it; the
differences are exact ties and one 2e-9 float32 near-tie. The engines build identical graphs.

**Benchmarks** (single thread, recall@10; `BENCHMARKS.md`, raw files in `results/ann/`):

| Dataset | Max recall gap to FAISS at equal efSearch | Fewer distances than FAISS | QPS at recall >= 0.95: ours (compiled) / hnswlib / FAISS |
|---|---:|---:|---|
| SIFT 100K | 0.0096 | 0.8-4.4% | 13,142 / 19,043 / 33,338 |
| Fashion-MNIST 60K | 0.0033 | 0.8-9.0% | 6,948 / 7,810 / 23,982 |
| GloVe 100K | 0.0072 | 3.1-5.9% | 2,055 / 2,676 / 2,833 |

Build time for 100K SIFT vectors: 20.1 s (compiled), 206.4 s (pure Python), 15.4 s (hnswlib),
9.6 s (FAISS). Mission targets (build under 10 minutes; recall 0.95 under 1 ms): met, with recall
0.9678 at 0.0761 ms (`PERFORMANCE.md`).

**Studies** (`STUDIES.md`): heuristic versus simple selection on clustered data, recall 0.9533 versus
0.5004 at efSearch 320; M = 16 and efConstruction = 200 near the knee on SIFT; 25% exact duplicates
cap the strict heuristic's recall at 0.9654, `keep_pruned_connections` restores 1.0000; recall at
efSearch 40 falls from 1.0000 to 0.5212 as Gaussian dimension grows from 4 to 256.

**Scale (Tier 3).** On all 1,000,000 SIFT vectors the compiled engine serves 4,447 queries/s at
recall >= 0.95 (hnswlib 6,193, FAISS 10,301), within 1.23-1.59x of hnswlib across recall targets
0.90-0.99. Ten times more data cost our engine 2.96x in throughput and exact search 11.90x, so the
graph's advantage over exact search grew from 9.7x to 39.2x. Recall at equal efSearch is
0.0006-0.0171 below FAISS, with 4.5-6.4% fewer distance computations (`BENCHMARKS.md`).

**Extensions (Tier 3).** *Filtered search* (hnswlib semantics, identical in both engines) keeps
recall@10 at 0.95 or more from efSearch 20 up, for random and position-correlated predicates from
50% down to 1% selectivity, but its work grows as the predicate tightens (at efSearch 40, 18,080
distances per query at 1% random and 49,526 at 1% correlated); exact search over the allowed
vectors is faster from about 5% down for random and from 10% down for correlated predicates,
while post-filtering and FAISS's result-only filter lose
recall on selective predicates (`STUDIES.md` section 7). *int8 scalar quantization* is lossless on
SIFT and Fashion-MNIST (8-bit data) with a quarter of the vector memory; on GloVe it costs up to
0.0214 recall, which exact re-ranking of the beam recovers to within 0.0011 (`STUDIES.md` section
8). *Memory-mapped native files* open a 1M index without reading it (`PERFORMANCE.md` section 5).

## 6. Analysis

1. **The algorithm is faithful.** Equal recall at equal parameters across three datasets, and the
   same amount of work as FAISS within a few percent, show that the implementation follows the
   paper. The remaining recall differences at small efSearch (up to 0.0096) are about twice the
   spread between our own builds with different seeds (0.0050 over five seeds on SIFT 100K), so
   they reflect implementation details as well as chance; they vanish as efSearch grows.
2. **The cost is per operation, and it is understood.** The profile of the reference engine shows
   about half its build time going to the overhead of millions of tiny distance calls; compiling
   the same loops removes it (10.3x faster builds on SIFT). The remaining gap to FAISS shrinks as
   efSearch grows (2.59x at efSearch 10, 1.07x at 800 on SIFT), consistent with fixed per-query
   costs (about 7 µs of Python, heap allocation) plus float64 instead of SIMD float32 arithmetic,
   which weighs most on 784-dimensional Fashion-MNIST.
3. **Design choices matter in predictable ways.** The heuristic's value appears only on clustered
   data, as the paper argues; its strict inequality turns exact duplicates into isolated groups,
   which the paper does not discuss.
4. **Brute force is a strong baseline at 100K vectors, not at 1M.** On GloVe, graph search at
   recall 0.95 is only 1.09x faster than FAISS exact search, and IVF wins; on SIFT and
   Fashion-MNIST the compiled engine is 9.7x and 18.4x faster than exact search, and 39.2x on the
   full SIFT 1M.
5. **Filtering is a planning problem.** No single filtered strategy wins everywhere: the
   in-graph filter is accurate but its cost grows with selectivity, exact search over the allowed
   subset is cheap when that subset is small, and predicates correlated with position defeat both
   post-filtering and simple selectivity thresholds.

## 7. Limitations

* Pure Python and Numba cannot match hand-tuned C++ with SIMD; bit-exactness with the reference
  was preferred over the last factor of speed.
* Single-threaded; no deletions or updates.
* The three-library comparison uses 100K-vector subsets of SIFT and GloVe; the full SIFT 1M was
  run without the pure-Python engine. One hardware platform (a laptop, so absolute timings vary
  between runs; comparisons within a run are reliable).
* `QuantizedIndex` keeps the float32 vectors next to the codes for re-ranking; the memory saving
  needs them on disk (memory-mapped), a combination not measured here.
* Resident-memory measurements on macOS are noisy (memory compression, allocator retention): the
  pure-Python and compiled builds grew the process by 99.0 MB and 110.3 MB for the same 65.3 MB
  index; libraries are compared by index size instead.

## 8. Future work

A query planner that chooses between the in-graph filter and exact search over the allowed
subset; per-dimension quantization steps (a weighted distance kernel); an array-based heap and
float32 SIMD path behind a flag, to measure exactly what exactness costs; multi-threaded batch
search.

## References

* Malkov, Yashunin. Efficient and robust approximate nearest neighbor search using Hierarchical
  Navigable Small World graphs. arXiv:1603.09320 (TPAMI 2018).
* Aumüller, Bernhardsson, Faithfull. ANN-Benchmarks: A benchmarking tool for approximate nearest
  neighbor algorithms. Information Systems, 2020.
* Johnson, Douze, Jégou. Billion-scale similarity search with GPUs. IEEE Transactions on Big Data
  (FAISS), 2019.
* Jayaram Subramanya et al. DiskANN: Fast accurate billion-point nearest neighbor search on a
  single node. NeurIPS 2019.
* Jégou, Douze, Schmid. Product quantization for nearest neighbor search. TPAMI 2011.
