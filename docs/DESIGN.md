# hnsw-engine: design

Status: living document. Sections marked *Revised* record where reality differed from the plan.

## 1. Goals

1. Implement Hierarchical Navigable Small World graphs (Malkov & Yashunin, arXiv:1603.09320)
   from first principles, with the paper's algorithms (1, 2, 3, 4, 5) recognisable in the code.
2. Prove correctness with tests: an exact brute-force oracle, structural invariants, an exactness
   regime (no pruning, `ef = n`) that must equal the oracle, property-based tests, and bit-for-bit
   equivalence between the readable reference engine and the optimized engine.
3. Provide an optimized engine (Numba) that is fast enough to index 100K SIFT vectors in minutes
   and answer queries in well under a millisecond, single-threaded.
4. Benchmark against FAISS (`IndexHNSWFlat`, `IndexFlat*`, `IndexIVFFlat`) and hnswlib on
   standard ANN-Benchmarks datasets, with a fair, documented, reproducible methodology.
5. Explain the results honestly, including where and why this implementation is slower.

## 2. Non-goals

* Beating FAISS or hnswlib on raw speed. They are C++ with SIMD distance kernels and years of
  tuning; the goal is to understand and measure the gap, not to close it.
* Multi-threaded construction or search. All engines and all baselines run on one thread so the
  comparison measures algorithms and implementations, not core counts.
* Deletion and in-place updates. (hnswlib supports soft deletes; this project does not.)
* Distributed or disk-resident indexes. Discussed in the README as production context only.

## 3. Architecture

```
src/hnsw_engine/
  _validation.py   input checking shared by every public entry point
  metrics.py       metric definitions: preparation (cosine normalisation) and distance kernels
  oracle.py        exact brute-force k-NN, the ground truth for everything else
  levels.py        level sampling: floor(-ln(U) * mL)
  selection.py     neighbour selection: simple (Alg. 3) and heuristic (Alg. 4), pure functions
  reference.py     ReferenceEngine: pure Python + NumPy HNSW core (readable, the oracle engine)
  numba_engine.py  NumbaEngine: compiled kernels over flat arrays (added in milestone A3)
  index.py         HNSWIndex: public facade (validation, level sampling, persistence, stats)
  persistence.py   versioned .npz format shared by both engines (no pickle)
  datasets.py      ANN-Benchmarks HDF5 loader, subsets, cached exact ground truth
  synthetic.py     synthetic data generators for tests and studies
  bench/           benchmark harness: adapters, isolated worker, metrics, environment capture
```

The facade owns everything that must be identical across engines: input validation, metric
preparation of vectors, the seeded random generator and level sampling. Engines receive
prepared `float32` vectors together with their pre-drawn levels, and implement only insertion
and search. This is what makes bit-for-bit equivalence between engines testable: given the same
inputs and levels, both must build the same graph.

## 4. Public API

```python
from hnsw_engine import HNSWIndex, exact_search

index = HNSWIndex(dim=128, metric="l2", M=16, ef_construction=200, ef_search=50,
                  seed=42, selection="heuristic", engine="numba")
ids = index.add(vectors)                       # (n, dim) array; returns assigned ids
ids, dists = index.search(query, k=10)         # one query, ef defaults to ef_search
ids, dists = index.batch_search(queries, k=10, ef=100)   # (nq, k) arrays
index.save("index.npz"); index = HNSWIndex.load("index.npz")
index.stats()                                  # IndexStats: sizes, layers, degrees, counters

# Tier 3 (section 9)
ids, dists = index.search(query, k=10, filter=mask)      # mask: one bool per indexed vector
index.save_native("dir"); index = HNSWIndex.load_native("dir", mmap=True)
QuantizedIndex(index).batch_search(queries, k=10, ef=80, rerank=80)
```

* `metric`: `"l2"` (squared Euclidean), `"cosine"` (1 - cosine similarity), `"ip"` (negative
  inner product). Distances are always "smaller is closer".
* `selection`: `"simple"` (M closest) or `"heuristic"` (paper's Algorithm 4), plus
  `extend_candidates` and `keep_pruned_connections` flags; `m_L` overrides `1/ln(M)`.
* Results are always length `k`. If fewer than `k` nodes are reachable (only possible in a
  disconnected graph), the tail is padded with id `-1` and distance `+inf`.
* Invalid input (wrong shape or dimension, NaN or infinity, zero vectors under cosine,
  `k` outside `[1, len(index)]`) raises `ValueError`.

## 5. Canonical semantics (both engines must follow these exactly)

These rules remove every source of non-determinism, so that two engines either agree bit-for-bit
or have a bug.

1. **Storage.** Vectors are stored as contiguous `float32`. Under `cosine`, vectors and queries
   are normalised in `float64` by the facade and then cast to `float32`.
2. **Distance.** Computed from `float32` inputs with `float64` accumulation:
   `l2 = sum((x - q)^2)`, `ip = -sum(x * q)`, `cosine = 1 - sum(x * q)`. The engines may sum in
   different orders, so continuous data can differ in the last bits; integer-valued data (such as
   SIFT) is exact in any order, which the equivalence tests exploit.
3. **Levels.** `level = floor(-ln(U) * m_L)` with `U = 1 - rng.random()`, so `U` is in `(0, 1]`.
   Drawn by the facade from `numpy.random.default_rng(seed)`, one per vector, in insertion order.
4. **Search within a layer (Algorithm 2).** Candidates: min-heap on `(distance, id)`. Results: at
   most `ef` elements; the worst is the maximum `(distance, id)`. Pop the closest candidate; stop
   if its distance is strictly greater than the worst result's distance. Scan its links in stored
   order, skipping visited nodes; a neighbour enters both heaps if results hold fewer than `ef`
   elements or its distance is strictly smaller than the worst distance; then evict the worst if
   results exceed `ef`. Output is sorted ascending by `(distance, id)`.
5. **Query (Algorithm 5).** From the entry point, run Algorithm 2 with `ef = 1` on each layer from
   the top down to layer 1, then with `ef = max(ef_search, k)` on layer 0; return the first `k`.
6. **Insertion (Algorithm 1).** Greedy descent (`ef = 1`) on layers above the new node's level.
   On each layer `lc = min(level, max_level) .. 0`: search with `ef_construction`, select `M`
   neighbours, set the new node's links to them (in selection order), then for each selected
   neighbour in that order append the new node to its links and, if it now exceeds
   `M_max(lc)` (`M` above layer 0, `2M` on layer 0), re-select its links with the same strategy.
   The whole candidate list becomes the entry set for the next layer. If the new node's level
   exceeds the current top level, it becomes the entry point after it is linked.
7. **Simple selection (Algorithm 3).** The first `M` candidates in ascending `(distance, id)`.
8. **Heuristic selection (Algorithm 4).** Optionally extend the working set with neighbours of
   the candidates on the same layer (excluding the base element itself and duplicates). Walk the
   working set in ascending `(distance, id)`; stop once `M` are kept. Keep `e` iff
   `d(e, base) < d(e, r)` for every already kept `r` (strict, as worded in the paper); otherwise
   record it as discarded. With `keep_pruned_connections`, fill remaining slots from the discarded
   list in order. Output order: kept (in walk order), then filled. The heuristic always runs, even
   when there are at most `M` candidates (the paper's behaviour; hnswlib skips it in that case).
9. **Filtered query (Tier 3).** A boolean mask `allowed` over node ids. The descent through the
   upper layers ignores the mask, so layer 0 starts from the same entry point as an unfiltered
   query. On layer 0, rule 4 changes in three places: a node enters the results only if it is
   allowed (disallowed nodes still enter the candidate heap, so the search can pass through
   them); the worst distance of an empty result set is `+inf`; and the search stops only when the
   results hold `ef` elements and the closest candidate is strictly farther than the worst
   result. These are the semantics of hnswlib's filter functor. A mask with no allowed node
   returns padding without searching.

## 6. Data

Standard ANN-Benchmarks HDF5 files (`train`, `test`, `neighbors`, `distances`):
`sift-128-euclidean`, `fashion-mnist-784-euclidean`, `glove-100-angular`. Provenance and SHA-256
checksums are recorded in `docs/DATASETS.md`. Provided ground truth is only valid for the full
training set, so subsets get exact ground truth from `oracle.py`, cached as `.npy` under `data/`.
Synthetic data (`synthetic.py`) covers uniform, clustered (Gaussian mixture), duplicate-heavy and
varying-dimension cases.

## 7. Evaluation plan

* **Correctness:** unit tests per module; invariants after every build in tests; exactness regime;
  `hypothesis` property tests; reference vs optimized equivalence on fixed seeds.
* **Benchmarks:** ANN-Benchmarks methodology. One index per fresh subprocess (clean memory
  measurement, crash isolation); single thread for every library; queries issued one at a time;
  warm-up before timing; three repetitions; sweep `efSearch` over 10..800. Metrics: recall@10,
  QPS, mean/p50/p99 latency, build time, index memory (RSS delta plus a compact logical size).
* **Studies:** one variable at a time: selection strategy on uniform vs clustered data, `M`,
  `efConstruction`, reachability after pruning, duplicates, dimension.

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Pure-Python reference too slow for 100K vectors | Vectorised per-expansion distance calls; Numba engine in A3; scale up gradually |
| Floating-point summation order breaks engine equivalence | Canonical float64 accumulation; integer-valued equivalence tests are exact |
| Unfair comparison (threads, warm-up, parameters) | Thread caps for every library, warm-up, identical M/efConstruction, environment recorded per run |
| Noisy timings on a laptop with performance and efficiency cores | Three repetitions, median reported, all repetitions saved |
| Ground-truth ties make id-based recall pessimistic | Recall counts a returned id as correct if its exact distance is within the true k-th distance |
| Graph disconnection from heuristic pruning | Reachability measured in the studies; `keep_pruned_connections` option |

## 9. Tier 3 extensions (designed before they were implemented)

### 9.1 Filtered search

Real queries carry metadata predicates ("in stock", "language = en"). `search` and
`batch_search` take `filter`, a boolean array with one entry per indexed vector, and follow rule
9 of section 5 in both engines (an equivalence test requires identical results). The cost model
is the point of the study: to collect `ef` allowed results the search must visit roughly
`ef / selectivity` nodes, so in-graph filtering degrades as the predicate gets more selective,
and below some selectivity exact search over the allowed vectors is cheaper. The study
(`benchmarks/run_filtered.py`) measures recall and throughput of in-graph filtering,
post-filtering (search unfiltered for `k / selectivity` or more results, keep the allowed ones)
and exact search over the allowed subset, on SIFT 100K at selectivities from 50% to 1%.

### 9.2 int8 scalar quantization

`ScalarQuantizer` maps each vector to one byte per dimension: `code = round((x - offset) / step)`
clipped to `[0, 255]`, with a per-dimension `offset` (the minimum) and one `step` shared by all
dimensions (the largest range divided by 255). If every training value is an integer and every
dimension spans at most 256 values, `step` is 1 and the codes are lossless (SIFT and
Fashion-MNIST are natively 8-bit). Sharing one step keeps the plain metrics usable in code space:
for L2 the distance between the transformed query `(q - offset) / step` and the codes is the
distance to the decoded vectors divided by `step^2`; for inner product and cosine the code-space
score with the unchanged query is an increasing affine function of the decoded one. Either way
the ranking is the same as against the decoded vectors, so the compiled Algorithm 2 and 5 code
runs unchanged over a `uint8` array (Numba compiles a second specialisation). The graph is still
built from the `float32` vectors. Optionally the best `R` candidates are re-scored with exact
`float32` distances. The study (`benchmarks/run_quantization.py`) reports recall and vector
memory (codes: `n * dim` bytes, against `4 * n * dim`) with and without re-ranking on SIFT,
Fashion-MNIST and GloVe. A per-dimension step would be more accurate on data with unequal ranges
but needs a weighted distance kernel; it is left out deliberately.

### 9.3 Memory-mapped persistence

`HNSWIndex.save(path)` writes one `.npz` archive, and loading it rebuilds the engine arrays, which
costs time proportional to the index and a full copy in memory. The native format
(`save_native(directory)`, `HNSWIndex.load_native(directory, mmap=True)`) writes the compiled
engine's own arrays as separate `.npy` files plus a JSON header (parameters, sizes, format
version, the random generator state). With `mmap=True` the arrays are memory-mapped read-only:
the operating system loads pages as searches touch them, and several processes share one copy in
the page cache. Adding vectors to a memory-mapped index copies the arrays into memory first
(copy-on-grow), so the files are never modified. Only the Numba engine has native arrays; the
reference engine keeps `.npz`.

**Validation on load (both formats).** The compiled kernels do not bounds-check, so a damaged or
hostile file with an out-of-range link could make a search read or write outside its arrays.
Every load therefore checks, vectorised: link counts within the degree caps, every followed link
a valid node id, upper-layer links pointing to nodes present on that layer, each node's
upper-layer rows tiling `links_up` without overlap, and an entry point on the top layer. For
memory-mapped files this reads the link arrays once; `validate=False` skips it for files the
caller produced, restoring constant-time opening.
