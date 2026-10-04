# Performance: from the readable reference to a compiled engine

Milestone A3: profile the pure-Python reference, find the hotspots, build an optimized engine that
reproduces the reference *exactly*, and measure the result. Raw outputs: `results/profile/`
(`benchmarks/profile_engine.py`, `benchmarks/query_overhead.py`); end-to-end benchmark numbers come
from `docs/BENCHMARKS.md`.

## 1. Where the reference spends its time

`cProfile` on the first 10,000 SIFT vectors (M = 16, efConstruction = 200), then 500 queries at
efSearch = 100 (`results/profile/reference/summary.json`). Profiling inflates the absolute times;
the shares are what matter.

| Build phase (26.52 s under the profiler) | Calls | Self time (s) |
|---|---:|---:|
| `reference.py` `_search_layer` (heap loop) | 38,357 | 4.97 |
| `metrics.py` `distances_to_many` | 3,917,466 | 4.63 |
| `reference.py` `_distances` (wrapper, counters, gather) | 3,917,466 | 4.19 |
| NumPy `einsum` | 3,917,466 | 2.45 |
| `selection.py` `select_heuristic` | 11,616 | 1.59 |
| NumPy `astype` (float32 to float64) | 5,961,852 | 1.59 |
| NumPy `asarray` | 2,140,547 | 0.94 |

**Diagnosis.** Building 10,000 nodes makes 3.9 million distance calls, each on a handful of vectors
(one node's unvisited neighbours, or one candidate against the neighbours already kept by the
heuristic). The arithmetic per call is tiny, so the fixed cost of entering Python and NumPy
functions dominates: the distance wrapper, `distances_to_many`, `einsum` and `astype` together take
about half of the profiled build time, and the Python heap loop another 19%. Search shows the same
pattern. Faster arithmetic alone would not help; the per-call overhead has to go.

## 2. The optimized engine (`numba_engine.py`)

Same algorithm, same comparison and tie rules (`docs/DESIGN.md` section 5), different execution:

* **Flat storage.** Vectors in one contiguous float32 array. Layer-0 links in a fixed-width int32
  array `(capacity, 2M + 1)` with a count per node; upper-layer links in a `(rows, M + 1)` array
  where a node of level L owns L consecutive rows. The extra column holds the one link that may
  overflow a node before it is shrunk, exactly as the reference appends and then shrinks.
* **Compiled everything.** Distance kernels (float64 accumulation, like the reference), layer
  search, neighbour selection with both options, insertion and batched search are Numba
  `@njit(cache=True)` functions. A whole insertion batch or query batch runs in one call.
* **Same semantics, not just similar.** Heaps are `heapq` lists of `(distance, id)` tuples inside
  Numba, so ties resolve exactly as in the reference; the visited set is a preallocated `uint32`
  array stamped with an epoch counter (wrap-around tested).
* **No `fastmath`.** Allowing the compiler to reorder floating-point sums would make the engines
  disagree on near-ties of continuous data; exactness was the requirement.

**Equivalence evidence** (`tests/test_numba_engine.py`, 34 tests): identical graphs and bit-identical
search results on integer-valued data for every metric and selection configuration, including
60 random property-based cases; identical graphs and ids on continuous data (distances equal to
1e-12; observed differences are about 1e-16 relative, from summation order); cross-engine
save/load; growth beyond the initial capacity. The behavioural test modules additionally run every
test against both engines (274 tests in total at the end of Tier 2; 336 after the follow-ups).

## 3. Results

Profile of the compiled engine on the same workload (`results/profile/numba/summary.json`): the
build takes 1.20 s under the profiler (against 26.52 s), all of it inside the compiled insertion
kernel; the 500 queries take 0.04 s (against 0.48 s).

End-to-end (single thread, SIFT 100K, M = 16, efConstruction = 200): see the tables in
[`BENCHMARKS.md`](BENCHMARKS.md). Mission targets:

<!-- table:targets -->
| Target | Measured | Met |
|---|---|:---:|
| Build 100,000 SIFT vectors in under 10 minutes | 20.1 s | yes |
| Recall@10 >= 0.95 at under 1 ms mean latency | recall 0.9678 at 0.0761 ms (efSearch 40) | yes |

Source: `results/ann/sift-128-euclidean-n100000-main/hnsw-engine-numba-M16-ef_construction200.json` (engine `numba`, one thread, public `HNSWIndex.search` per query).
<!-- /table:targets -->

## 4. What a query costs now

`benchmarks/query_overhead.py` times the same 1,000 SIFT queries (100K vectors) three ways
(`results/profile/query_overhead.json`, microseconds per query, median of three passes):

| efSearch | Public API (`HNSWIndex.search`) | Engine call (no validation) | Batch (one compiled loop) | Python overhead |
|---:|---:|---:|---:|---:|
| 10 | 32.28 | 27.04 | 25.40 | 6.89 |
| 40 | 73.29 | 67.28 | 66.65 | 6.64 |
| 160 | 217.58 | 205.11 | 208.54 | 9.04 |

About 7 microseconds per query is Python: input validation, metric preparation and the call into
compiled code. The rest is the compiled search itself, so the remaining gap to FAISS lies inside
the kernel: float64 instead of SIMD float32 arithmetic, heap operations on Python-style tuples
(allocations), and no prefetching of neighbour vectors. Each could be removed, at the cost of
giving up bit-exact equivalence with the reference (or of a second, lower-level heap
implementation); with every target met by a wide margin, exactness was kept.

## 5. Opening a saved index (Tier 3)

`HNSWIndex.save` writes one `.npz` archive; loading it rebuilds the engine arrays in Python.
`save_native` writes the compiled engine's own arrays as `.npy` files that `load_native` can map
into memory read-only (`docs/DESIGN.md` section 9.3, `DECISIONS.md` D21).
`benchmarks/run_persistence.py` builds the full SIFT 1M index once, saves it both ways, then
opens it in fresh processes, three times per mode (medians below). Each process searches the
same 1,000 queries twice at efSearch 80; memory is the growth of the process's resident set over
its state before opening (`results/studies/persistence.json`).

<!-- table:persistence -->
| Open with | File size (MB) | Open (s) | Memory after open (MB) | First 1,000 queries (s) | Next 1,000 queries (s) | Memory after queries (MB) |
|---|---:|---:|---:|---:|---:|---:|
| `load` (.npz archive) | 701.8 | 1.665 | 1,506.7 | 0.202 | 0.183 | 1,513.9 |
| `load_native(mmap=False)` | 668.8 | 0.109 | 828.6 | 0.189 | 0.183 | 829.0 |
| `load_native(mmap=True)` | 668.8 | 0.058 | 316.9 | 0.200 | 0.182 | 829.1 |
| `load_native(mmap=True, validate=False)` | 668.8 | 0.001 | 8.0 | 0.205 | 0.182 | 672.9 |
<!-- /table:persistence -->

*Findings.* Every process searched the same index (identical result ids in all twelve, recall@10
0.9709, the same as the SIFT 1M benchmark at efSearch 80). Saving took 0.52 s as `.npz` and
0.07 s as native files. Conditions: AC power, one-minute load average 3.2-3.6 and no thermal
warnings (recorded per process in the JSON); the operating system's page cache was warm.

1. **Opening stops costing time.** The `.npz` archive takes 1.665 s, because loading rebuilds the
   engine arrays, and leaves 1,506.7 MB resident (the snapshot and the engine arrays both). The
   native files load into memory in 0.109 s. Memory-mapped with link validation, opening takes
   0.058 s; trusted (`validate=False`), 0.001 s with 8.0 MB resident.
2. **Validation is the price of safety.** Checking every link (`DECISIONS.md`, D22) reads the link
   arrays: opening takes 0.058 s instead of 0.001 s and leaves 316.9 MB resident instead of 8.0 MB
   at 1M vectors. That is why the trusted mode exists, and why it is documented as safe only for
   files the caller produced.
3. **Searching is the same once pages are resident.** The second batch of 1,000 queries took
   0.182-0.183 s in every mode. The first batch after a memory-mapped open took 0.200-0.205 s
   against 0.189 s after an in-memory load, the cost of the page faults with a warm cache; a cold
   cache would add disk reads.
4. **Mapping does not shrink steady-state memory for this workload.** After 1,000 queries the
   trusted memory-mapped process held 672.9 MB, essentially the whole 668.8 MB index, because
   queries touch vectors and links all over the graph. The benefits are start-up time and one
   shared copy of the index for all processes on a machine, not a smaller working set.

### Cold cache (follow-up, 2026-10-04)

The table above was measured with the index files already in the page cache. Dropping the whole
cache needs administrator rights (`sudo purge`), but a process can evict the pages of one file it
can read (`hnsw_engine/bench/pagecache.py`: `msync(MS_INVALIDATE)` on macOS, the method of
`vmtouch -e`, and `posix_fadvise` on Linux), and `mincore` shows whether it worked.
`run_persistence.py --cache both` rebuilds the 1M index and, in each of three repetitions, opens it
in every mode twice, once after reading the index files into the cache and once after evicting
them (`results/studies/persistence_cache.json`). Library files stay cached in both cases, so the
comparison isolates the index. Conditions: battery power, Low Power Mode off, one-minute load
average 2.6-2.9; recall@10 0.9709 and identical result ids in all 24 runs.

<!-- table:persistence_cache -->
| Open with | Open: warm / cold (s) | First 1,000 queries: warm / cold (s) | Next 1,000 queries: warm / cold (s) | Read from disk during the run: warm / cold (MB) |
|---|---:|---:|---:|---:|
| `load` (.npz archive) | 1.648 / 1.649 | 0.197 / 0.190 | 0.185 / 0.185 | 1.0 / 702.0 |
| `load_native(mmap=False)` | 0.102 / 0.155 | 0.186 / 0.186 | 0.185 / 0.185 | 0.1 / 667.0 |
| `load_native(mmap=True)` | 0.056 / 0.102 | 0.200 / 1.420 | 0.183 / 0.183 | 0.0 / 669.3 |
| `load_native(mmap=True, validate=False)` | 0.001 / 0.009 | 0.205 / 1.761 | 0.183 / 0.182 | 0.0 / 669.0 |
<!-- /table:persistence_cache -->

1. **The eviction worked, and the run shows it.** Every cold run started with none of the index
   files' pages in memory and read the whole index from disk while it ran (667.0-706.5 MB per run
   by the system's page-in counter); every warm run started with all pages resident and read at
   most 1.16 MB.
2. **Reading a file costs little on this SSD; faulting it in page by page costs a lot.** Loading
   into memory took 0.155 s cold against 0.102 s warm, and the `.npz` load did not change
   (1.649 s against 1.648 s), because rebuilding the arrays dominates it. Memory mapping moves the
   disk reads into the first queries, which then wait for one page fault at a time in the order
   the search happens to touch the pages: the first 1,000 queries took 1.420 s (validated) and
   1.761 s (trusted) cold, against 0.200-0.205 s warm.
3. **So on a cold cache, mapping is the slower way to start serving this index.** The trusted
   mapping opens in 0.009 s but needs 1.761 s for its first 1,000 queries; the in-memory load
   takes 0.155 s and then 0.186 s. Mapping pays off when the files are already cached (another
   process uses them, or the service restarts) or when queries touch a small part of the index;
   here 1,000 queries touch practically all of it.
4. **Afterwards nothing differs.** The second 1,000 queries took 0.182-0.185 s in every mode, warm
   or cold.

