# Decision log (hnsw-engine)

Each entry: context, decision, alternatives considered, consequences. Newest last.

## D1. Environment: one uv-managed Python 3.12 environment

* **Context.** The machine's system Python is 3.14; several scientific wheels lag new Python
  releases. `uv` was not installed and Homebrew is absent.
* **Decision.** Install `uv` at user level from PyPI (`python3 -m pip install --user uv`), let it
  download a managed CPython 3.12.15, and create `.venv` at the workspace root. hnswlib 0.8.0 has
  no macOS arm64 wheel, so it is compiled from source with Xcode's clang during install.
* **Alternatives.** The `curl | sh` uv installer (modifies shell profiles); Python 3.14 (risk of
  missing wheels); conda (heavier, not requested).
* **Consequences.** Exact versions are recorded with every benchmark and in
  `requirements-lock.txt` at the workspace root.

## D2. float32 storage, float64 distance accumulation, in both engines

* **Context.** The optimized engine must be provably equivalent to the reference. Summing in a
  different order changes the last bits of a float32 sum often enough to flip graph decisions.
* **Decision.** Store vectors as float32 (like FAISS) but accumulate every distance in float64.
  Equivalence is then exact on integer-valued data (any summation order is exact) and differs
  only at the ~1e-16 relative level on real data.
* **Alternatives.** float32 accumulation (faster SIMD, but engines would disagree on near-ties);
  calling the same compiled kernel from both engines (the reference would no longer be an
  independent implementation).
* **Consequences.** Distance kernels do slightly more work than FAISS's float32 SIMD kernels; this
  is one of the explained sources of the speed gap.

## D3. The neighbour-selection heuristic follows the paper literally

* **Context.** Algorithm 4 keeps `e` if it is "closer to q compared to any element from R".
  hnswlib keeps `e` unless some kept `r` is *strictly* closer (ties kept) and skips the heuristic
  entirely when there are fewer than `M` candidates.
* **Decision.** Strict inequality `d(e, base) < d(e, r)` and no early exit, as worded in the paper
  and in the mission brief.
* **Alternatives.** hnswlib's conventions.
* **Consequences.** With exact duplicate vectors, a node whose duplicate is selected first prunes
  every other candidate at equal distance (pinned by `test_duplicate_of_base_starves_the_node`).
  Measured in the duplicate study (A6).

## D4. Upper-layer descent reuses SEARCH-LAYER with ef = 1

* **Context.** Many implementations write a separate greedy loop for the upper layers.
* **Decision.** Call Algorithm 2 with `ef = 1`, exactly as the paper's Algorithms 1 and 5 do.
* **Alternatives.** A dedicated greedy loop (slightly faster, but a second code path whose tie
  behaviour must be kept consistent).
* **Consequences.** One search routine to test and to port to the optimized engine.

## D5. Exact oracle: float32 filter plus float64 re-ranking with a proven margin

* **Context.** Ground truth for 1M vectors needs BLAS speed, but float32 matrix products can
  reorder near-ties, and a float64 copy of SIFT-1M would need 1 GB.
* **Decision.** Filter candidates with float32 GEMM keeping everything within twice a rigorous
  error bound of the k-th distance, then re-rank survivors in float64 with ties broken by id.
* **Alternatives.** Pure float64 (memory and speed); trusting float32 rankings (not exact).
* **Consequences.** Exact results even under catastrophic cancellation (tested), chunked memory.

## D6. Persistence as versioned `.npz`, no pickle

* **Context.** Pickle files execute code when loaded and tie the format to class internals.
* **Decision.** Save vectors, levels and per-layer CSR adjacency plus JSON metadata (parameters
  and the random generator's state) with `numpy.savez`; load with `allow_pickle=False`.
* **Consequences.** Loading cannot execute code embedded in the file (and, since D22, a damaged
  file is rejected before the compiled code touches it), the format is engine-independent, and
  `load` followed by `add` continues the random stream exactly (tested).

## D7. `results/` is git-ignored; published figures and tables live in `docs/`

* **Context.** The mission asks for `results/` in `.gitignore`, yet documents must be traceable to
  raw outputs.
* **Decision.** Raw outputs (JSON with configs and environment) go to `results/`. Scripts write
  the figures and summary tables that documents cite into `docs/figures/` and `docs/tables/`,
  which are versioned.
* **Consequences.** A reader can verify every published number against a tracked table and
  regenerate the raw files with the reproduction commands.

## D8. Search results are fixed length

* **Decision.** `search` and `batch_search` always return `k` results; if fewer nodes are reachable
  (disconnected graph) the tail is `-1` / `+inf`, like FAISS. `k > len(index)` raises.
* **Consequences.** Batch results are rectangular arrays; recall code treats `-1` as a miss.

## D9. "l2" means squared Euclidean distance

* **Decision.** Return squared L2, as FAISS and hnswlib do; the square root never changes a
  ranking. Recall is computed from ids and exact distances, so units never mix across libraries.

## D10. ANN-Benchmarks "single query" mode, median of three repetitions

* **Context.** Libraries can be timed per query or in batches; batches hide per-call overhead and
  favour C++ libraries with internal batching.
* **Decision.** One Python call per query, timed with `perf_counter_ns`, 100 warm-up queries, three
  timed passes; report the median of the per-pass statistics and save every pass.
* **Alternatives.** Best of N (ANN-Benchmarks' choice; more optimistic); batch mode.
* **Consequences.** Latency includes each library's Python-binding overhead, a large share of the
  fastest configurations (a whole FAISS query takes 12.5 µs at efSearch = 10 on SIFT 100K); stated as a threat
  to validity in `BENCHMARKS.md`.

## D11. Common graph parameters: M = 16, efConstruction = 200

* **Decision.** Every HNSW implementation is built with M = 16 and efConstruction = 200 (the values
  in hnswlib's documentation and common in ANN-Benchmarks configurations) and swept over the same
  efSearch values (10 to 800).
* **Consequences.** Differences at equal parameters isolate the implementation; parameter studies
  (A6) vary one at a time.

## D12. IVF partition count: 4 * sqrt(n) rounded to a power of two

* **Decision.** `nlist = 2 ** round(log2(4 * sqrt(n)))` (1,024 for 100K vectors), following the FAISS
  guideline of a few times sqrt(n); nprobe swept over powers of two up to nlist.

## D13. Two memory measures

* **Decision.** Report both the resident-set growth of a fresh worker process across the build
  (what the process really costs, including interpreter and allocator overhead) and a
  library-specific data-structure size (ours: compact fixed-width layout; FAISS: serialised size;
  hnswlib: saved file size).
* **Consequences.** Index sizes are comparable across libraries (65.3 MB ours, 65.6 MB FAISS on
  SIFT 100K). RSS growth is reported but treated as indicative only: on macOS it is affected by
  memory compression and allocator retention (in the published run the pure-Python and compiled
  builds grew the process by 99.0 MB and 110.3 MB for the same 65.3 MB index), and an earlier run of
  the same build measured a much larger growth.

## D14. Figures use the validated default categorical palette, plus markers

* **Context.** Colour-vision-deficiency safety should be checked with a validator script, which
  needs Node.js; Node is not installed on this machine.
* **Decision.** Use the documented, pre-validated reference palette unchanged, in its fixed slot
  order, with each index keeping its colour and marker in every figure, and distinct markers as a
  second, colour-independent encoding; light and dark variants are rendered.

## D15. Numba engine: same semantics first, speed second

* **Context.** The optimized engine must be provably equivalent to the reference (bit-identical
  graphs on fixed seeds), and the profile showed per-call overhead, not arithmetic, dominating.
* **Decision.** Compile the whole insertion and search loops with Numba over flat arrays
  (fixed-width int32 link tables with one overflow column per row, a preallocated epoch-stamped
  visited array), keep float64 accumulation, and keep Python-semantics `heapq` lists of
  `(distance, id)` tuples inside Numba so ties resolve exactly as in the reference. No `fastmath`.
* **Alternatives.** Array-based heaps and SIMD float32 kernels (faster, but tie and rounding
  behaviour would diverge from the reference); Cython or Rust (more toolchain, same idea).
* **Consequences.** On SIFT 100K, 10.3x faster builds and 4.5-6.6x faster queries than the
  reference (`results/ann/derived_stats.json`), every A3 target met, and equivalence proven by 34
  tests. The remaining gap to FAISS is documented in `PERFORMANCE.md`.

## D16. The compiled engine is the default; both engines run the behavioural tests

* **Decision.** `HNSWIndex(engine="numba")` is the default since equivalence is proven. The
  behavioural test modules (`test_index.py`, `test_properties.py`) run every test once per engine
  through a module-level fixture, so the reference stays fully exercised as the oracle.
* **Consequences.** 274 tests at the time of writing; the suite takes about 35 seconds on an idle machine.

## D17. Benchmark hardening after a mid-run sleep

* **Context.** During the first full run the laptop lid was closed: macOS slept for 886 s and
  again for 825 s while Fashion-MNIST workers were running (power log). `perf_counter` stops while
  the machine sleeps, so the recorded timings looked normal; the machine was also on battery and
  hot after hours of work (a fanless MacBook Air), which slowed some runs by up to 41% relative to
  an earlier run.
* **Decision.** (1) Every worker records `suspended_seconds` = wall-clock time minus monotonic time;
  the runner retries any run that slept longer than 1 s (up to three attempts). (2) Every result
  records the power source, Low Power Mode and the thermal warnings reported by `pmset`, before and
  after the run. (3) 30 s cool-down between workers, and the long pure-Python run goes last.
  (4) Benchmarks run under `caffeinate -i`. (5) A worker's output file is deleted before each
  attempt, so a stale result can never be mistaken for a fresh one. All published numbers come
  from the re-run with these safeguards.
* **Consequences.** Comparisons within one run are taken under recorded, similar conditions;
  absolute numbers can still vary between runs on a laptop, which `BENCHMARKS.md` states.

## D18. SIFT 1M without the pure-Python engine

* **Context.** Tier 3 asks for the full SIFT dataset. The pure-Python engine needed 206.4 s for
  100K vectors. HNSW construction grows faster than linearly: the compiled engine took 18.8x
  longer for 10x the data. Applying the same factor (an extrapolation, not a measurement) puts a
  pure-Python 1M build at about an hour, beyond the 20-minute time box for one experiment.
* **Decision.** `make bench-sift1m` runs FAISS HNSW, hnswlib, the compiled engine, FAISS IVF and
  FAISS flat on all 1,000,000 vectors with the unchanged harness and parameters. The pure-Python
  engine is left out; its equivalence with the compiled engine is established by the test suite
  and by identical recall and distance counts on the three 100K-scale benchmarks.
* **Alternatives.** Run the pure-Python engine on a subset (adds nothing: the engines are
  identical); raise the time box (against the mission's rules).
* **Consequences.** At 1M the compiled engine reaches recall 0.9709 at efSearch 80 and is within
  1.23-1.59x of hnswlib at recall 0.90-0.99 (`results/ann/derived_stats.json`). Process memory
  deltas at this scale are smaller than the index arrays themselves (121.5 MB for our 652.6 MB
  index), confirming that RSS deltas on macOS are not a memory measure (D13); index sizes are
  compared instead.

## D19. Filtered search follows hnswlib's semantics, in both engines

* **Context.** Filtered queries need a rule for nodes that fail the predicate. Options: skip them
  entirely (the search cannot cross regions where the predicate fails), traverse them but never
  return them (hnswlib), or bound the candidate list as FAISS does.
* **Decision.** Traverse disallowed nodes, admit only allowed ones to the results, and stop only
  when the result set is full and the closest candidate is worse than its worst element (design
  rule 9). The upper-layer descent ignores the filter. The filter is a boolean mask, validated
  for dtype and length; an all-false mask returns padding without searching. The reference and
  compiled engines implement it separately and an equivalence test requires identical results and
  identical distance counts.
* **Alternatives.** A callable predicate (flexible, but a Python call per node would defeat the
  compiled engine); ACORN-style two-hop expansion (better at very low selectivity, more complex).
* **Consequences.** With every node allowed the filtered search takes exactly the unfiltered
  steps (tested). At low selectivity the search must visit many nodes; the study in
  `STUDIES.md` section 7 measures where exact search over the allowed subset becomes cheaper.

## D20. int8 quantization with one shared step, used for traversal only

* **Context.** Scalar quantization cuts vector memory by 4x. A per-dimension step is more
  accurate, but then distances need a weighted kernel and a second search implementation.
* **Decision.** Per-dimension offset, one step for all dimensions (step 1, lossless, when the data
  are 8-bit integers). Because the step is shared, code-space distances rank candidates exactly as
  distances to the decoded vectors do, so the existing compiled search runs over a `uint8` array
  unchanged (Numba compiles a second specialisation). The graph is still built from `float32`
  vectors; optional re-ranking of the best candidates uses exact `float32` distances.
* **Alternatives.** Per-dimension steps (FAISS `QT_8bit`); product quantization (much smaller
  codes, much larger accuracy loss, a separate codebook training step); building the graph from
  codes (cheaper build, worse graph).
* **Consequences.** One code path for float32 and int8 traversal; tests prove the code-space to
  decoded-distance conversion for all three metrics and exact equality with the float32 search on
  lossless data. Results in `STUDIES.md` section 8.

## D21. Native format: the compiled engine's arrays as memory-mappable `.npy` files

* **Context.** `.npz` loading rebuilds the engine arrays in Python, which takes time and memory
  proportional to the index; production services open large indexes many times.
* **Decision.** `save_native` writes each engine array as its own `.npy` file plus a JSON header
  (parameters, generator state, dtypes and shapes, format version); `load_native(mmap=True)` maps
  the files read-only and the engine uses them in place. Any insertion first copies the arrays
  into memory (copy-on-write), so the files are never modified. The header is written last, so an
  interrupted save leaves no loadable index.
* **Alternatives.** A single custom binary file with offsets (fewer files, more code); keeping
  only `.npz` (simple, slow to open at scale).
* **Consequences.** Opening no longer rebuilds the index (measured in `PERFORMANCE.md`). Only the
  Numba engine has native arrays; the reference engine keeps `.npz`. Pages are read from disk as
  searches touch them, so on a cold cache the first queries pay for it: measured later, the first
  1,000 queries on the 1M index took 1.761 s after a trusted cold open against 0.186 s after
  loading the files into memory (`PERFORMANCE.md`, cold cache).

## D22. Every load validates the graph, because the compiled code does not bounds-check

* **Context.** Self-review found that "pickle-free, so loading cannot run code" was not the whole
  story: Numba kernels index arrays without bounds checks and write `visited[e]` for every link
  `e` they follow, so a file with an out-of-range link id could make a search read or write outside
  its arrays. Both the `.npz` loader and the native loader trusted the files.
* **Decision.** `validate_state` (for `.npz`) and `_validate_native` (for native files) check every
  link a search can follow, layer membership of upper-layer links, the tiling of upper-layer rows,
  degree caps and the entry point, all vectorised; they run on every load. `load_native` takes
  `validate=False` for files the caller produced, because validation reads the link arrays and so
  costs memory-mapped opening its constant time.
* **Alternatives.** Bounds checks inside the kernels (Numba's `boundscheck=True`; slower on every
  search, and an error mid-search instead of at load); trusting files (unsafe for an index format).
* **Consequences.** Corrupted files fail at load with a message naming the problem (tested for
  out-of-range links, negative ids, bad counts, overlapping rows, links to nodes absent from a
  layer and a bad entry point). The cost of validation is measured in `PERFORMANCE.md` section 5.
