# Datasets

All benchmark data comes from the standard ANN-Benchmarks HDF5 files
(https://github.com/erikbern/ann-benchmarks). Each file holds `train` (vectors to index),
`test` (queries), and `neighbors` / `distances` (the true 100 nearest training vectors of each
query, valid for the full training set only).

## Provenance and integrity

Downloaded on 2026-10-02. ANN-Benchmarks publishes no checksums, so the SHA-256 values below pin
the exact files every result in this repository was produced from; `hnsw_engine.datasets.download`
refuses a file whose checksum differs. Raw verification records:
`results/datasets/<name>.json` (produced by `benchmarks/verify_datasets.py`).

| Dataset | URL | Size (bytes) | SHA-256 |
|---|---|---:|---|
| sift-128-euclidean | https://ann-benchmarks.com/sift-128-euclidean.hdf5 | 525,128,288 | `dd6f0a6ed6b7ebb8934680f861a33ed01ff33991eaee4fd60914d854a0ca5984` |
| fashion-mnist-784-euclidean | https://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5 | 227,528,288 | `7f24e5122b71346346686a6ea80e08d9145b9c020f9f2db4c4c799dfcfc96a07` |
| glove-100-angular | https://ann-benchmarks.com/glove-100-angular.hdf5 | 485,413,888 | `544af1d5e84e112cd4749571dcfd8ca109818a572f850af75a3a09e093a953c4` |

Total download: 1,238,070,464 bytes (about 1.24 GB), within the 2 GB budget.

## Contents (verified)

| Dataset | Metric | train | test | neighbors | dtype | non-finite values | zero rows |
|---|---|---|---|---|---|---:|---:|
| sift-128-euclidean | L2 | 1,000,000 x 128 | 10,000 x 128 | 10,000 x 100 (int32) | float32 | 0 | 0 |
| fashion-mnist-784-euclidean | L2 | 60,000 x 784 | 10,000 x 784 | 10,000 x 100 (int32) | float32 | 0 | 0 |
| glove-100-angular | angular (cosine) | 1,183,514 x 100 | 10,000 x 100 | 10,000 x 100 (int32) | float32 | 0 | 0 |

No zero vectors exist in GloVe, so cosine distance is defined for every vector.

## Ground truth

The provided ground truth is valid only for the full training set; benchmarks on a prefix
subset (for example the first 100,000 SIFT vectors) need their own. `hnsw_engine.datasets.load`
therefore computes exact ground truth with the oracle (`oracle.exact_search`) for every
configuration and caches it in `data/ground_truth/` under a key that includes the dataset file's
SHA-256, the subset size, the number of queries and `k`.

**Cross-check against the provided ground truth** (full training sets, first 1,000 test queries,
top-10). "Valid under exact re-scoring" means every provided neighbour is within our exact 10th
distance (tolerance `1e-9 * max(1, d_10)`), i.e. the provided list is also a correct answer.

| Dataset | Identical top-10 lists | Mean overlap | Provided lists valid under exact re-scoring | Same sorted distances |
|---|---:|---:|---:|---:|
| sift-128-euclidean | 0.937 | 0.9991 | 1.000 | 1.000 |
| fashion-mnist-784-euclidean | 1.000 | 1.0000 | 1.000 | 1.000 |
| glove-100-angular | 0.999 | 0.9999 | 0.999 | 0.999 |

Interpretation: SIFT vectors are integer-valued, so exact distance ties are common. Of the 63
SIFT lists (6.3%) that are not identical, 54 contain the same ten neighbours in a different order
(tied points sorted differently) and 9 keep a different member of a tie at the 10th distance;
every provided list is still a correct answer. For one GloVe query the provided 10th neighbour is
`1.99e-9` (cosine distance) farther than the true 10th, consistent with float32 rounding when the
published ground truth was computed. Our oracle breaks ties by lower id and re-ranks in float64.

## Benchmark recall definition

Because of these ties, recall@10 counts a returned id as correct when its exact distance is no
greater than the exact 10th-neighbour distance (same tolerance), rather than requiring a specific
id. Id-based recall is recorded alongside it in every results file.

## Reproduction

```bash
python -m hnsw_engine.datasets download sift-128-euclidean --data-dir data
python -m hnsw_engine.datasets download fashion-mnist-784-euclidean --data-dir data
python -m hnsw_engine.datasets download glove-100-angular --data-dir data
python benchmarks/verify_datasets.py --queries 1000
```
