"""hnsw-engine: Hierarchical Navigable Small World search, built from first principles.

Quick start::

    import numpy as np
    from hnsw_engine import HNSWIndex

    data = np.random.default_rng(0).normal(size=(10_000, 64)).astype("float32")
    index = HNSWIndex(dim=64, metric="l2", M=16, ef_construction=200, seed=0)
    index.add(data)
    ids, distances = index.search(data[0], k=10)
"""

from .index import ENGINES, SELECTIONS, HNSWIndex, IndexStats
from .invariants import check_invariants, reachable_from_entry, searchable_from_entry
from .metrics import METRICS
from .oracle import exact_distances, exact_search
from .persistence import GraphState
from .quantization import QuantizedIndex, ScalarQuantizer

__all__ = [
    "ENGINES",
    "METRICS",
    "SELECTIONS",
    "GraphState",
    "HNSWIndex",
    "IndexStats",
    "QuantizedIndex",
    "ScalarQuantizer",
    "check_invariants",
    "exact_distances",
    "exact_search",
    "reachable_from_entry",
    "searchable_from_entry",
]

__version__ = "0.1.0"
