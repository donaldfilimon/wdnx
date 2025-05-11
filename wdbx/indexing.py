"""
indexing.py - Approximate nearest neighbor index utilities for WDBX using HNSW.
"""
from typing import List, Tuple

try:
    import hnswlib
except ImportError:
    hnswlib = None

class VectorIndex:
    """
    HNSW-based approximate nearest neighbor index.
    """
    def __init__(self, space: str = 'l2', dim: int = 128, max_elements: int = 10000):
        if hnswlib is None:
            raise ImportError("hnswlib is required for VectorIndex")
        self.index = hnswlib.Index(space=space, dim=dim)
        self.index.init_index(max_elements=max_elements, ef_construction=200, M=16)
        self.next_id = 0

    def add(self, vectors: List[List[float]]) -> List[int]:
        """
        Add vectors to the index, returning their assigned IDs.
        """
        ids = list(range(self.next_id, self.next_id + len(vectors)))
        self.index.add_items(vectors, ids)
        self.next_id += len(vectors)
        return ids

    def search(self, query: List[float], k: int) -> List[Tuple[int, float]]:
        """
        Query the index for k nearest neighbors to the query vector.
        Returns list of (id, distance).
        """
        labels, distances = self.index.knn_query(query, k=k)
        return list(zip(labels[0], distances[0])) 