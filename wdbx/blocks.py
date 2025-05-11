"""
blocks.py - DataBlock definition for WDBX chain.
"""
import hashlib
import json
from typing import List, Dict, Optional

class DataBlock:
    """
    A data block storing embedding vectors and metadata, linked via cryptographic hash.
    """
    def __init__(
        self,
        vectors: List[List[float]],
        metadata: List[Dict],
        prev_hash: Optional[str] = None
    ):
        self.vectors = vectors
        self.metadata = metadata
        self.prev_hash = prev_hash
        self.hash = self.compute_hash()

    def compute_hash(self) -> str:
        """
        Compute SHA-256 hash of the block content.
        """
        block_content = {
            "vectors": self.vectors,
            "metadata": self.metadata,
            "prev_hash": self.prev_hash
        }
        block_bytes = json.dumps(block_content, sort_keys=True).encode("utf-8")
        return hashlib.sha256(block_bytes).hexdigest() 