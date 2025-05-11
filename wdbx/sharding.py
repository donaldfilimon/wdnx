"""
sharding.py - Shard management for WDBX cluster using consistent hashing.
"""
import bisect
import hashlib
from typing import Dict, List, Set

class ShardManager:
    """
    Manages shard mapping using consistent hashing.

    Example:
        shards = ['node1', 'node2', 'node3']
        manager = ShardManager(shards)
        node = manager.get_node('some_key')
    """
    def __init__(self, nodes: List[str], replicas: int = 100):
        self.replicas = replicas
        self.ring: List[int] = []
        self.node_map: Dict[int, str] = {}
        # Track shard health: failed nodes
        self.failed_nodes: Set[str] = set()
        for node in nodes:
            for i in range(replicas):
                key = f"{node}:{i}"
                h = int(hashlib.md5(key.encode('utf-8')).hexdigest(), 16)
                self.ring.append(h)
                self.node_map[h] = node
        self.ring.sort()

    def mark_node_failed(self, node: str) -> None:
        """Mark a node as failed (exclude from selection)."""
        self.failed_nodes.add(node)

    def mark_node_healthy(self, node: str) -> None:
        """Mark a node as healthy (include in selection)."""
        self.failed_nodes.discard(node)

    def get_node(self, key: str) -> str:
        """
        Get the healthiest shard node for the given key based on consistent hashing.
        Falls back to next replica on failure.
        """
        if not self.ring:
            raise ValueError("Shard ring is empty")
        h = int(hashlib.md5(key.encode('utf-8')).hexdigest(), 16)
        idx = bisect.bisect(self.ring, h)
        n = len(self.ring)
        # iterate over ring to find first healthy node
        for i in range(n):
            pos = (idx + i) % n
            node_hash = self.ring[pos]
            node = self.node_map[node_hash]
            if node not in self.failed_nodes:
                return node
        # if all nodes failed, return first by default
        node_hash = self.ring[idx % n]
        return self.node_map[node_hash] 