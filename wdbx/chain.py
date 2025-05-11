"""
chain.py - Blockchain management for WDBX.
"""

from threading import Lock
from typing import List

from .blocks import DataBlock


class BlockChain:
    """
    Manages a chain of DataBlocks with integrity checks.
    """

    def __init__(self) -> None:
        self.chain: List[DataBlock] = []
        self.lock = Lock()

    def add_block(self, block: DataBlock) -> None:
        """
        Add a new block to the chain after validating previous hash.
        """
        with self.lock:
            if self.chain:
                last = self.chain[-1]
                if block.prev_hash != last.hash:
                    raise ValueError("Invalid previous hash for new block.")
            self.chain.append(block)

    def validate_chain(self) -> bool:
        """
        Validate the entire chain for integrity.
        """
        with self.lock:
            for i in range(1, len(self.chain)):
                prev = self.chain[i - 1]
                curr = self.chain[i]
                if curr.prev_hash != prev.hash or curr.hash != curr.compute_hash():
                    return False
        return True
