"""
wdbx - Core module for the WDBX vector database and blockchain engine (modular version).

This package provides the main interface and utilities for WDBX integration.
"""

from .async_client import AsyncWDBX, initialize_async_backend
from .blocks import DataBlock
from .chain import BlockChain
from .client import WDBX, initialize_backend
from .download import configure_database, download_file
from .exceptions import (
    AuthenticationError,
    AuthorizationError,
    RateLimitError,
    WDBXError,
)
from .indexing import VectorIndex
from .metrics import start_metrics_server
from .self_update import CodeUpdater
from .self_update_advanced import AdvancedUpdater
from .sharding import ShardManager
from .transactions import TransactionManager

__all__ = [
    "WDBX",
    "AsyncWDBX",
    "initialize_backend",
    "initialize_async_backend",
    "WDBXError",
    "RateLimitError",
    "AuthenticationError",
    "AuthorizationError",
    "start_metrics_server",
    "download_file",
    "configure_database",
    "store_model",
    "load_model",
    "DataBlock",
    "BlockChain",
    "TransactionManager",
    "VectorIndex",
    "ShardManager",
    "CodeUpdater",
    "AdvancedUpdater",
    # Vector utilities
    "normalize",
    "cosine_similarity",
    "bulk_similarity",
    "FaissIndexer",
]

__version__ = "0.1.0"
