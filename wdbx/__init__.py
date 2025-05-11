"""
wdbx - Core module for the WDBX vector database and blockchain engine (modular version).

This package provides the main interface and utilities for WDBX integration.
"""

from .client import WDBX, initialize_backend
from .async_client import AsyncWDBX, initialize_async_backend
from .exceptions import WDBXError, RateLimitError, AuthenticationError, AuthorizationError
from .metrics import start_metrics_server
from .download import download_file, configure_database
from .blocks import DataBlock
from .chain import BlockChain
from .transactions import TransactionManager
from .indexing import VectorIndex
from .sharding import ShardManager
from .self_update import CodeUpdater
from .self_update_advanced import AdvancedUpdater

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
    "normalize", "cosine_similarity", "bulk_similarity", "FaissIndexer",
]

__version__ = "0.1.0" 