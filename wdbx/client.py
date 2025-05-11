"""
client.py - Main WDBX client interface for the wdbx package.
"""

import asyncio
import atexit
import importlib.metadata
import logging
import threading
import time
from contextlib import contextmanager
from typing import Any, Coroutine, Dict, Iterator, List, Optional, Tuple, Type, Union

from cachetools import LRUCache
from pybreaker import CircuitBreakerError

from .discovery import NodeDiscovery
from .exceptions import WDBXError
from .metrics import RPC_CALL_COUNT, RPC_CALL_LATENCY
from .self_update_advanced import AdvancedUpdater
from .utils import CIRCUIT_BREAKER, with_retry

# Placeholder for the actual underlying asynchronous client.
# In a real scenario, this would be a concrete class import.
# from wdbx_engine import AsyncClient as _WDBXClient # Example
_UnderlyingAsyncClient = Any

logger = logging.getLogger(__name__)


class WDBX:
    """
    Synchronous interface for the WDBX client.

    This class wraps an underlying asynchronous WDBX client, providing a blocking,
    optionally thread-safe API. All async methods from the underlying client are run
    in a managed event loop and their results are returned synchronously.

    It supports features like vector storage, search, metadata management, blockchain operations,
    sharding, auto-discovery, caching, and self-update capabilities.

    Args:
        vector_dimension: The dimension of the vectors to be stored.
        enable_plugins: Whether to enable plugin loading (default True).
        thread_safe: If True, a lock will be used around asyncio.run calls (default False).
        shards: Optional dictionary defining shard configurations.
        auto_discovery: If True, attempt to discover other WDBX nodes on the network (default False).
        discovery_port: Port used for node discovery (default 9999).
        rpc_port: The RPC port this node/client expects services to be on if discovered (default None).
        cache_size: Maximum number of items in the LRU search cache (default 128).
        **kwargs: Additional keyword arguments passed to the underlying WDBX client constructor.
    """

    _client: _UnderlyingAsyncClient
    _search_cache: LRUCache[Tuple[Tuple[Union[float, int], ...], int], List[Dict[str, Any]]]
    auto_discovery: bool
    discovery_port: int
    rpc_port: Optional[int]
    discovery: Optional[NodeDiscovery]
    _thread_safe: bool
    _lock: Optional[threading.Lock]
    vector_dimension: int
    shards: Dict[str, Dict[str, Any]]
    shard_manager: Optional[Any]  # Replace Any with actual ShardManager type if available
    shard_clients: Dict[str, _UnderlyingAsyncClient]
    _shard_health: Dict[str, bool]
    wdbx_repo_url: Optional[str]  # For self-update features

    def __init__(
        self,
        vector_dimension: int,
        enable_plugins: bool = True,
        thread_safe: bool = False,
        shards: Optional[Dict[str, Dict[str, Any]]] = None,
        auto_discovery: bool = False,
        discovery_port: int = 9999,
        rpc_port: Optional[int] = None,
        cache_size: int = 128,
        wdbx_repo_url: Optional[str] = None,  # For self-update
        **kwargs: Any,
    ):
        """
        Initializes the WDBX synchronous client.

        This involves setting up caching, discovery, sharding (if configured),
        and the underlying asynchronous client.
        """
        logger.info(f"Initializing WDBX client with vector_dimension={vector_dimension}, enable_plugins={enable_plugins}")
        self._search_cache = LRUCache(maxsize=cache_size)
        self.auto_discovery = auto_discovery
        self.discovery_port = discovery_port
        self.rpc_port = rpc_port
        self.discovery = None
        self.wdbx_repo_url = wdbx_repo_url

        # This is a placeholder. In a real app, you'd import the actual async client class.
        # For example: from wdbx.async_internal import AsyncInternalClient
        # For now, we assign Any to self._client and assume it has the necessary async methods.
        ActualAsyncClient: Type[_UnderlyingAsyncClient]
        try:
            # Attempt to import a hypothetical internal async client
            from wdbx.async_internal import (
                AsyncInternalClient as ConcreteAsyncClient,  # type: ignore
            )

            ActualAsyncClient = ConcreteAsyncClient
            logger.info("Successfully loaded ConcreteAsyncClient (AsyncInternalClient).")
        except ImportError:
            logger.warning("Concrete underlying async WDBX client (AsyncInternalClient) not found. " "Using a generic placeholder. Client functionality will be impaired.")
            ActualAsyncClient = lambda **_kwargs: object()  # A callable that returns a dummy object

        self._client = ActualAsyncClient(
            vector_dimension=vector_dimension,
            enable_plugins=enable_plugins,  # Assuming async client also handles this
            **kwargs,
        )

        if enable_plugins:
            logger.info("Plugin support enabled. Loading entrypoint plugins.")
            # Initialize plugin registry and load plugins
            self._plugin_registry: Dict[str, Dict[str, Any]] = {}
            self._load_entrypoint_plugins()

        self._thread_safe = thread_safe
        self._lock = threading.Lock() if thread_safe else None
        self.vector_dimension = vector_dimension

        self.shards = shards or {}
        self.shard_manager = None
        self.shard_clients = {}

        if self.shards:
            logger.info(f"Sharding configured with nodes: {list(self.shards.keys())}")
            try:
                from .sharding import ShardManager

                self.shard_manager = ShardManager(list(self.shards.keys()))
                for node, cfg in self.shards.items():
                    shard_client_instance = ActualAsyncClient(**cfg)
                    # Ensure initialize is an async method on ActualAsyncClient
                    if hasattr(shard_client_instance, "initialize") and asyncio.iscoroutinefunction(shard_client_instance.initialize):
                        asyncio.run(shard_client_instance.initialize())
                    self.shard_clients[node] = shard_client_instance
            except ImportError:
                logging.error("ShardManager not found. Sharding disabled.")
                logger.error("ShardManager not found. Sharding disabled.")
            except Exception as e:
                logging.error(f"Error initializing shards: {e}. Sharding may be impaired.")
                logger.error(f"Error initializing shards: {e}. Sharding may be impaired.")

        elif self.auto_discovery:
            logger.info(f"Auto-discovery enabled. Discovery port: {self.discovery_port}, RPC port: {self.rpc_port}")
            try:
                self.discovery = NodeDiscovery(service_port=self.rpc_port or 0, discovery_port=self.discovery_port)
                self.discovery.start()
                logger.info("Node discovery service started. Waiting for peers...")
                time.sleep(2)
                peers = self.discovery.get_peers()
                logger.info(f"Discovered peers: {peers}")
                discovered_shards: Dict[str, Dict[str, Any]] = {}
                for ip, port_val in peers:
                    node_name = f"{ip}:{port_val}"
                    cfg = {
                        "vector_dimension": vector_dimension,
                        "enable_plugins": enable_plugins,
                        "host": ip,
                        "port": port_val,
                    }
                    discovered_shards[node_name] = cfg

                if discovered_shards:
                    self.shards = discovered_shards
                    from .sharding import ShardManager

                    self.shard_manager = ShardManager(list(self.shards.keys()))
                    for node, cfg in self.shards.items():
                        shard_client_instance = ActualAsyncClient(**cfg)
                        if hasattr(shard_client_instance, "initialize") and asyncio.iscoroutinefunction(shard_client_instance.initialize):
                            asyncio.run(shard_client_instance.initialize())
                        self.shard_clients[node] = shard_client_instance
            except ImportError:
                logging.error("ShardManager or NodeDiscovery not found for auto-discovery. Sharding disabled.")
                logger.error("ShardManager or NodeDiscovery not found for auto-discovery. Sharding disabled.")
            except Exception as e:
                logging.error(f"Error during auto-discovery/sharding setup: {e}. Sharding may be impaired.")
                logger.error(f"Error during auto-discovery/sharding setup: {e}. Sharding may be impaired.")

        self._shard_health = {node: True for node in self.shard_clients}
        atexit.register(self.shutdown)
        logger.info("WDBX client initialization complete.")

    def __enter__(self) -> "WDBX":
        """
        Enter the runtime context for the WDBX client.
        Currently, this just returns the client instance. Initialization should be explicit.
        """
        # self.initialize() # Typically, initialize() is called explicitly after __init__.
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],  # Actually TracebackType, but Any for simplicity if not importing `types`
    ) -> None:
        """
        Exit the runtime context, ensuring the client is properly shut down.
        This is critical for releasing resources like network connections or background threads.
        """
        self.shutdown()

    @CIRCUIT_BREAKER
    @with_retry
    def _run_async(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """
        Executes an awaitable coroutine synchronously, handling retries and circuit breaking.

        This is the core mechanism for converting async calls from the underlying
        client into synchronous calls for this wrapper class.

        Args:
            coro: The coroutine to execute.

        Returns:
            The result of the coroutine's execution.

        Raises:
            WDBXError: If the circuit breaker is open, or for other general WDBX errors.
            ConnectionError: If a connection-related issue occurs (conceptual, depends on exceptions from `coro`).
            TimeoutError: If the operation times out (conceptual, depends on exceptions from `coro`).
            Exception: Propagates other unexpected exceptions from the coroutine.
        """
        method_name = getattr(coro, "__name__", "unknown_coroutine")
        logger.debug(f"Attempting to run async method: {method_name}")
        RPC_CALL_COUNT.labels(method=method_name).inc()
        with RPC_CALL_LATENCY.labels(method=method_name).time():
            try:
                if self._thread_safe and self._lock:
                    with self._lock:
                        logger.debug(f"Executing async method '{method_name}' with lock.")
                        result = asyncio.run(coro)
                else:
                    logger.debug(f"Executing async method '{method_name}' without lock.")
                    result = asyncio.run(coro)
                logger.debug(f"Async method '{method_name}' completed successfully.")
                return result
            except CircuitBreakerError as cb_err:
                logger.error(f"Circuit breaker open for '{method_name}'. Call aborted.")
                raise WDBXError(f"Circuit breaker for '{method_name}' is open. The service may be temporarily unavailable or overloaded.") from cb_err
            # Example of how more specific error handling could be added if the underlying client raises them:
            # except SomeUnderlyingConnectionError as conn_err:
            #     logger.error(f"Connection error during '{method_name}': {conn_err}")
            #     raise ConnectionError(f"Failed to connect during '{method_name}'. Please check network and server status.") from conn_err
            # except SomeUnderlyingTimeoutError as timeout_err:
            #     logger.error(f"Timeout during '{method_name}': {timeout_err}")
            #     raise TimeoutError(f"Operation '{method_name}' timed out. The server might be busy or unresponsive.") from timeout_err
            except Exception as e:
                logger.exception(f"Unexpected error in async operation '{method_name}'")
                raise WDBXError(f"Async operation '{method_name}' failed with an unexpected error: {e}") from e

    def initialize(self) -> None:
        """
        Initializes the WDBX backend and its resources.
        This typically involves establishing connections to the database or backend services,
        and preparing the client for operations. It should be called before other methods.

        Raises:
            WDBXError: If initialization fails.
        """
        logger.info("Initializing WDBX backend...")
        if not hasattr(self._client, "initialize") or not asyncio.iscoroutinefunction(self._client.initialize):
            logger.warning("Underlying client does not have a valid async initialize method. Skipping actual backend initialization.")
            return
        self._run_async(self._client.initialize())
        logger.info("WDBX backend initialized successfully.")

    def shutdown(self) -> None:
        """
        Shuts down the WDBX backend and releases all associated resources.
        This includes closing network connections, stopping discovery services, and cleaning up.
        It's crucial to call this for a graceful exit, often via `atexit` or a context manager.
        """
        logger.info("Shutting down WDBX client and backend resources...")
        if self.auto_discovery and self.discovery:
            try:
                logger.info("Stopping node discovery service...")
                self.discovery.stop()
                logger.info("Node discovery service stopped.")
            except Exception as e:
                logger.warning(f"Error stopping discovery service during shutdown: {e}")

        if hasattr(self._client, "shutdown") and asyncio.iscoroutinefunction(self._client.shutdown):
            try:
                logger.info("Shutting down main WDBX backend client...")
                self._run_async(self._client.shutdown())
                logger.info("Main WDBX backend client shut down.")
            except Exception as e:
                logger.warning(f"Error during WDBX client shutdown: {e}. Attempting to proceed.")
        else:
            logger.warning("Underlying client does not have a valid async shutdown method.")

        # Shutdown shard clients if they exist
        if self.shard_clients:
            logger.info(f"Shutting down {len(self.shard_clients)} shard client(s)...")
            for node_name, shard_client_instance in self.shard_clients.items():
                if hasattr(shard_client_instance, "shutdown") and asyncio.iscoroutinefunction(shard_client_instance.shutdown):
                    try:
                        logger.info(f"Shutting down shard client: {node_name}")
                        asyncio.run(shard_client_instance.shutdown())
                        logger.info(f"Shard client {node_name} shut down.")
                    except Exception as e:
                        logger.warning(f"Error shutting down shard client {node_name}: {e}")
            self.shard_clients.clear()
            logger.info("All shard clients processed for shutdown.")
        logger.info("WDBX client shutdown process complete.")

    def store(self, vector: List[float], metadata: Dict[str, Any]) -> int:
        """
        Store a vector and its associated metadata in the database.

        Args:
            vector: A list of floats representing the vector.
            metadata: A dictionary containing metadata associated with the vector.

        Returns:
            The unique ID assigned to the stored vector.

        Raises:
            TypeError: If `vector` is not a list of numbers or `metadata` is not a dict.
            NotImplementedError: If the underlying client does not support the required async method.
            WDBXError: If the storage operation fails for other reasons.
        """
        if not isinstance(vector, list) or not all(isinstance(x, (float, int)) for x in vector):
            raise TypeError("Input vector must be a list of numbers (floats or ints).")
        if not isinstance(metadata, dict):
            raise TypeError("Input metadata must be a dictionary.")

        logger.debug(f"Storing vector (size: {len(vector)}) with metadata (keys: {list(metadata.keys())})")
        if not hasattr(self._client, "vector_store_async") or not asyncio.iscoroutinefunction(getattr(self._client, "vector_store_async")):
            logger.error("Underlying client does not support async method 'vector_store_async'.")
            raise NotImplementedError("Underlying client does not support async method 'vector_store_async'.")
        result_id = self._run_async(self._client.vector_store_async(vector, metadata))
        logger.info(f"Stored vector with ID: {result_id}")
        return result_id

    def search(self, vector: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for vectors similar to the query vector, using an LRU cache for hot queries.

        Args:
            vector: The query vector (list of floats).
            limit: The maximum number of similar vectors to return (default 10).

        Returns:
            A list of dictionaries, where each dictionary represents a search result
            and typically contains keys like 'id', 'score', and 'metadata'.

        Raises:
            TypeError: If `vector` is not a list of numbers.
            ValueError: If `limit` is not a positive integer.
            NotImplementedError: If the underlying client does not support the required async method.
            WDBXError: If the search operation fails.
        """
        if not isinstance(vector, list) or not all(isinstance(x, (float, int)) for x in vector):
            raise TypeError("Query vector must be a list of numbers (floats or ints).")
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("Search limit must be a positive integer.")

        logger.debug(f"Searching for vector (size: {len(vector)}), limit: {limit}")
        cache_key: Tuple[Tuple[Union[float, int], ...], int] = (tuple(vector), limit)

        cached_result: Optional[List[Dict[str, Any]]] = None
        if self._thread_safe and self._lock:
            with self._lock:
                cached_result = self._search_cache.get(cache_key)
        else:
            cached_result = self._search_cache.get(cache_key)

        if cached_result is not None:
            logger.debug(f"Search cache hit for vector (size: {len(vector)}), limit: {limit}")
            return cached_result

        logger.debug(f"Search cache miss for vector (size: {len(vector)}), limit: {limit}. Querying backend.")
        if not hasattr(self._client, "vector_search_async") or not asyncio.iscoroutinefunction(getattr(self._client, "vector_search_async")):
            logger.error("Underlying client does not support async method 'vector_search_async'.")
            raise NotImplementedError("Underlying client does not support async method 'vector_search_async'.")

        result: List[Dict[str, Any]] = self._run_async(self._client.vector_search_async(vector, limit=limit))

        if self._thread_safe and self._lock:
            with self._lock:
                self._search_cache[cache_key] = result
        else:
            self._search_cache[cache_key] = result
        logger.info(f"Search returned {len(result)} results for vector (size: {len(vector)}), limit: {limit}")
        return result

    def bulk_store(self, items: List[Tuple[List[float], Dict[str, Any]]]) -> List[int]:
        """
        Store multiple vectors and their metadata in bulk.
        Note: This default implementation iterates and calls `store` for each item
              if a native `bulk_vector_store_async` is not available on the underlying client.

        Args:
            items: A list of tuples, where each tuple contains a vector (list of floats)
                   and its metadata (dictionary).

        Returns:
            A list of unique IDs assigned to the stored vectors, in the same order as input.

        Raises:
            TypeError: If input `items` is not a list of (vector, metadata) tuples, or if
                       any vector/metadata within the tuples has an incorrect type.
            NotImplementedError: If the underlying client does not support the required async method
                               (and no fallback is used, though this impl has a fallback).
            WDBXError: If any part of the bulk storage operation fails.
        """
        if not isinstance(items, list):
            raise TypeError("Input 'items' must be a list.")
        if not all(isinstance(item, tuple) and len(item) == 2 for item in items):
            raise TypeError("Each item in 'items' must be a (vector, metadata) tuple.")

        logger.info(f"Bulk storing {len(items)} items.")
        # Consider if the underlying client has a more efficient bulk_store_async method
        # If not, this list comprehension is a simple synchronous loop over the async store.
        # For true async bulk, the underlying client would need to support it.
        results: List[int] = []
        for v_idx, (v, m) in enumerate(items):
            if not isinstance(v, list) or not all(isinstance(x, (float, int)) for x in v):
                raise TypeError(f"Vector at index {v_idx} must be a list of numbers.")
            if not isinstance(m, dict):
                raise TypeError(f"Metadata at index {v_idx} must be a dictionary.")
            results.append(self.store(v, m))
        logger.info(f"Bulk store completed for {len(items)} items. Result IDs: {results}")
        return results

    def bulk_search(self, vectors: List[List[float]], limit: int = 10) -> List[List[Dict[str, Any]]]:
        """
        Search for multiple query vectors in bulk.
        Note: This default implementation iterates and calls `search` for each query vector
              if a native `bulk_vector_search_async` is not available on the underlying client.

        Args:
            vectors: A list of query vectors (each a list of floats).
            limit: The maximum number of results to return for each query vector (default 10).

        Returns:
            A list of lists of search results. Each inner list corresponds to a query vector.

        Raises:
            TypeError: If input `vectors` is not a list of lists of numbers, or if any individual vector is invalid.
            ValueError: If `limit` is not a positive integer.
            NotImplementedError: If the underlying client does not support the required async method
                               (and no fallback is used, though this impl has a fallback).
            WDBXError: If any part of the bulk search operation fails.
        """
        if not isinstance(vectors, list) or not all(isinstance(v, list) for v in vectors):
            raise TypeError("Vectors must be a list of lists of numbers.")
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("Search limit must be a positive integer.")

        logger.info(f"Bulk searching for {len(vectors)} query vectors with limit {limit}.")
        # Similar to bulk_store, check if self._client has bulk_vector_search_async
        # If self._client.bulk_vector_search_async(vectors, limit=limit) exists:
        # return self._run_async(self._client.bulk_vector_search_async(vectors, limit=limit))

        # Fallback to iterative calls
        results: List[List[Dict[str, Any]]] = []
        for v_idx, v_query in enumerate(vectors):
            if not all(isinstance(x, (float, int)) for x in v_query):
                raise TypeError(f"Query vector at index {v_idx} must be a list of numbers.")
            results.append(self.search(v_query, limit))
        logger.info(f"Bulk search completed for {len(vectors)} query vectors.")
        return results

    def delete(self, vector_id: int) -> bool:
        """
        Delete a stored vector by its ID.

        Args:
            vector_id: The ID of the vector to delete.

        Returns:
            True if the deletion was successful or if the vector ID did not exist (providing idempotent behavior).
            False if the deletion specifically failed for an existing vector (this is backend-dependent).

        Raises:
            TypeError: If `vector_id` is not an integer.
            NotImplementedError: If the underlying client does not support the required async method.
            WDBXError: If the deletion operation fails for other reasons.
        """
        if not isinstance(vector_id, int):
            raise TypeError("Vector ID must be an integer.")

        logger.info(f"Deleting vector with ID: {vector_id}")
        deleted = self._run_async(self._client.vector_delete_async(vector_id))  # type: ignore
        if deleted:
            logger.info(f"Successfully deleted vector ID: {vector_id}")
        else:
            logger.warning(f"Failed to delete vector ID: {vector_id} or vector not found.")
        return deleted

    def update_metadata(self, vector_id: int, metadata: Dict[str, Any]) -> bool:
        """
        Update the metadata for a stored vector.
        The exact behavior (e.g., complete replacement vs. merge) of the metadata update
        depends on the backend implementation.

        Args:
            vector_id: The ID of the vector whose metadata is to be updated.
            metadata: A dictionary containing the new or updated metadata fields.

        Returns:
            True if the update was successful.
            False if the vector ID was not found or the update failed for other reasons (backend-dependent).

        Raises:
            TypeError: If `vector_id` is not an integer or `metadata` is not a dictionary.
            NotImplementedError: If the underlying client does not support the required async method.
            WDBXError: If the metadata update operation fails for other reasons.
        """
        if not isinstance(vector_id, int):
            raise TypeError("Vector ID must be an integer.")
        if not isinstance(metadata, dict):
            raise TypeError("Metadata must be a dictionary.")

        logger.info(f"Updating metadata for vector ID: {vector_id} with keys: {list(metadata.keys())}")
        updated = self._run_async(self._client.vector_update_metadata_async(vector_id, metadata))  # type: ignore
        if updated:
            logger.info(f"Successfully updated metadata for vector ID: {vector_id}")
        else:
            logger.warning(f"Failed to update metadata for vector ID: {vector_id}")
        return updated

    def get_metadata(self, vector_id: int) -> Optional[Dict[str, Any]]:
        """
        Get the metadata for a specific vector by its ID.

        Args:
            vector_id: The ID of the vector.

        Returns:
            A dictionary containing the vector's metadata if the vector is found, otherwise None.

        Raises:
            TypeError: If `vector_id` is not an integer.
            NotImplementedError: If the underlying client does not support the required async method.
            WDBXError: If the operation to retrieve metadata fails for reasons other than the vector not being found.
        """
        if not isinstance(vector_id, int):
            raise TypeError("Vector ID must be an integer.")

        logger.debug(f"Fetching metadata for vector ID: {vector_id}")
        try:
            metadata = self._run_async(self._client.vector_get_metadata_async(vector_id))  # type: ignore
            if metadata is not None:
                logger.info(f"Successfully fetched metadata for vector ID: {vector_id}")
            else:
                logger.info(f"No metadata found for vector ID: {vector_id}")
            return metadata
        except WDBXError as e:
            # Check if the error indicates "not found". This is backend-dependent.
            # For now, re-raise unless a more specific error handling for "not found" is implemented.
            # if "not found" in str(e).lower(): # Example, very fragile
            #     logger.info(f"No metadata found for vector ID: {vector_id} (error indicates not found)")
            #     return None
            logger.error(f"Error fetching metadata for vector ID {vector_id}: {e}")
            raise

    def count(self) -> int:
        """
        Return the total number of vectors stored in the database.

        Returns:
            The total count of vectors.

        Raises:
            NotImplementedError: If the underlying client does not support the required async method.
            WDBXError: If the count operation fails.
        """
        logger.debug("Fetching total vector count.")
        total_count = self._run_async(self._client.vector_count_async())  # type: ignore
        logger.info(f"Total vector count: {total_count}")
        return total_count

    def ping(self) -> bool:
        """
        Ping the backend server to check if it's alive and responsive.

        Returns:
            True if the server responds positively (e.g., an HTTP 200 or equivalent),
            False if the server responds negatively or does not respond.

        Raises:
            NotImplementedError: If the underlying client does not support a ping method.
            WDBXError: If the ping attempt itself encounters a network error or unexpected server response.
        """
        logger.debug("Pinging WDBX backend...")
        is_alive = self._run_async(self._client.ping())  # type: ignore
        logger.info(f"Ping result: {'alive' if is_alive else 'unreachable'}")
        return is_alive

    def flush(self) -> None:
        """
        Remove ALL vectors and their associated data from the backend.
        This is a destructive operation and should be used with extreme caution.

        Raises:
            NotImplementedError: If the underlying client does not support the required async method.
            WDBXError: If the flush operation fails.
        """
        logger.warning("Executing FLUSH operation. This will remove ALL vectors and data.")
        self._run_async(self._client.vector_flush_async())  # type: ignore
        logger.info("Flush operation completed.")

    # --- Blockchain specific methods ---
    def create_block(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new block in the blockchain with the given metadata.

        Args:
            metadata: A dictionary containing metadata for the new block.

        Returns:
            A dictionary representing the created block, typically including its ID and hash.

        Raises:
            TypeError: If `metadata` is not a dictionary.
            NotImplementedError: If the underlying client does not support the required async method.
            WDBXError: If block creation fails.
        """
        if not isinstance(metadata, dict):
            raise TypeError("Block metadata must be a dictionary.")

        logger.info(f"Creating blockchain block with metadata keys: {list(metadata.keys())}")
        block_info = self._run_async(self._client.block_create_async(metadata))  # type: ignore
        logger.info(f"Blockchain block created successfully: {block_info.get('id', 'N/A')}")
        return block_info

    def validate_block(self, block_id: Union[int, str]) -> bool:
        """
        Validate the integrity of a block in the blockchain by its ID or hash.

        Args:
            block_id: The ID (integer) or hash (string) of the block to validate.

        Returns:
            True if the block is valid, False otherwise.

        Raises:
            TypeError: If `block_id` is not an int or str.
            NotImplementedError: If the underlying client does not support the required async method.
            WDBXError: If the validation process encounters an error.
        """
        if not isinstance(block_id, (int, str)):
            raise TypeError("Block ID must be an integer or string hash.")

        logger.info(f"Validating blockchain block: {block_id}")
        is_valid = self._run_async(self._client.block_validate_async(block_id))  # type: ignore
        logger.info(f"Block {block_id} validation result: {'valid' if is_valid else 'invalid'}")
        return is_valid

    def list_blocks(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        List blocks stored in the blockchain, with pagination.

        Args:
            limit: Maximum number of blocks to return (default 100).
            offset: Number of blocks to skip from the beginning (default 0).

        Returns:
            A list of dictionaries, where each dictionary represents a block.

        Raises:
            ValueError: If `limit` or `offset` are negative.
            NotImplementedError: If the underlying client does not support the required async method.
            WDBXError: If listing blocks fails.
        """
        if not isinstance(limit, int) or limit < 0:
            raise ValueError("Limit must be a non-negative integer.")
        if not isinstance(offset, int) or offset < 0:
            raise ValueError("Offset must be a non-negative integer.")

        logger.info(f"Listing blockchain blocks with limit={limit}, offset={offset}")
        if not hasattr(self._client, "block_list_async") or not asyncio.iscoroutinefunction(getattr(self._client, "block_list_async")):
            logger.error("Underlying client does not support async method 'block_list_async'.")
            raise NotImplementedError("Underlying client does not support async method 'block_list_async'.")
        blocks = self._run_async(self._client.block_list_async(limit=limit, offset=offset))
        logger.info(f"Retrieved {len(blocks)} blockchain blocks.")
        return blocks

    def get_block(self, block_id: Union[int, str]) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific block from the blockchain by its ID or hash.

        Args:
            block_id: The ID (integer) or hash (string) of the block.

        Returns:
            A dictionary representing the block if found, otherwise None.

        Raises:
            TypeError: If `block_id` is not an int or str.
            NotImplementedError: If the underlying client does not support the required async method.
            WDBXError: If retrieving the block fails for reasons other than not found.
        """
        if not isinstance(block_id, (int, str)):
            raise TypeError("Block ID must be an integer or string hash.")

        logger.info(f"Getting blockchain block with ID/hash: {block_id}")
        if not hasattr(self._client, "block_get_async") or not asyncio.iscoroutinefunction(getattr(self._client, "block_get_async")):
            logger.error("Underlying client does not support async method 'block_get_async'.")
            raise NotImplementedError("Underlying client does not support async method 'block_get_async'.")
        try:
            block_data = self._run_async(self._client.block_get_async(block_id))
            if block_data:
                logger.info(f"Successfully retrieved block: {block_id}")
            else:
                logger.info(f"Block not found: {block_id}")
            return block_data  # Could be None if underlying client returns None for not found
        except WDBXError as e:
            error_message_lower = str(e).lower()
            if "not found" in error_message_lower or "does not exist" in error_message_lower or "no such block" in error_message_lower:
                logger.info(f"Block not found (via WDBXError): {block_id}")
                return None
            logger.error(f"Error retrieving block {block_id}: {e}", exc_info=True)
            raise

    # --- Transaction specific methods ---
    def begin_transaction(self) -> Any:
        """
        Begin a new transaction with the backend.
        The nature of the returned transaction identifier is backend-dependent.

        Returns:
            A transaction identifier or context, specific to the backend.

        Raises:
            NotImplementedError: If the underlying client does not support the required async method.
            WDBXError: If beginning the transaction fails.
        """
        logger.info("Attempting to begin transaction...")
        if not hasattr(self._client, "transaction_begin_async") or not asyncio.iscoroutinefunction(getattr(self._client, "transaction_begin_async")):
            logger.error("Underlying client does not support async method 'transaction_begin_async'.")
            raise NotImplementedError("Underlying client does not support async method 'transaction_begin_async'.")
        tx_id = self._run_async(self._client.transaction_begin_async())
        logger.info(f"Transaction begun successfully. Transaction ID/Context: {tx_id}")
        return tx_id

    def commit_transaction(self, transaction_id: Any = None) -> None:
        """
        Commit the currently active transaction.

        Args:
            transaction_id: Optional transaction identifier, if required by the backend
                              to specify which transaction to commit. Often implicit.

        Raises:
            NotImplementedError: If the underlying client does not support the required async method.
            WDBXError: If committing the transaction fails.
        """
        logger.info(f"Attempting to commit transaction: {transaction_id or 'current/implicit'}")
        if not hasattr(self._client, "transaction_commit_async") or not asyncio.iscoroutinefunction(getattr(self._client, "transaction_commit_async")):
            logger.error("Underlying client does not support async method 'transaction_commit_async'.")
            raise NotImplementedError("Underlying client does not support async method 'transaction_commit_async'.")
        if transaction_id:
            self._run_async(self._client.transaction_commit_async(transaction_id=transaction_id))
        else:
            self._run_async(self._client.transaction_commit_async())
        logger.info(f"Transaction committed successfully: {transaction_id or 'current/implicit'}")

    def rollback_transaction(self, transaction_id: Any = None) -> None:
        """
        Roll back the currently active transaction.

        Args:
            transaction_id: Optional transaction identifier, if required by the backend.

        Raises:
            NotImplementedError: If the underlying client does not support the required async method.
            WDBXError: If rolling back the transaction fails.
        """
        logger.info(f"Attempting to roll back transaction: {transaction_id or 'current/implicit'}")
        if not hasattr(self._client, "transaction_rollback_async") or not asyncio.iscoroutinefunction(getattr(self._client, "transaction_rollback_async")):
            logger.error("Underlying client does not support async method 'transaction_rollback_async'.")
            raise NotImplementedError("Underlying client does not support async method 'transaction_rollback_async'.")
        if transaction_id:
            self._run_async(self._client.transaction_rollback_async(transaction_id=transaction_id))
        else:
            self._run_async(self._client.transaction_rollback_async())
        logger.info(f"Transaction rolled back: {transaction_id or 'current/implicit'}")

    @contextmanager
    def transaction(self) -> Iterator["WDBX"]:
        """
        Provides a context manager for database transactions.
        Ensures that `begin_transaction` is called on entry, `commit_transaction`
        on successful exit, and `rollback_transaction` if an exception occurs.

        Usage:
            with client.transaction() as tx_client:
                tx_client.store(...)
                tx_client.update_metadata(...)

        Yields:
            The client instance itself, to be used within the transaction context.

        Raises:
            WDBXError: If starting, committing, or rolling back the transaction fails.
        """
        logger.info("Entering transaction context...")  # Changed from "Beginning transaction..."
        tx_context = self.begin_transaction()  # tx_context might be used by commit/rollback if backend needs it
        try:
            yield self
            logger.info(f"Committing transaction (context: {tx_context})...")
            self.commit_transaction(transaction_id=tx_context if tx_context is not None else None)  # Pass if not None
            logger.info(f"Transaction (context: {tx_context}) committed successfully.")
        except Exception as e:
            logger.error(
                f"Exception occurred within transaction context (context: {tx_context}): {e}. Rolling back...",
                exc_info=True,
            )
            self.rollback_transaction(transaction_id=tx_context if tx_context is not None else None)  # Pass if not None
            logger.info(f"Transaction (context: {tx_context}) rolled back.")
            raise

    def schedule_self_update(self, interval: int, repo_dir: str, module_paths: Optional[List[str]] = None) -> None:
        """
        Schedule periodic Git-based code updates for specified modules.

        Args:
            interval: The update interval in seconds.
            repo_dir: The local directory of the Git repository.
            module_paths: An optional list of relative paths to Python modules/files within
                          the repo to be updated. If None, implies updating the whole repo
                          or a predefined set of modules.
        """
        logger.info(f"Scheduling self-update from Git every {interval}s. Repo: {repo_dir}, Modules: {module_paths}")
        # AdvancedUpdater is instantiated directly, its methods are synchronous
        adv = AdvancedUpdater(repo_url=getattr(self, "wdbx_repo_url", None))  # Assuming repo_url can be an attribute
        adv.schedule(interval, adv.update_from_git, repo_dir, module_paths)

    def stop_self_update(self) -> None:
        """
        Stop any scheduled self-update tasks.
        """
        logger.info("Stopping scheduled self-update tasks.")
        AdvancedUpdater().stop()  # Assumes AdvancedUpdater manages its singleton/state correctly

    def git_update(self, local_dir: str, module_paths: Optional[List[str]] = None) -> None:
        """
        Perform an immediate Git pull from the configured repository and apply patches for modules.

        Args:
            local_dir: The local directory of the Git repository.
            module_paths: Optional list of module paths to update after pulling.
        """
        logger.info(f"Performing immediate Git update. Local dir: {local_dir}, Modules: {module_paths}")
        adv = AdvancedUpdater(repo_url=getattr(self, "wdbx_repo_url", None))
        adv.update_from_git(local_dir, module_paths)
        logger.info("Git update process completed.")

    def rollback_update(self, file_path: str, backup_file: Optional[str] = None) -> None:
        """
        Roll back a patched file to its previous backup.

        Args:
            file_path: The path to the file to roll back.
            backup_file: Optional path to a specific backup file to restore from.
                         If None, AdvancedUpdater uses its default backup mechanism.
        """
        logger.info(f"Rolling back update for file: {file_path}. Backup: {backup_file or 'latest'}")
        AdvancedUpdater().rollback(file_path, backup_file)
        logger.info(f"Rollback for {file_path} completed.")

    def ai_update(
        self,
        file_path: str,
        instruction: str,
        model_name: str = "gpt2",
        backend: str = "pt",
        memory_limit: int = 5,
    ) -> None:
        """
        Use an AI agent to generate and apply a code patch to a file based on an instruction.

        Args:
            file_path: The path to the file to be updated.
            instruction: A natural language instruction describing the desired change.
            model_name: The name of the AI model to use for generating the patch (default "gpt2").
            backend: The AI backend to use (e.g., "pt" for PyTorch, "tf" for TensorFlow) (default "pt").
            memory_limit: Memory limit for the AI model (implementation-specific) (default 5).
        """
        logger.info(f"Performing AI-driven update for file: {file_path}. Instruction: '{instruction[:50]}...'")
        adv = AdvancedUpdater(repo_url=getattr(self, "wdbx_repo_url", None))
        adv.ai_update(
            file_path,
            instruction,
            model_name=model_name,
            backend=backend,
            memory_limit=memory_limit,
        )
        logger.info(f"AI update for {file_path} completed.")

    def _load_entrypoint_plugins(self) -> None:
        """
        Discover and load plugins registered via entry points, maintaining a registry
        and invoking lifecycle hooks (wdbx_plugin_load, on_enable).
        """
        # Load synchronous plugins
        for entry_point in importlib.metadata.entry_points().select(group="wdbx.plugins"):
            name = entry_point.name
            info: Dict[str, Any] = {}
            try:
                plugin_module = entry_point.load()
                info = getattr(plugin_module, "PLUGIN_INFO", {})
                if not hasattr(plugin_module, "wdbx_plugin_load"):
                    logger.warning(f"Plugin {name} missing 'wdbx_plugin_load'. Skipping.")
                    self._plugin_registry[name] = {
                        "module": plugin_module,
                        "info": info,
                        "status": "error",
                        "error": "Missing load hook",
                    }
                    continue
                logger.info(f"Loading plugin: {name} v{info.get('version','?')} by {info.get('author','?')}")
                plugin_module.wdbx_plugin_load(self)
                if hasattr(plugin_module, "on_enable"):
                    plugin_module.on_enable(self)
                self._plugin_registry[name] = {
                    "module": plugin_module,
                    "info": info,
                    "status": "enabled",
                }
                logger.info(f"Successfully loaded plugin: {name}")
            except Exception as e:
                logger.error(f"Failed to load plugin {name}: {e}", exc_info=True)
                self._plugin_registry[name] = {
                    "module": None,
                    "info": info,
                    "status": "error",
                    "error": str(e),
                }
        # Optionally load async plugins
        if hasattr(self._client, "load_async_plugin"):
            logger.info("Loading async WDBX plugins...")
            for entry_point in importlib.metadata.entry_points().select(group="wdbx.async_plugins"):
                name = entry_point.name
                try:
                    self._run_async(self._client.load_async_plugin(entry_point))  # type: ignore
                    logger.info(f"Successfully loaded async plugin: {name}")
                except Exception as e:
                    logger.error(f"Failed to load async plugin {name}: {e}", exc_info=True)

    def check_shards_health(self) -> Dict[str, bool]:
        """
        Check the health status of all configured shard clients.
        Updates internal health status and returns the current snapshot.

        Returns:
            A dictionary mapping shard node names to their health status (True for healthy, False for unhealthy).
        """
        if not self.shard_clients:
            logger.debug("No shard clients configured, skipping health check.")
            return {}

        logger.info("Checking health of configured shard clients...")
        current_health: Dict[str, bool] = {}
        for node_name, shard_client in self.shard_clients.items():
            try:
                logger.debug(f"Pinging shard: {node_name}")
                # Assuming shard_client.ping() is an async method
                is_alive = asyncio.run(shard_client.ping())
                current_health[node_name] = is_alive
                self._shard_health[node_name] = is_alive  # Update internal state
                logger.info(f"Shard {node_name} health: {'UP' if is_alive else 'DOWN'}")
            except Exception as e:
                logger.warning(f"Shard {node_name} health check failed: {e}")
                current_health[node_name] = False
                self._shard_health[node_name] = False
        logger.info("Shard health check complete.")
        return current_health

    @property
    def shard_health(self) -> Dict[str, bool]:
        """
        Returns the last known health status of shards.
        Does not perform a new health check; use `check_shards_health()` for that.
        """
        return self._shard_health.copy()  # Return a copy

    # --- Application-specific/AI Feature Methods ---
    def neural_backtrack(self, pattern: List[float], depth: int = 3) -> Dict[str, Any]:
        """
        Perform neural backtracking from a given pattern to find related or causal data points.

        Args:
            pattern: A list of floats representing the pattern to backtrack from.
            depth: The depth of backtracking to perform (default 3).

        Returns:
            A dictionary containing the backtracking results, structured by the backend.

        Raises:
            TypeError: If `pattern` is not a list of numbers.
            ValueError: If `depth` is not a positive integer.
            NotImplementedError: If the underlying client does not support this feature.
            WDBXError: If the neural backtrack operation fails.
        """
        if not isinstance(pattern, list) or not all(isinstance(x, (float, int)) for x in pattern):
            raise TypeError("Pattern for neural backtrack must be a list of numbers.")
        if not isinstance(depth, int) or depth <= 0:
            raise ValueError("Depth for neural backtrack must be a positive integer.")

        logger.info(f"Performing neural backtrack for pattern (size: {len(pattern)}), depth: {depth}")
        if not hasattr(self._client, "neural_backtrack_async") or not asyncio.iscoroutinefunction(getattr(self._client, "neural_backtrack_async")):
            logger.error("Underlying client does not support async method 'neural_backtrack_async'.")
            raise NotImplementedError("Underlying client does not support async method 'neural_backtrack_async'.")
        results = self._run_async(self._client.neural_backtrack_async(pattern, depth=depth))
        logger.info(f"Neural backtrack completed. Result keys: {list(results.keys()) if isinstance(results, dict) else 'N/A'}")
        return results

    def detect_drift(self, threshold: float = 0.1, dataset_sample: Optional[List[List[float]]] = None) -> Dict[str, Any]:
        """
        Detect concept drift in the stored data or a provided sample against the model.

        Args:
            threshold: The sensitivity threshold for drift detection (default 0.1).
            dataset_sample: Optional list of vectors to check for drift against the main dataset/model.
                            If None, drift is typically checked within the existing dataset.

        Returns:
            A dictionary containing drift detection results (e.g., {'drift_detected': bool, 'score': float}).

        Raises:
            TypeError: If `dataset_sample` is provided and not a list of lists of numbers.
            ValueError: If `threshold` is not a float, typically expected to be between 0 and 1.
            NotImplementedError: If the underlying client does not support this feature.
            WDBXError: If the drift detection operation fails.
        """
        if not isinstance(threshold, float):
            raise TypeError("Drift detection threshold must be a float.")
        if dataset_sample is not None:
            if not isinstance(dataset_sample, list) or not all(isinstance(vec, list) and all(isinstance(x, (float, int)) for x in vec) for vec in dataset_sample):
                raise TypeError("Dataset sample for drift detection must be a list of lists of numbers.")

        logger.info(f"Detecting drift with threshold: {threshold}. Sample provided: {dataset_sample is not None}")
        if not hasattr(self._client, "drift_detect_async") or not asyncio.iscoroutinefunction(getattr(self._client, "drift_detect_async")):
            logger.error("Underlying client does not support async method 'drift_detect_async'.")
            raise NotImplementedError("Underlying client does not support async method 'drift_detect_async'.")
        results = self._run_async(self._client.drift_detect_async(threshold=threshold, dataset_sample=dataset_sample))
        logger.info(f"Drift detection completed. Drift detected: {results.get('drift_detected', 'N/A')}, Score: {results.get('score', 'N/A')}")
        return results

    def create_persona(self, name: str, config: Dict[str, Any]) -> int:
        """
        Create a new persona or agent configuration in the system.

        Args:
            name: The unique name for the persona.
            config: A dictionary containing configuration parameters for the persona.

        Returns:
            An integer ID for the newly created persona.

        Raises:
            TypeError: If `name` is not a string or `config` is not a dictionary.
            NotImplementedError: If the underlying client does not support this feature.
            WDBXError: If persona creation fails (e.g., name conflict, invalid config).
        """
        if not isinstance(name, str) or not name.strip():
            raise TypeError("Persona name must be a non-empty string.")
        if not isinstance(config, dict):
            raise TypeError("Persona config must be a dictionary.")

        logger.info(f"Creating persona with name: '{name}', config keys: {list(config.keys())}")
        if not hasattr(self._client, "persona_create_async") or not asyncio.iscoroutinefunction(getattr(self._client, "persona_create_async")):
            logger.error("Underlying client does not support async method 'persona_create_async'.")
            raise NotImplementedError("Underlying client does not support async method 'persona_create_async'.")
        persona_id = self._run_async(self._client.persona_create_async(name, config))
        logger.info(f"Persona '{name}' created successfully with ID: {persona_id}")
        return persona_id

    def switch_persona(self, name: str) -> bool:
        """
        Switch the active persona for the current client session or context.

        Args:
            name: The name of the persona to activate.

        Returns:
            True if the persona was switched successfully, False otherwise (e.g., persona not found).

        Raises:
            TypeError: If `name` is not a string.
            NotImplementedError: If the underlying client does not support this feature.
            WDBXError: If switching persona fails for other reasons.
        """
        if not isinstance(name, str) or not name.strip():
            raise TypeError("Persona name must be a non-empty string.")

        logger.info(f"Switching to persona: '{name}'")
        if not hasattr(self._client, "persona_switch_async") or not asyncio.iscoroutinefunction(getattr(self._client, "persona_switch_async")):
            logger.error("Underlying client does not support async method 'persona_switch_async'.")
            raise NotImplementedError("Underlying client does not support async method 'persona_switch_async'.")
        success = self._run_async(self._client.persona_switch_async(name))
        logger.info(f"Switch persona to '{name}' {'succeeded' if success else 'failed'}.")
        return success

    def list_personas(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        List available personas in the system, with pagination.

        Args:
            limit: Maximum number of personas to return (default 100).
            offset: Number of personas to skip (default 0).

        Returns:
            A list of dictionaries, where each dictionary represents a persona (e.g., its name and config snippet).

        Raises:
            ValueError: If `limit` or `offset` are negative.
            NotImplementedError: If the underlying client does not support this feature.
            WDBXError: If listing personas fails.
        """
        if not isinstance(limit, int) or limit < 0:
            raise ValueError("Limit must be a non-negative integer.")
        if not isinstance(offset, int) or offset < 0:
            raise ValueError("Offset must be a non-negative integer.")

        logger.info(f"Listing personas with limit={limit}, offset={offset}")
        if not hasattr(self._client, "persona_list_async") or not asyncio.iscoroutinefunction(getattr(self._client, "persona_list_async")):
            logger.error("Underlying client does not support async method 'persona_list_async'.")
            raise NotImplementedError("Underlying client does not support async method 'persona_list_async'.")
        personas = self._run_async(self._client.persona_list_async(limit=limit, offset=offset))
        logger.info(f"Retrieved {len(personas)} personas.")
        return personas

    def filter_content(self, text: str, ruleset_id: Optional[Union[int, str]] = None) -> str:
        """
        Filter input text based on predefined rules or a specified ruleset.

        Args:
            text: The text content to filter.
            ruleset_id: Optional ID or name of a specific ruleset to apply.
                        If None, a default ruleset might be used by the backend.

        Returns:
            The filtered text.

        Raises:
            TypeError: If `text` is not a string.
            NotImplementedError: If the underlying client does not support this feature.
            WDBXError: If the content filtering operation fails.
        """
        if not isinstance(text, str):
            raise TypeError("Text for content filtering must be a string.")
        if ruleset_id is not None and not isinstance(ruleset_id, (int, str)):
            raise TypeError("Ruleset ID must be an int or str if provided.")

        logger.info(f"Filtering content (length: {len(text)}). Ruleset ID: {ruleset_id or 'default'}")
        if not hasattr(self._client, "content_filter_async") or not asyncio.iscoroutinefunction(getattr(self._client, "content_filter_async")):
            logger.error("Underlying client does not support async method 'content_filter_async'.")
            raise NotImplementedError("Underlying client does not support async method 'content_filter_async'.")
        filtered_text = self._run_async(self._client.content_filter_async(text, ruleset_id=ruleset_id))
        logger.info(f"Content filtering completed. Original length: {len(text)}, Filtered length: {len(filtered_text)}")
        return filtered_text

    def mitigate_bias(self, text: str, method: Optional[str] = None) -> str:
        """
        Apply bias mitigation techniques to the input text.

        Args:
            text: The text content to process for bias mitigation.
            method: Optional string specifying the bias mitigation method or strategy to use.
                    If None, a default method might be applied by the backend.

        Returns:
            The text after bias mitigation has been applied.

        Raises:
            TypeError: If `text` is not a string.
            NotImplementedError: If the underlying client does not support this feature.
            WDBXError: If the bias mitigation operation fails.
        """
        if not isinstance(text, str):
            raise TypeError("Text for bias mitigation must be a string.")
        if method is not None and not isinstance(method, str):
            raise TypeError("Bias mitigation method must be a string if provided.")

        logger.info(f"Mitigating bias in content (length: {len(text)}). Method: {method or 'default'}")
        if not hasattr(self._client, "bias_mitigate_async") or not asyncio.iscoroutinefunction(getattr(self._client, "bias_mitigate_async")):
            logger.error("Underlying client does not support async method 'bias_mitigate_async'.")
            raise NotImplementedError("Underlying client does not support async method 'bias_mitigate_async'.")
        mitigated_text = self._run_async(self._client.bias_mitigate_async(text, method=method))
        logger.info(f"Bias mitigation completed. Original length: {len(text)}, Mitigated length: {len(mitigated_text)}")
        return mitigated_text

    # --- HTTP Server and API Management Methods ---
    def start_http_server(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        use_ssl: bool = False,
        certfile: Optional[str] = None,
        keyfile: Optional[str] = None,
    ) -> None:
        """
        Start an HTTP server for API access if supported by the backend.

        Args:
            host: Hostname or IP address to bind the server to (default '127.0.0.1').
            port: Port number for the server (default 8000).
            use_ssl: Whether to use SSL/TLS (HTTPS) (default False).
            certfile: Path to the SSL certificate file (required if use_ssl is True).
            keyfile: Path to the SSL private key file (required if use_ssl is True).

        Raises:
            TypeError: If host, certfile, or keyfile are not strings when provided, or port is not int.
            ValueError: If use_ssl is True but certfile or keyfile are missing.
            NotImplementedError: If the underlying client does not support this feature.
            WDBXError: If starting the server fails.
        """
        if not isinstance(host, str):
            raise TypeError("Host must be a string.")
        if not isinstance(port, int):
            raise TypeError("Port must be an integer.")
        if use_ssl:
            if not certfile or not keyfile:
                raise ValueError("Both certfile and keyfile are required when use_ssl is True.")
            if not isinstance(certfile, str) or not isinstance(keyfile, str):
                raise TypeError("certfile and keyfile must be strings.")

        logger.info(f"Starting HTTP server on {host}:{port}. SSL: {use_ssl}")
        if not hasattr(self._client, "http_server_start_async") or not asyncio.iscoroutinefunction(getattr(self._client, "http_server_start_async")):
            logger.error("Underlying client does not support async method 'http_server_start_async'.")
            raise NotImplementedError("Underlying client does not support async method 'http_server_start_async'.")
        self._run_async(
            self._client.http_server_start_async(
                host=host,
                port=port,
                use_ssl=use_ssl,
                certfile=certfile,
                keyfile=keyfile,
            )
        )
        logger.info(f"HTTP server start initiated on {host}:{port}.")

    def stop_http_server(self) -> None:
        """
        Stop the HTTP server if it is running.

        Raises:
            NotImplementedError: If the underlying client does not support this feature.
            WDBXError: If stopping the server fails.
        """
        logger.info("Stopping HTTP server...")
        if not hasattr(self._client, "http_server_stop_async") or not asyncio.iscoroutinefunction(getattr(self._client, "http_server_stop_async")):
            logger.error("Underlying client does not support async method 'http_server_stop_async'.")
            raise NotImplementedError("Underlying client does not support async method 'http_server_stop_async'.")
        self._run_async(self._client.http_server_stop_async())
        logger.info("HTTP server stop initiated.")

    def get_api_endpoints(self) -> List[Dict[str, Any]]:
        """
        Retrieve a list of available API endpoints provided by the HTTP server.

        Returns:
            A list of dictionaries, each describing an API endpoint (e.g., path, methods, description).

        Raises:
            NotImplementedError: If the underlying client does not support this feature.
            WDBXError: If retrieving API endpoints fails.
        """
        logger.info("Getting API endpoints...")
        if not hasattr(self._client, "api_get_endpoints_async") or not asyncio.iscoroutinefunction(getattr(self._client, "api_get_endpoints_async")):
            logger.error("Underlying client does not support async method 'api_get_endpoints_async'.")
            raise NotImplementedError("Underlying client does not support async method 'api_get_endpoints_async'.")
        endpoints = self._run_async(self._client.api_get_endpoints_async())
        logger.info(f"Retrieved {len(endpoints)} API endpoints.")
        return endpoints

    def set_api_auth_required(self, required: bool = True, endpoint_paths: Optional[List[str]] = None) -> None:
        """
        Configure API authentication requirements for all or specific endpoints.

        Args:
            required: True to require authentication, False to make it optional/public (default True).
            endpoint_paths: Optional list of specific endpoint paths (e.g., '/search') to apply this setting to.
                            If None, applies globally to all manageable endpoints.

        Raises:
            TypeError: If `required` is not a boolean or `endpoint_paths` is not a list of strings when provided.
            NotImplementedError: If the underlying client does not support this feature.
            WDBXError: If setting API authentication fails.
        """
        if not isinstance(required, bool):
            raise TypeError("API authentication 'required' flag must be a boolean.")
        if endpoint_paths is not None:
            if not isinstance(endpoint_paths, list) or not all(isinstance(p, str) for p in endpoint_paths):
                raise TypeError("endpoint_paths must be a list of strings if provided.")

        target_endpoints = "globally" if endpoint_paths is None else f"for endpoints: {endpoint_paths}"
        logger.info(f"Setting API authentication required: {required} {target_endpoints}")
        if not hasattr(self._client, "api_set_auth_required_async") or not asyncio.iscoroutinefunction(getattr(self._client, "api_set_auth_required_async")):
            logger.error("Underlying client does not support async method 'api_set_auth_required_async'.")
            raise NotImplementedError("Underlying client does not support async method 'api_set_auth_required_async'.")
        self._run_async(self._client.api_set_auth_required_async(required=required, endpoint_paths=endpoint_paths))
        logger.info(f"API authentication requirement set to {required} {target_endpoints}.")

    def set_api_cors_origins(self, origins: List[str]) -> None:
        """
        Set allowed CORS (Cross-Origin Resource Sharing) origins for the API.

        Args:
            origins: A list of origin strings (e.g., ['https://example.com', 'http://localhost:3000']).
                     An empty list might disable CORS or revert to a default restrictive policy.

        Raises:
            TypeError: If `origins` is not a list of strings.
            NotImplementedError: If the underlying client does not support this feature.
            WDBXError: If setting CORS origins fails.
        """
        if not isinstance(origins, list) or not all(isinstance(o, str) for o in origins):
            raise TypeError("CORS origins must be a list of strings.")

        logger.info(f"Setting API CORS origins to: {origins}")
        if not hasattr(self._client, "api_set_cors_origins_async") or not asyncio.iscoroutinefunction(getattr(self._client, "api_set_cors_origins_async")):
            logger.error("Underlying client does not support async method 'api_set_cors_origins_async'.")
            raise NotImplementedError("Underlying client does not support async method 'api_set_cors_origins_async'.")
        self._run_async(self._client.api_set_cors_origins_async(origins))
        logger.info(f"API CORS origins set successfully.")

    # --- Plugin Management Methods ---
    def enable_plugin(self, plugin_name: str) -> bool:
        """
        Enable a specific plugin by name.

        Args:
            plugin_name: The registered name of the plugin to enable.

        Returns:
            True if the plugin was enabled successfully, False otherwise (e.g., plugin not found or already enabled).

        Raises:
            TypeError: If `plugin_name` is not a string.
            NotImplementedError: If the underlying client does not support this feature.
            WDBXError: If enabling the plugin fails.
        """
        if not isinstance(plugin_name, str) or not plugin_name.strip():
            raise TypeError("Plugin name must be a non-empty string.")

        logger.info(f"Enabling plugin: '{plugin_name}'")
        if not hasattr(self._client, "plugin_enable_async") or not asyncio.iscoroutinefunction(getattr(self._client, "plugin_enable_async")):
            logger.error("Underlying client does not support async method 'plugin_enable_async'.")
            raise NotImplementedError("Underlying client does not support async method 'plugin_enable_async'.")
        result = self._run_async(self._client.plugin_enable_async(plugin_name))
        if result and plugin_name in self._plugin_registry:
            plugin = self._plugin_registry[plugin_name]
            module = plugin.get("module")
            if module and hasattr(module, "on_enable"):
                module.on_enable(self)
            plugin["status"] = "enabled"
        return result

    def disable_plugin(self, plugin_name: str) -> bool:
        """
        Disable a specific plugin by name.

        Args:
            plugin_name: The registered name of the plugin to disable.

        Returns:
            True if the plugin was disabled successfully, False otherwise (e.g., plugin not found or already disabled).

        Raises:
            TypeError: If `plugin_name` is not a string.
            NotImplementedError: If the underlying client does not support this feature.
            WDBXError: If disabling the plugin fails.
        """
        if not isinstance(plugin_name, str) or not plugin_name.strip():
            raise TypeError("Plugin name must be a non-empty string.")

        logger.info(f"Disabling plugin: '{plugin_name}'")
        if not hasattr(self._client, "plugin_disable_async") or not asyncio.iscoroutinefunction(getattr(self._client, "plugin_disable_async")):
            logger.error("Underlying client does not support async method 'plugin_disable_async'.")
            raise NotImplementedError("Underlying client does not support async method 'plugin_disable_async'.")
        result = self._run_async(self._client.plugin_disable_async(plugin_name))
        if result and plugin_name in self._plugin_registry:
            plugin = self._plugin_registry[plugin_name]
            module = plugin.get("module")
            if module and hasattr(module, "on_disable"):
                module.on_disable(self)
            plugin["status"] = "disabled"
        return result

    def list_plugins(self, active_only: bool = False) -> List[Dict[str, Any]]:
        """
        List available plugins and their status.

        Args:
            active_only: If True, list only currently enabled plugins (default False).

        Returns:
            A list of dictionaries, each representing a plugin (e.g., name, status, description).

        Raises:
            TypeError: If `active_only` is not a boolean.
            NotImplementedError: If the underlying client does not support this feature.
            WDBXError: If listing plugins fails.
        """
        if not isinstance(active_only, bool):
            raise TypeError("active_only flag must be a boolean.")

        logger.info(f"Listing plugins. Active only: {active_only}")
        plugins_list = []
        for name, meta in self._plugin_registry.items():
            if active_only and meta.get("status") != "enabled":
                continue
            plugins_list.append(
                {
                    "name": name,
                    "version": meta["info"].get("version"),
                    "author": meta["info"].get("author"),
                    "description": meta["info"].get("description"),
                    "status": meta["status"],
                    "error": meta.get("error"),
                }
            )
        return plugins_list

    def reload_plugin(self, plugin_name: str) -> bool:
        """Reload a specific plugin by calling its on_reload hook."""
        if plugin_name not in self._plugin_registry:
            return False
        plugin = self._plugin_registry[plugin_name]
        module = plugin.get("module")
        if module and hasattr(module, "on_reload"):
            try:
                module.on_reload(self)
                plugin["status"] = "reloaded"
                return True
            except Exception as e:
                logger.error(f"Failed to reload plugin {plugin_name}: {e}", exc_info=True)
                plugin["error"] = str(e)
                plugin["status"] = "error"
        return False

    # --- Visualization Methods ---
    def visualize_vectors(self, vectors: List[List[float]], method: str = "tsne", **kwargs: Any) -> Dict[str, Any]:
        """
        Generate a 2D or 3D visualization of the given vectors using a dimensionality reduction technique.

        Args:
            vectors: A list of high-dimensional vectors to visualize.
            method: The dimensionality reduction method to use (e.g., "tsne", "pca", "umap") (default "tsne").
            **kwargs: Additional keyword arguments to pass to the visualization/reduction algorithm.

        Returns:
            A dictionary containing the visualization data, often including coordinates and labels/metadata.
            Example: {'coordinates': [[x1,y1], [x2,y2], ...], 'method': 'tsne'}

        Raises:
            TypeError: If `vectors` is not a list of lists of numbers, or `method` is not a string.
            NotImplementedError: If the underlying client does not support this feature or method.
            WDBXError: If the visualization generation fails.
        """
        if not isinstance(vectors, list) or not all(isinstance(v, list) and all(isinstance(x, (float, int)) for x in v) for v in vectors):
            raise TypeError("Input for visualize_vectors must be a list of lists of numbers.")
        if not isinstance(method, str) or not method.strip():
            raise TypeError("Visualization method must be a non-empty string.")

        logger.info(f"Visualizing {len(vectors)} vectors using method: '{method}'. Additional args: {kwargs}")
        if not hasattr(self._client, "vectors_visualize_async") or not asyncio.iscoroutinefunction(getattr(self._client, "vectors_visualize_async")):
            logger.error("Underlying client does not support async method 'vectors_visualize_async'.")
            raise NotImplementedError("Underlying client does not support async method 'vectors_visualize_async'.")
        visualization_data = self._run_async(self._client.vectors_visualize_async(vectors, method=method, **kwargs))
        logger.info(f"Vector visualization generated using '{method}'. Result keys: {list(visualization_data.keys()) if isinstance(visualization_data, dict) else 'N/A'}")
        return visualization_data

    # --- Security, Access Control, and Data Management ---
    def set_access_token(self, token: str) -> None:
        """
        Set an access token for authenticating subsequent requests to the backend.
        The token is typically a JWT or an API key.

        Args:
            token: The access token string.

        Raises:
            TypeError: If `token` is not a string.
            NotImplementedError: If the underlying client does not support this feature.
            WDBXError: If setting the token fails (e.g., token format invalid).
        """
        if not isinstance(token, str):
            raise TypeError("Access token must be a string.")

        logger.info("Setting access token for WDBX client.")
        if not hasattr(self._client, "set_access_token_async") or not asyncio.iscoroutinefunction(getattr(self._client, "set_access_token_async")):
            logger.error("Underlying client does not support async method 'set_access_token_async'.")
            raise NotImplementedError("Underlying client does not support async method 'set_access_token_async'.")
        self._run_async(self._client.set_access_token_async(token))
        logger.info("Access token set successfully (first few chars for verification if appropriate).")  # Be cautious logging tokens

    def encrypt_data(self, data: Any, key_id: Optional[str] = None) -> str:
        """
        Encrypt provided data using the backend's encryption mechanism.

        Args:
            data: The data to encrypt (can be any serializable type supported by the backend).
            key_id: Optional identifier for a specific encryption key to use.
                    If None, a default key management strategy is used.

        Returns:
            A string representation of the encrypted data (e.g., base64 encoded ciphertext).

        Raises:
            NotImplementedError: If the underlying client does not support this feature.
            WDBXError: If encryption fails.
        """
        logger.info(f"Encrypting data. Key ID: {key_id or 'default'}")
        if not hasattr(self._client, "encrypt_data_async") or not asyncio.iscoroutinefunction(getattr(self._client, "encrypt_data_async")):
            logger.error("Underlying client does not support async method 'encrypt_data_async'.")
            raise NotImplementedError("Underlying client does not support async method 'encrypt_data_async'.")
        encrypted_data_str = self._run_async(self._client.encrypt_data_async(data, key_id=key_id))
        logger.info(f"Data encrypted successfully. Encrypted data length: {len(encrypted_data_str)}")
        return encrypted_data_str

    def decrypt_data(self, encrypted_data: str, key_id: Optional[str] = None) -> Any:
        """
        Decrypt data that was previously encrypted by the backend.

        Args:
            encrypted_data: The string representation of the encrypted data.
            key_id: Optional identifier for the specific decryption key to use.

        Returns:
            The original decrypted data.

        Raises:
            TypeError: If `encrypted_data` is not a string.
            NotImplementedError: If the underlying client does not support this feature.
            WDBXError: If decryption fails (e.g., wrong key, corrupted data).
        """
        if not isinstance(encrypted_data, str):
            raise TypeError("Encrypted data for decryption must be a string.")

        logger.info(f"Decrypting data (length: {len(encrypted_data)}). Key ID: {key_id or 'default'}")
        if not hasattr(self._client, "decrypt_data_async") or not asyncio.iscoroutinefunction(getattr(self._client, "decrypt_data_async")):
            logger.error("Underlying client does not support async method 'decrypt_data_async'.")
            raise NotImplementedError("Underlying client does not support async method 'decrypt_data_async'.")
        decrypted_data = self._run_async(self._client.decrypt_data_async(encrypted_data, key_id=key_id))
        logger.info("Data decrypted successfully.")
        return decrypted_data

    def generate_api_key(
        self,
        user_id: str,
        permissions: Optional[List[str]] = None,
        expires_at: Optional[Union[str, int]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a new API key for a user, optionally with specific permissions and an expiry.

        Args:
            user_id: The identifier for the user for whom the key is generated.
            permissions: Optional list of permission strings associated with the key.
            expires_at: Optional expiration timestamp (ISO 8601 string or Unix timestamp int) for the key.

        Returns:
            A dictionary containing the new API key and its details (e.g., {'key': '...', 'user_id': ..., 'expires_at': ...}).

        Raises:
            TypeError: If `user_id` is not a string, or other params have invalid types.
            NotImplementedError: If the underlying client does not support this feature.
            WDBXError: If API key generation fails.
        """
        if not isinstance(user_id, str) or not user_id.strip():
            raise TypeError("User ID must be a non-empty string.")
        if permissions is not None and (not isinstance(permissions, list) or not all(isinstance(p, str) for p in permissions)):
            raise TypeError("Permissions must be a list of strings if provided.")
        if expires_at is not None and not isinstance(expires_at, (str, int)):
            raise TypeError("expires_at must be a string (ISO 8601) or integer (Unix timestamp) if provided.")

        logger.info(f"Generating API key for user_id: '{user_id}'. Permissions: {permissions}, Expires: {expires_at}")
        if not hasattr(self._client, "api_key_generate_async") or not asyncio.iscoroutinefunction(getattr(self._client, "api_key_generate_async")):
            logger.error("Underlying client does not support async method 'api_key_generate_async'.")
            raise NotImplementedError("Underlying client does not support async method 'api_key_generate_async'.")
        key_info = self._run_async(self._client.api_key_generate_async(user_id, permissions=permissions, expires_at=expires_at))
        logger.info(f"API key generated successfully for user_id: '{user_id}'. Key ID/Prefix: {key_info.get('id', key_info.get('key_prefix', 'N/A'))}")
        return key_info

    def revoke_api_key(self, key_identifier: str) -> bool:
        """
        Revoke an existing API key.

        Args:
            key_identifier: The API key itself, or a unique ID/prefix representing the key to revoke.

        Returns:
            True if the key was revoked successfully, False otherwise (e.g., key not found).

        Raises:
            TypeError: If `key_identifier` is not a string.
            NotImplementedError: If the underlying client does not support this feature.
            WDBXError: If API key revocation fails.
        """
        if not isinstance(key_identifier, str) or not key_identifier.strip():
            raise TypeError("API key identifier for revocation must be a non-empty string.")

        logger.info(f"Revoking API key with identifier/prefix: '{key_identifier[:10]}...'")  # Log only a prefix
        if not hasattr(self._client, "api_key_revoke_async") or not asyncio.iscoroutinefunction(getattr(self._client, "api_key_revoke_async")):
            logger.error("Underlying client does not support async method 'api_key_revoke_async'.")
            raise NotImplementedError("Underlying client does not support async method 'api_key_revoke_async'.")
        success = self._run_async(self._client.api_key_revoke_async(key_identifier))
        logger.info(f"API key revocation for '{key_identifier[:10]}...' {'succeeded' if success else 'failed'}.")
        return success

    def assign_role(self, user_id: str, roles: List[str]) -> bool:
        """
        Assign one or more roles to a user.

        Args:
            user_id: The identifier of the user.
            roles: A list of role names to assign.

        Returns:
            True if roles were assigned successfully, False otherwise.

        Raises:
            TypeError: If `user_id` is not a string or `roles` is not a list of strings.
            NotImplementedError: If the underlying client does not support this feature.
            WDBXError: If role assignment fails.
        """
        if not isinstance(user_id, str) or not user_id.strip():
            raise TypeError("User ID must be a non-empty string.")
        if not isinstance(roles, list) or not all(isinstance(r, str) and r.strip() for r in roles):
            raise TypeError("Roles must be a list of non-empty strings.")

        logger.info(f"Assigning roles {roles} to user_id: '{user_id}'")
        if not hasattr(self._client, "user_assign_roles_async") or not asyncio.iscoroutinefunction(getattr(self._client, "user_assign_roles_async")):  # Assuming plural roles
            logger.error("Underlying client does not support async method for assigning roles.")
            raise NotImplementedError("Underlying client does not support async method for assigning roles.")
        success = self._run_async(self._client.user_assign_roles_async(user_id, roles))
        logger.info(f"Role assignment for user_id '{user_id}' {'succeeded' if success else 'failed'}.")
        return success

    def remove_role(self, user_id: str, roles: List[str]) -> bool:
        """
        Remove one or more roles from a user.

        Args:
            user_id: The identifier of the user.
            roles: A list of role names to remove.

        Returns:
            True if roles were removed successfully, False otherwise.

        Raises:
            TypeError: If `user_id` is not a string or `roles` is not a list of strings.
            NotImplementedError: If the underlying client does not support this feature.
            WDBXError: If role removal fails.
        """
        if not isinstance(user_id, str) or not user_id.strip():
            raise TypeError("User ID must be a non-empty string.")
        if not isinstance(roles, list) or not all(isinstance(r, str) and r.strip() for r in roles):
            raise TypeError("Roles must be a list of non-empty strings.")

        logger.info(f"Removing roles {roles} from user_id: '{user_id}'")
        if not hasattr(self._client, "user_remove_roles_async") or not asyncio.iscoroutinefunction(getattr(self._client, "user_remove_roles_async")):  # Assuming plural roles
            logger.error("Underlying client does not support async method for removing roles.")
            raise NotImplementedError("Underlying client does not support async method for removing roles.")
        success = self._run_async(self._client.user_remove_roles_async(user_id, roles))
        logger.info(f"Role removal for user_id '{user_id}' {'succeeded' if success else 'failed'}.")
        return success

    def get_audit_logs(
        self,
        limit: int = 100,
        offset: int = 0,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        since: Optional[Union[str, int]] = None,
        until: Optional[Union[str, int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve audit logs, with filtering and pagination.

        Args:
            limit: Maximum number of log entries to return (default 100).
            offset: Number of log entries to skip (default 0).
            user_id: Optional filter by user ID.
            action: Optional filter by action performed (e.g., 'login', 'delete_vector').
            since: Optional start timestamp (ISO 8601 string or Unix timestamp int) for logs.
            until: Optional end timestamp (ISO 8601 string or Unix timestamp int) for logs.

        Returns:
            A list of dictionaries, where each dictionary represents an audit log entry.

        Raises:
            ValueError: If `limit` or `offset` are negative.
            TypeError: For invalid filter parameter types.
            NotImplementedError: If the underlying client does not support this feature.
            WDBXError: If retrieving audit logs fails.
        """
        if not isinstance(limit, int) or limit < 0:
            raise ValueError("Limit must be a non-negative integer.")
        if not isinstance(offset, int) or offset < 0:
            raise ValueError("Offset must be a non-negative integer.")
        if user_id is not None and (not isinstance(user_id, str) or not user_id.strip()):
            raise TypeError("user_id filter must be a non-empty string if provided.")
        if action is not None and (not isinstance(action, str) or not action.strip()):
            raise TypeError("action filter must be a non-empty string if provided.")
        if since is not None and not isinstance(since, (str, int)):
            raise TypeError("since filter must be a string (ISO 8601) or int (Unix timestamp) if provided.")
        if until is not None and not isinstance(until, (str, int)):
            raise TypeError("until filter must be a string (ISO 8601) or int (Unix timestamp) if provided.")

        logger.info(f"Fetching audit logs with limit={limit}, offset={offset}, user_id={user_id}, action={action}, since={since}, until={until}")
        if not hasattr(self._client, "audit_logs_get_async") or not asyncio.iscoroutinefunction(getattr(self._client, "audit_logs_get_async")):
            logger.error("Underlying client does not support async method 'audit_logs_get_async'.")
            raise NotImplementedError("Underlying client does not support async method 'audit_logs_get_async'.")

        logs = self._run_async(
            self._client.audit_logs_get_async(
                limit=limit,
                offset=offset,
                user_id=user_id,
                action=action,
                since=since,
                until=until,
            )
        )
        logger.info(f"Retrieved {len(logs)} audit log entries.")
        return logs

    def set_rate_limit(self, endpoint: str, limit: int, period_seconds: int, scope: str = "ip") -> bool:
        """
        Configure a rate limit for a specific API endpoint.

        Args:
            endpoint: The API endpoint path (e.g., '/search') or a pattern.
            limit: The maximum number of requests allowed.
            period_seconds: The time window in seconds for the limit.
            scope: The scope of the rate limit ('ip', 'user', 'global') (default 'ip').

        Returns:
            True if the rate limit was set successfully, False otherwise.

        Raises:
            TypeError: For invalid parameter types.
            ValueError: For invalid limit, period, or scope values.
            NotImplementedError: If the underlying client does not support this feature.
            WDBXError: If setting the rate limit fails.
        """
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise TypeError("Endpoint for rate limit must be a non-empty string.")
        if not isinstance(limit, int) or limit < 0:  # Allow 0 for blocking? Backend dependent.
            raise ValueError("Rate limit must be a non-negative integer.")
        if not isinstance(period_seconds, int) or period_seconds <= 0:
            raise ValueError("Rate limit period must be a positive integer in seconds.")
        if scope not in ["ip", "user", "global"]:
            raise ValueError("Rate limit scope must be one of 'ip', 'user', 'global'.")

        logger.info(f"Setting rate limit for endpoint '{endpoint}': {limit} requests per {period_seconds}s, scope: {scope}")
        if not hasattr(self._client, "rate_limit_set_async") or not asyncio.iscoroutinefunction(getattr(self._client, "rate_limit_set_async")):
            logger.error("Underlying client does not support async method 'rate_limit_set_async'.")
            raise NotImplementedError("Underlying client does not support async method 'rate_limit_set_async'.")
        success = self._run_async(self._client.rate_limit_set_async(endpoint, limit, period_seconds, scope=scope))
        logger.info(f"Rate limit setting for endpoint '{endpoint}' {'succeeded' if success else 'failed'}.")
        return success

    def enable_cache(self, cache_name: Optional[str] = None) -> bool:
        """
        Enable caching for the backend, optionally for a specific named cache.

        Args:
            cache_name: Optional name of a specific cache to enable. If None, may enable a default/global cache.

        Returns:
            True if caching was enabled successfully, False otherwise.

        Raises:
            TypeError: If `cache_name` is provided and not a string.
            NotImplementedError: If the underlying client does not support this feature.
            WDBXError: If enabling cache fails.
        """
        if cache_name is not None and (not isinstance(cache_name, str) or not cache_name.strip()):
            raise TypeError("Cache name must be a non-empty string if provided.")

        logger.info(f"Enabling cache: {cache_name or 'default/global'}")
        if not hasattr(self._client, "cache_enable_async") or not asyncio.iscoroutinefunction(getattr(self._client, "cache_enable_async")):
            logger.error("Underlying client does not support async method 'cache_enable_async'.")
            raise NotImplementedError("Underlying client does not support async method 'cache_enable_async'.")
        success = self._run_async(self._client.cache_enable_async(cache_name=cache_name))
        logger.info(f"Cache enabling for '{cache_name or 'default/global'}' {'succeeded' if success else 'failed'}.")
        return success

    def disable_cache(self, cache_name: Optional[str] = None) -> bool:
        """
        Disable caching for the backend, optionally for a specific named cache.

        Args:
            cache_name: Optional name of a specific cache to disable. If None, may disable a default/global cache.

        Returns:
            True if caching was disabled successfully, False otherwise.

        Raises:
            TypeError: If `cache_name` is provided and not a string.
            NotImplementedError: If the underlying client does not support this feature.
            WDBXError: If disabling cache fails.
        """
        if cache_name is not None and (not isinstance(cache_name, str) or not cache_name.strip()):
            raise TypeError("Cache name must be a non-empty string if provided.")

        logger.info(f"Disabling cache: {cache_name or 'default/global'}")
        if not hasattr(self._client, "cache_disable_async") or not asyncio.iscoroutinefunction(getattr(self._client, "cache_disable_async")):
            logger.error("Underlying client does not support async method 'cache_disable_async'.")
            raise NotImplementedError("Underlying client does not support async method 'cache_disable_async'.")
        success = self._run_async(self._client.cache_disable_async(cache_name=cache_name))
        logger.info(f"Cache disabling for '{cache_name or 'default/global'}' {'succeeded' if success else 'failed'}.")
        return success

    def get_cache_stats(self, cache_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve statistics for a specific cache or the default/global cache.

        Args:
            cache_name: Optional name of the cache to get stats for.

        Returns:
            A dictionary containing cache statistics (e.g., hits, misses, size, eviction count).

        Raises:
            TypeError: If `cache_name` is provided and not a string.
            NotImplementedError: If the underlying client does not support this feature.
            WDBXError: If retrieving cache stats fails.
        """
        if cache_name is not None and (not isinstance(cache_name, str) or not cache_name.strip()):
            raise TypeError("Cache name must be a non-empty string if provided.")

        logger.info(f"Getting cache stats for: {cache_name or 'default/global'}")
        if not hasattr(self._client, "cache_get_stats_async") or not asyncio.iscoroutinefunction(getattr(self._client, "cache_get_stats_async")):
            logger.error("Underlying client does not support async method 'cache_get_stats_async'.")
            raise NotImplementedError("Underlying client does not support async method 'cache_get_stats_async'.")
        stats = self._run_async(self._client.cache_get_stats_async(cache_name=cache_name))
        logger.info(f"Cache stats for '{cache_name or 'default/global'}': {stats}")
        return stats

    def backup_database(self, path: str, strategy: str = "full") -> bool:
        """
        Perform a backup of the database to the specified path.

        Args:
            path: The file path or directory where the backup should be stored.
            strategy: The backup strategy to use (e.g., "full", "incremental"). Default is "full".

        Returns:
            True if the backup was successful, False otherwise.

        Raises:
            TypeError: If `path` or `strategy` are not strings.
            NotImplementedError: If the underlying client does not support this feature.
            WDBXError: If the backup operation fails.
        """
        if not isinstance(path, str) or not path.strip():
            raise TypeError("Backup path must be a non-empty string.")
        if not isinstance(strategy, str) or not strategy.strip():
            raise TypeError("Backup strategy must be a non-empty string.")

        logger.info(f"Backing up database to '{path}' using strategy: '{strategy}'")
        if not hasattr(self._client, "database_backup_async") or not asyncio.iscoroutinefunction(getattr(self._client, "database_backup_async")):
            logger.error("Underlying client does not support async method 'database_backup_async'.")
            raise NotImplementedError("Underlying client does not support async method 'database_backup_async'.")
        success = self._run_async(self._client.database_backup_async(path, strategy=strategy))
        logger.info(f"Database backup to '{path}' {'succeeded' if success else 'failed'}.")
        return success

    def restore_database(self, path: str, strategy: str = "full") -> bool:
        """
        Restore the database from a backup at the specified path.

        Args:
            path: The file path or directory of the backup to restore from.
            strategy: The restore strategy (should match backup, e.g., "full"). Default is "full".

        Returns:
            True if the restore was successful, False otherwise.

        Raises:
            TypeError: If `path` or `strategy` are not strings.
            NotImplementedError: If the underlying client does not support this feature.
            WDBXError: If the restore operation fails.
        """
        if not isinstance(path, str) or not path.strip():
            raise TypeError("Restore path must be a non-empty string.")
        if not isinstance(strategy, str) or not strategy.strip():
            raise TypeError("Restore strategy must be a non-empty string.")

        logger.warning(f"Restoring database from '{path}' using strategy: '{strategy}'. This is a potentially destructive operation.")
        if not hasattr(self._client, "database_restore_async") or not asyncio.iscoroutinefunction(getattr(self._client, "database_restore_async")):
            logger.error("Underlying client does not support async method 'database_restore_async'.")
            raise NotImplementedError("Underlying client does not support async method 'database_restore_async'.")
        success = self._run_async(self._client.database_restore_async(path, strategy=strategy))
        logger.info(f"Database restore from '{path}' {'succeeded' if success else 'failed'}.")
        return success

    def get_database_version(self) -> str:
        """
        Get the version of the underlying database or storage engine.

        Returns:
            A string representing the database version.

        Raises:
            NotImplementedError: If the underlying client does not support this feature.
            WDBXError: If retrieving the version fails.
        """
        logger.info("Getting database version...")
        if not hasattr(self._client, "database_get_version_async") or not asyncio.iscoroutinefunction(getattr(self._client, "database_get_version_async")):
            logger.error("Underlying client does not support async method 'database_get_version_async'.")
            raise NotImplementedError("Underlying client does not support async method 'database_get_version_async'.")
        version = self._run_async(self._client.database_get_version_async())
        logger.info(f"Database version: {version}")
        return version

    def supports_feature(self, feature_name: str) -> bool:
        """
        Check if the backend supports a specific named feature.

        Args:
            feature_name: The name of the feature to check (e.g., "GEO_SPATIAL_SEARCH", "TRANSACTIONS").

        Returns:
            True if the feature is supported, False otherwise.

        Raises:
            TypeError: If `feature_name` is not a string.
            NotImplementedError: If the underlying client does not support this capability check itself.
            WDBXError: If the feature check fails.
        """
        if not isinstance(feature_name, str) or not feature_name.strip():
            raise TypeError("Feature name must be a non-empty string.")

        logger.info(f"Checking if feature '{feature_name}' is supported...")
        if not hasattr(self._client, "feature_supports_async") or not asyncio.iscoroutinefunction(getattr(self._client, "feature_supports_async")):
            logger.error("Underlying client does not support async method 'feature_supports_async'.")
            raise NotImplementedError("Underlying client does not support async method 'feature_supports_async'.")
        supported = self._run_async(self._client.feature_supports_async(feature_name))
        logger.info(f"Feature '{feature_name}' is {'supported' if supported else 'not supported'}.")
        return supported

    # --- Shard related Passthrough Methods ---
    # These methods might be useful if the main client needs to directly interact with a shard's
    # unique capabilities not covered by the general sharded operations. This is highly backend-specific.

    def shard_passthrough(self, shard_id: str, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """
        Allows calling a specific method directly on a designated shard client.
        This is for advanced use cases where a shard might expose unique functionality
        not available through the aggregated WDBX client methods.

        Args:
            shard_id: The identifier of the target shard (must be a key in `self.shard_clients`).
            method_name: The name of the asynchronous method to call on the shard client.
            *args: Positional arguments for the shard client's method.
            **kwargs: Keyword arguments for the shard client's method.

        Returns:
            The result of the method call on the shard client.

        Raises:
            KeyError: If the `shard_id` is not found in `self.shard_clients`.
            AttributeError: If the `method_name` does not exist on the shard client or is not async.
            WDBXError: If the call to the shard client fails.
        """
        logger.info(f"Executing passthrough command '{method_name}' on shard '{shard_id}' with args: {args}, kwargs: {kwargs}")
        if shard_id not in self.shard_clients:
            logger.error(f"Shard ID '{shard_id}' not found for passthrough operation.")
            raise KeyError(f"Shard ID '{shard_id}' not found. Available shards: {list(self.shard_clients.keys())}")

        shard_client = self.shard_clients[shard_id]
        if not hasattr(shard_client, method_name):
            logger.error(f"Method '{method_name}' not found on shard client for shard '{shard_id}'.")
            raise AttributeError(f"Method '{method_name}' not found on client for shard '{shard_id}'.")

        method_to_call = getattr(shard_client, method_name)
        if not asyncio.iscoroutinefunction(method_to_call):
            logger.error(f"Method '{method_name}' on shard client for shard '{shard_id}' is not an async coroutine function.")
            raise AttributeError(f"Method '{method_name}' on shard client for shard '{shard_id}' is not an async coroutine function.")

        # Construct the coroutine call
        coro = method_to_call(*args, **kwargs)

        # Use _run_async to execute it. This handles circuit breaking and retries for the shard call.
        # Note: Metrics from _run_async will be labeled with 'method_name' from the shard client's perspective.
        try:
            result = self._run_async(coro)
            logger.info(f"Passthrough command '{method_name}' on shard '{shard_id}' completed successfully.")
            return result
        except Exception as e:  # Catches WDBXError from _run_async or other direct exceptions
            logger.error(
                f"Error executing passthrough command '{method_name}' on shard '{shard_id}': {e}",
                exc_info=True,
            )
            # Re-raise as WDBXError or let specific WDBXError propagate
            if not isinstance(e, WDBXError):
                raise WDBXError(f"Shard passthrough command '{method_name}' on '{shard_id}' failed: {e}") from e
            raise


# Factory function for convenience, if needed
def initialize_backend(
    vector_dimension: int,
    enable_plugins: bool = True,
    thread_safe: bool = False,
    shards: Optional[Dict[str, Dict[str, Any]]] = None,
    auto_discovery: bool = False,
    discovery_port: int = 9999,
    rpc_port: Optional[int] = None,
    cache_size: int = 128,
    wdbx_repo_url: Optional[str] = None,
    **kwargs: Any,
) -> WDBX:
    client = WDBX(
        vector_dimension=vector_dimension,
        enable_plugins=enable_plugins,
        thread_safe=thread_safe,
        shards=shards,
        auto_discovery=auto_discovery,
        discovery_port=discovery_port,
        rpc_port=rpc_port,
        cache_size=cache_size,
        wdbx_repo_url=wdbx_repo_url,
        **kwargs,
    )
    client.initialize()
    return client
