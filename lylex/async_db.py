"""
async_db.py - Asynchronous database wrapper for Lylex using AsyncWDBX.
"""

import asyncio
import atexit
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from wdbx import AsyncWDBX

from .db import SearchResults

try:
    from .ai import LylexModelHandler
except Exception:
    LylexModelHandler = None

try:
    from .embeddings import EmbeddingHandler
except ImportError:
    EmbeddingHandler = None

logger = logging.getLogger(__name__)

# Type alias for embedding vectors
Vector = List[float]


class AsyncLylexDB:
    """
    Asynchronous wrapper around AsyncWDBX for storing and querying prompt-response interactions.
    """

    def __init__(
        self,
        vector_dimension: int = 384,
        enable_plugins: bool = False,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        shards: Optional[Dict[str, Dict[str, Any]]] = None,
        embedding_handler: Optional[EmbeddingHandler] = None,
        model_handler: Optional[LylexModelHandler] = None,
        **kwargs: Any,
    ):
        self.vector_dimension = vector_dimension
        self.embed_fn = embed_fn
        self.embedding_handler = embedding_handler
        self.model_handler = model_handler

        if self.embedding_handler and self.embedding_handler.model == "text-embedding-ada-002":
            openai_ada_dim = 1536
            if self.vector_dimension != openai_ada_dim:
                logger.warning(f"AsyncLylexDB initialized with vector_dimension={self.vector_dimension}, " f"but provided EmbeddingHandler uses '{self.embedding_handler.model}' which has dimension {openai_ada_dim}. " f"Ensure dimensions match to avoid errors or truncation.")

        # Initialize AsyncWDBX client with optional sharding
        client_args: Dict[str, Any] = {
            "vector_dimension": vector_dimension,
            "enable_plugins": enable_plugins,
        }
        if shards is not None:
            client_args["shards"] = shards
        client_args.update(kwargs)
        self.client = AsyncWDBX(**client_args)

    async def initialize(self) -> None:
        """Initialize the AsyncWDBX backend."""
        await self.client.initialize()
        atexit.register(lambda: None)  # No shutdown on exit for async
        logger.info("AsyncLylexDB initialized.")

    async def shutdown(self) -> None:
        """Shutdown the AsyncWDBX backend."""
        await self.client.shutdown()
        logger.info("AsyncLylexDB shutdown.")

    async def _get_vector(self, prompt: str) -> Vector:
        """
        Generate an embedding vector for a prompt using embed_fn or zero-vector fallback.
        Priority: embed_fn > embedding_handler (OpenAI) > model_handler (future) > zero-vector.
        """
        if self.embed_fn:
            if asyncio.iscoroutinefunction(self.embed_fn):
                return await self.embed_fn(prompt)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, self.embed_fn, prompt)

        if self.embedding_handler and os.getenv("OPENAI_API_KEY"):
            try:
                loop = asyncio.get_event_loop()
                vector = await loop.run_in_executor(None, lambda: self.embedding_handler.embed([prompt])[0])

                if len(vector) != self.vector_dimension:
                    logger.error(f"OpenAI EmbeddingHandler returned vector of dimension {len(vector)}, " f"but AsyncLylexDB is configured for {self.vector_dimension}. Falling back to zero-vector.")
                    return [0.0] * self.vector_dimension
                return vector
            except Exception as e:
                logger.error(f"Error using EmbeddingHandler: {e}. Falling back to zero-vector.")
                return [0.0] * self.vector_dimension

        # Use model_handler for embedding if provided
        if self.model_handler and hasattr(self.model_handler, "get_embedding"):
            try:
                loop = asyncio.get_event_loop()
                vector = await loop.run_in_executor(None, self.model_handler.get_embedding, prompt)
                if len(vector) != self.vector_dimension:
                    logger.error(f"ModelHandler returned vector of dimension {len(vector)}, " f"but AsyncLylexDB is configured for {self.vector_dimension}. Falling back to zero-vector.")
                    return [0.0] * self.vector_dimension
                return vector
            except Exception as e:
                logger.error(f"Error using ModelHandler for embedding: {e}. Falling back to zero-vector.")
                return [0.0] * self.vector_dimension

        # Fallback to zero-vector
        return [0.0] * self.vector_dimension

    async def store_interaction(self, prompt: str, response: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Asynchronously store a prompt-response interaction.
        """
        if metadata is None:
            metadata = {}
        metadata.update({"prompt": prompt, "response": response})
        vector = await self._get_vector(prompt)
        if hasattr(self.client, "distributed_store") and getattr(self.client, "shard_manager", None):
            return await self.client.distributed_store(prompt, vector, metadata)
        return await self.client.store(vector, metadata)

    async def bulk_store_interactions(self, items: List[Tuple[str, str, Optional[Dict[str, Any]]]]) -> List[int]:
        """
        Asynchronously store multiple prompt-response interactions in bulk.
        """
        results = []
        for prompt, response, metadata in items:
            vid = await self.store_interaction(prompt, response, metadata)
            results.append(vid)
        return results

    async def search_interactions(self, prompt: str, limit: int = 5) -> SearchResults:
        """
        Asynchronously search for similar interactions based on prompt.
        """
        vector = await self._get_vector(prompt)
        if hasattr(self.client, "distributed_search") and getattr(self.client, "shard_manager", None):
            return await self.client.distributed_search(prompt, vector, limit=limit)
        return await self.client.search(vector, limit=limit)

    async def broadcast_search_interactions(self, prompt: str, limit: int = 5) -> Dict[str, SearchResults]:
        """
        Broadcast search across all shards asynchronously.
        """
        vector = await self._get_vector(prompt)
        if hasattr(self.client, "broadcast_search") and getattr(self.client, "shard_clients", None):
            return await self.client.broadcast_search(vector, limit=limit)
        results = await self.client.search(vector, limit=limit)
        return {"default": results}

    async def delete_interaction(self, vector_id: int) -> bool:
        """
        Delete a stored interaction by its vector ID asynchronously.
        """
        return await self.client.delete(vector_id)

    async def update_interaction_metadata(self, vector_id: int, metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for a stored interaction asynchronously.
        """
        return await self.client.update_metadata(vector_id, metadata)

    async def begin_transaction(self) -> Any:
        """
        Begin a new transaction asynchronously.
        """
        return await self.client.begin_transaction()

    async def commit_transaction(self) -> Any:
        """
        Commit the current transaction asynchronously.
        """
        return await self.client.commit_transaction()

    async def rollback_transaction(self) -> Any:
        """
        Rollback the current transaction asynchronously.
        """
        return await self.client.rollback_transaction()

    async def get_interaction_metadata(self, interaction_id: int) -> Dict[str, Any]:
        """Retrieve metadata for a stored interaction asynchronously."""
        return await self.client.get_metadata(interaction_id)

    async def count_interactions(self) -> int:
        """Return the total number of interactions stored asynchronously."""
        return await self.client.count()

    async def ping_backend(self) -> bool:
        """Ping the backend to check if it's alive asynchronously."""
        return await self.client.ping()

    async def flush_interactions(self) -> None:
        """Remove all stored interactions asynchronously."""
        return await self.client.flush()
