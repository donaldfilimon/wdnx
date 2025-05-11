"""
# db.py - Database wrapper for Lylex using WDBX as the vector store.
"""

import atexit
import logging
import os  # For checking OPENAI_API_KEY
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple

from wdbx import WDBX
from wdbx.self_update import CodeUpdater
from wdbx.self_update_advanced import AdvancedUpdater

try:
    from .ai import (
        LylexModelHandler,
    )  # For future use, or if it develops an embedding API
except Exception:
    LylexModelHandler = None

try:
    from .embeddings import EmbeddingHandler
except Exception:
    EmbeddingHandler = None

logger = logging.getLogger(__name__)

InteractionResult = Tuple[int, float, Dict[str, Any]]
SearchResults = List[InteractionResult]

class LylexDB:
    """
    Wrapper around WDBX for storing and retrieving prompt-response interactions.
    Supports custom embedding functions to generate real vectors.
    """

    def __init__(
        self,
        vector_dimension: int = 384,
        enable_plugins: bool = False,
        thread_safe: bool = False,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        shards: Optional[Dict[str, Dict[str, Any]]] = None,
        embedding_handler: Optional[EmbeddingHandler] = None,  # New parameter
        model_handler: Optional[LylexModelHandler] = None,  # New parameter (for future)
        **kwargs: Any,
    ):
        """
        Initialize the LylexDB.

        Parameters:
            vector_dimension: Dimension of embedding vectors.
            enable_plugins: Whether to enable WDBX plugins.
            thread_safe: Enable thread-safe client operations.
            embed_fn: Function to convert prompts to embedding vectors.
            shards: Optional sharding configuration.
            embedding_handler: Optional lylex.embeddings.EmbeddingHandler instance.
            model_handler: Optional lylex.ai.LylexModelHandler instance.
            kwargs: Additional arguments for WDBX initialization.
        """
        self.vector_dimension = vector_dimension
        self.embed_fn = embed_fn
        self.embedding_handler = embedding_handler
        self.model_handler = model_handler  # Stored for future use

        # Warn if vector_dimension seems mismatched with a provided OpenAI EmbeddingHandler
        if (
            self.embedding_handler
            and self.embedding_handler.model == "text-embedding-ada-002"
        ):
            openai_ada_dim = 1536
            if self.vector_dimension != openai_ada_dim:
                logger.warning(
                    f"LylexDB initialized with vector_dimension={self.vector_dimension}, "
                    f"but provided EmbeddingHandler uses '{self.embedding_handler.model}' which has dimension {openai_ada_dim}. "
                    f"Ensure dimensions match to avoid errors or truncation."
                )

        # Initialize WDBX client with optional sharding
        client_args: Dict[str, Any] = {
            "vector_dimension": vector_dimension,
            "enable_plugins": enable_plugins,
            "thread_safe": thread_safe,
        }
        if shards is not None:
            client_args["shards"] = shards
        client_args.update(kwargs)
        self.client = WDBX(**client_args)
        self.client.initialize()
        atexit.register(self.shutdown)
        logger.info("LylexDB initialized with WDBX client.")

    def _get_vector(self, prompt: str) -> List[float]:
        """
        Generate an embedding vector for a prompt using embed_fn or a zero-vector fallback.
        Priority: embed_fn > embedding_handler (OpenAI) > model_handler (future) > zero-vector.
        """
        if self.embed_fn:
            return self.embed_fn(prompt)

        if self.embedding_handler and os.getenv("OPENAI_API_KEY"):
            try:
                vector = self.embedding_handler.embed([prompt])[0]
                if len(vector) != self.vector_dimension:
                    logger.error(
                        f"OpenAI EmbeddingHandler returned vector of dimension {len(vector)}, "
                        f"but LylexDB is configured for {self.vector_dimension}. Falling back to zero-vector."
                    )
                    return [0.0] * self.vector_dimension
                return vector
            except Exception as e:
                logger.error(
                    f"Error using EmbeddingHandler: {e}. Falling back to zero-vector."
                )
                return [0.0] * self.vector_dimension

        # Use model_handler for embedding if provided
        if self.model_handler and hasattr(self.model_handler, "get_embedding"):
            try:
                vector = self.model_handler.get_embedding(prompt)
                if len(vector) != self.vector_dimension:
                    logger.error(
                        f"ModelHandler returned vector of dimension {len(vector)}, "
                        f"but LylexDB is configured for {self.vector_dimension}. Falling back to zero-vector."
                    )
                    return [0.0] * self.vector_dimension
                return vector
            except Exception as e:
                logger.error(
                    f"Error using ModelHandler for embedding: {e}. Falling back to zero-vector."
                )
                return [0.0] * self.vector_dimension

        # Fallback to zero-vector
        return [0.0] * self.vector_dimension

    def store_interaction(
        self, prompt: str, response: str, metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Store a prompt-response interaction in the vector database.

        If an embedding function is provided, use it; otherwise store a zero-vector.

        Returns:
            The stored vector ID.
        """
        if metadata is None:
            metadata = {}
        metadata.update({"prompt": prompt, "response": response})
        vector = self._get_vector(prompt)
        # Use distributed store if shards configured
        if hasattr(self.client, "distributed_store") and self.client.shards:
            return self.client.distributed_store(prompt, vector, metadata)
        return self.client.store(vector, metadata)

    def bulk_store_interactions(
        self, items: List[Tuple[str, str, Optional[Dict[str, Any]]]]
    ) -> List[int]:
        """
        Store multiple prompt-response interactions in bulk. Returns list of stored vector IDs.
        """
        return [
            self.store_interaction(prompt, response, metadata)
            for prompt, response, metadata in items
        ]

    def search_interactions(
        self, prompt: str, limit: int = 5
    ) -> SearchResults:
        """
        Retrieve similar interactions based on prompt similarity.

        If an embedding function is provided, use it; otherwise use a zero-vector.

        Returns:
            A list of tuples (id, score, metadata).
        """
        vector = self._get_vector(prompt)
        # Use distributed search if shards configured
        if hasattr(self.client, "distributed_search") and self.client.shards:
            return self.client.distributed_search(prompt, vector, limit=limit)
        return self.client.search(vector, limit=limit)

    def broadcast_search_interactions(
        self, prompt: str, limit: int = 5
    ) -> Dict[str, SearchResults]:
        """
        Broadcast search across all shards, returning per-shard results.
        """
        vector = self._get_vector(prompt)
        if hasattr(self.client, "broadcast_search") and self.client.shard_clients:
            return self.client.broadcast_search(vector, limit=limit)
        # Fallback single shard
        results = self.client.search(vector, limit=limit)
        return {"default": results}

    def delete_interaction(self, interaction_id: int) -> bool:
        """Delete a stored interaction by ID."""
        return self.client.delete(interaction_id)

    def update_interaction_metadata(
        self, interaction_id: int, metadata: Dict[str, Any]
    ) -> bool:
        """Update metadata for a stored interaction."""
        return self.client.update_metadata(interaction_id, metadata)

    def get_interaction_metadata(self, interaction_id: int) -> Dict[str, Any]:
        """Retrieve metadata for a stored interaction."""
        return self.client.get_metadata(interaction_id)

    def count_interactions(self) -> int:
        """Return the total number of interactions stored."""
        return self.client.count()

    def ping_backend(self) -> bool:
        """Ping the backend to check if it's alive."""
        return self.client.ping()

    def flush_interactions(self) -> None:
        """Remove all stored interactions."""
        return self.client.flush()

    def export_interactions(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Export stored interactions as a list of dictionaries with id, score, and metadata.
        """
        # Determine fetch limit; default to total count
        if limit is None:
            limit = self.count_interactions()
        # Use zero-vector query to retrieve all entries
        vector = [0.0] * self.vector_dimension
        raw = self.client.search(vector, limit=limit)
        # Format as list of dicts
        return [
            {"id": _id, "score": score, "metadata": meta} for _id, score, meta in raw
        ]

    def begin_transaction(self) -> Any:
        """
        Begin a new transaction in WDBX.
        """
        return self.client.begin_transaction()

    def commit_transaction(self) -> Any:
        """
        Commit the current transaction.
        """
        return self.client.commit_transaction()

    def rollback_transaction(self) -> Any:
        """
        Rollback the current transaction.
        """
        return self.client.rollback_transaction()

    @contextmanager
    def transaction(self):
        """
        Transaction context manager: begin, commit on success, rollback on error.
        Usage: with db.transaction():
                  # do operations
        """
        try:
            self.begin_transaction()
            yield
            self.commit_transaction()
        except Exception:
            self.rollback_transaction()
            raise

    def shutdown(self) -> None:
        """
        Shutdown the underlying WDBX client and release resources.
        """
        self.client.shutdown()
        logger.info("LylexDB WDBX client shutdown.")

    def store_model(
        self, file_path: str, metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Store a serialized model artifact via the underlying WDBX client.

        Returns:
            The artifact vector ID.
        """
        return self.client.store_model(file_path, metadata)

    def load_model(self, artifact_id: int, output_path: str) -> None:
        """
        Retrieve a stored model artifact and write it locally via the underlying WDBX client.
        """
        self.client.load_model(artifact_id, output_path)

    def neural_backtrace(self, prompt: str) -> Dict[str, Any]:
        """
        Perform a neural backtrace for the given prompt.

        Converts the prompt to an embedding vector (using embed_fn or zero-vector)
        and calls the WDBX neural_backtrack API.

        Returns:
            A dictionary containing backtrace results from the WDBX backend.
        """
        # Generate query vector
        vector = self._get_vector(prompt)
        # Call the underlying WDBX neural backtrack
        return self.client.neural_backtrack(vector)

    def backtrace_pattern(self, vector: List[float]) -> Dict[str, Any]:
        """
        Directly perform neural backtrace on a raw embedding vector.

        Parameters:
            vector: The embedding vector to backtrace.

        Returns:
            A dictionary of backtrace results.
        """
        return self.client.neural_backtrack(vector)

    def update_code(self, file_path: str, new_code: str) -> None:
        """
        Apply a code patch to a file and optionally reload modules.
        """
        CodeUpdater().apply_patch(file_path, new_code)

    def reload_self(self, module_name: str) -> None:
        """
        Reload a module after its source has been updated.
        """
        CodeUpdater().reload_module(module_name)

    def schedule_self_update(
        self, interval: int, repo_dir: str, module_paths: Optional[List[str]] = None
    ) -> None:
        """
        Schedule periodic Git-based code updates for Lylex modules.
        """
        adv = AdvancedUpdater(repo_url=None)
        adv.schedule(interval, adv.update_from_git, repo_dir, module_paths)

    def stop_self_update(self) -> None:
        """
        Stop scheduled updates.
        """
        AdvancedUpdater().stop()

    def git_update(
        self, local_dir: str, module_paths: Optional[List[str]] = None
    ) -> None:
        """
        Perform an immediate Git pull and apply patches.
        """
        adv = AdvancedUpdater(repo_url=None)
        adv.update_from_git(local_dir, module_paths)

    def rollback_update(
        self, file_path: str, backup_file: Optional[str] = None
    ) -> None:
        """
        Roll back a patched file.
        """
        AdvancedUpdater().rollback(file_path, backup_file)

    def ai_update(
        self,
        file_path: str,
        instruction: str,
        model_name: str = "gpt2",
        backend: str = "pt",
        memory_limit: int = 5,
    ) -> None:
        """
        Use an AI agent to generate and apply a patch based on the given instruction.
        """
        adv = AdvancedUpdater(repo_url=None)
        adv.ai_update(
            file_path,
            instruction,
            model_name=model_name,
            backend=backend,
            memory_limit=memory_limit,
        )

    def get_outdated_packages(self) -> list[Dict[str, Any]]:
        """
        Get a list of outdated pip packages without recording them.
        """
        from wdbx.update_utils import get_outdated_packages

        return get_outdated_packages()

    def search_outdated_packages(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Search WDBX for stored outdated package entries and return their metadata.
        """
        # Construct a zero-vector query to retrieve package entries
        vector = [0.0] * self.vector_dimension
        raw_results = self.client.search(vector, limit=limit)
        # Filter entries that have 'package' metadata
        return [
            metadata for _id, _score, metadata in raw_results if "package" in metadata
        ]
