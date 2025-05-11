"""
utils.py - Utility functions, decorators, and circuit breaker setup for the wdbx package.
"""

import asyncio
import logging
import os
from functools import wraps
from typing import Any, Callable, Coroutine, Sequence, TypeVar

import numpy as np
from pybreaker import CircuitBreaker, CircuitBreakerListener
from tenacity import retry, stop_after_attempt, wait_exponential

from .metrics import CB_FAILURES, CB_STATE, CB_TOTAL_FAILURES

T = TypeVar("T")
logger = logging.getLogger(__name__)

# Retry and circuit breaker config from environment
RETRY_ATTEMPTS = int(os.getenv("WDBX_RETRY_ATTEMPTS", "3"))
RETRY_WAIT_MULTIPLIER = float(os.getenv("WDBX_RETRY_WAIT_MULTIPLIER", "0.5"))
RETRY_WAIT_MIN = float(os.getenv("WDBX_RETRY_WAIT_MIN", "1"))
RETRY_WAIT_MAX = float(os.getenv("WDBX_RETRY_WAIT_MAX", "10"))
CIRCUIT_BREAKER = CircuitBreaker(
    fail_max=int(os.getenv("WDBX_CB_FAIL_MAX", "5")),
    reset_timeout=int(os.getenv("WDBX_CB_RESET_TIMEOUT", "60")),
)


class PrometheusCircuitBreakerListener(CircuitBreakerListener):
    def state_change(self, cb, old_state, new_state):
        CB_STATE.set(1 if new_state.name == "open" else 0)

    def failure(self, cb, exc):
        CB_TOTAL_FAILURES.inc()
        try:
            CB_FAILURES.set(cb.fail_counter)
        except Exception:
            pass


CIRCUIT_BREAKER.add_listener(PrometheusCircuitBreakerListener())


def with_retry(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to retry operations with exponential backoff."""
    return retry(
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=RETRY_WAIT_MULTIPLIER, min=RETRY_WAIT_MIN, max=RETRY_WAIT_MAX),
        reraise=True,
    )(func)


def sync_to_async(func: Callable[..., T]) -> Callable[..., Coroutine[Any, Any, T]]:
    """Convert a synchronous function to an asynchronous one."""

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> T:
        return func(*args, **kwargs)

    return wrapper


def async_to_sync(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., T]:
    """Convert an asynchronous function to a synchronous one."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        return asyncio.run(func(*args, **kwargs))

    return wrapper


# --- Vector Processing Utilities ---
__faiss_available = False
try:
    import faiss

    __faiss_available = True
except ImportError:
    faiss = None


def normalize(vector: Sequence[float], ord: int = 2) -> list[float]:
    """
    Normalize a vector to unit length using the specified norm (L2 by default).
    """
    arr = np.array(vector, dtype=float)
    norm = np.linalg.norm(arr, ord=ord)
    if norm == 0:
        return arr.tolist()
    return (arr / norm).tolist()


def cosine_similarity(v1: Sequence[float], v2: Sequence[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    """
    a = np.array(v1, dtype=float)
    b = np.array(v2, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def bulk_similarity(vectors: Sequence[Sequence[float]], queries: Sequence[Sequence[float]]) -> np.ndarray:
    """
    Compute similarity matrix between a set of stored vectors and query vectors using cosine similarity.
    Returns a 2D numpy array of shape (len(queries), len(vectors)).
    """
    mat = np.array(vectors, dtype=float)
    qmat = np.array(queries, dtype=float)
    # Normalize rows


__all__ = [
    "retry",
    "stop_after_attempt",
    "wait_exponential",
    "CIRCUIT_BREAKER",
    "with_retry",
    "sync_to_async",
    "async_to_sync",
]
