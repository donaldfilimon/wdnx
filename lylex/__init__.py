"""
lylex - Lexical analysis utilities for the project.

This package provides functions to tokenize and analyze text.
"""

__all__ = ["tokenize"]

from .ai import LylexModelHandler, LylexAgent
from .db import LylexDB
from .model import (
    TorchEmbeddingModel,
    init_torch_distributed,
    wrap_torch_ddp,
    JAXEmbeddingModel,
    jax_distributed,
)
from .async_db import AsyncLylexDB
from .training import TrainingManager
from .vision import VisionModelHandler
from .code import CodeModelHandler
from .image import ImageGenerator
from .embeddings import EmbeddingHandler
from .brain import Brain
from .neuron import Neuron

__all__.extend([
    "LylexModelHandler",
    "LylexAgent",
    "LylexDB",
    "TorchEmbeddingModel",
    "init_torch_distributed",
    "wrap_torch_ddp",
    "JAXEmbeddingModel",
    "jax_distributed",
    "store_model", "load_model",
    "neural_backtrace", "backtrace_pattern",
    "AsyncLylexDB",
    "TrainingManager",
    "VisionModelHandler",
    "CodeModelHandler",
    "ImageGenerator",
    "EmbeddingHandler",
    "Brain",
    "Neuron",
])

def tokenize(text: str) -> list[str]:
    """
    Split input text into a list of tokens.
    Tries to use the loaded model's HuggingFace tokenizer (via LylexModelHandler).
    If no model tokenizer is available, falls back to simple whitespace splitting.
    If 'tiktoken' is installed and can be invoked, it will be used when no HF tokenizer is present.
    """
    if text is None:
        return []
    # Try HuggingFace tokenizer from LylexModelHandler
    try:
        from .ai import LylexModelHandler
        handler = LylexModelHandler(backend='pt')
        tok = getattr(handler, 'tokenizer', None)
        if tok is not None:
            # Use fast tokenizer's tokenize method
            return tok.tokenize(text)
    except Exception:
        pass
    # Try tiktoken if available
    try:
        import tiktoken
        # Use default encoding
        enc = tiktoken.get_encoding('cl100k_base')
        ids = enc.encode(text)
        tokens = [enc.decode([i]) for i in ids]
        return tokens
    except Exception:
        pass
    # Fallback to simple whitespace tokenizer
    return text.split() 