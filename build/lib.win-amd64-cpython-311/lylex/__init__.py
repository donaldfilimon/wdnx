"""
lylex - Lexical analysis utilities for the project.

This package provides functions to tokenize and analyze text.
"""

import importlib
import logging

# Attempt to import tiktoken. If it's not found, tokenize will have one less option.
try:
    import tiktoken
except ImportError:
    tiktoken = None
    logging.info(
        "tiktoken library not found. Tokenization will rely on LylexModelHandler "
        "or fallback to whitespace splitting if LylexModelHandler is unavailable/fails."
    )

# Make symbols from submodules available in the lylex namespace.
# This ensures that `from lylex import SomeClass` works for items in __all__.
# These imports assume the submodules (ai, db, models, training, utils) and the
# specified names exist within them. If not, an ImportError will be raised at startup.

from .ai import (
    LylexModelHandler,
    LylexAgent,
    VisionModelHandler,
    CodeModelHandler,
    ImageGenerator,
    EmbeddingHandler,
    Brain,
    Neuron,
    neural_backtrace,
    backtrace_pattern,
)
from .db import LylexDB, AsyncLylexDB
from .model import (
    TorchEmbeddingModel,
    JAXEmbeddingModel,
    store_model,
    load_model,
    init_torch_distributed,
    wrap_torch_ddp,
    jax_distributed,
)
from .training import TrainingManager

__all__ = [
    "tokenize",
    "LylexModelHandler",
    "LylexAgent",
    "LylexDB",
    "TorchEmbeddingModel",
    "init_torch_distributed",
    "wrap_torch_ddp",
    "JAXEmbeddingModel",
    "jax_distributed",
    "store_model",
    "load_model",
    "neural_backtrace",
    "backtrace_pattern",
    "AsyncLylexDB",
    "TrainingManager",
    "VisionModelHandler",
    "CodeModelHandler",
    "ImageGenerator",
    "EmbeddingHandler",
    "Brain",
    "Neuron",
]


def tokenize(text: str) -> list[str]:
    """
    Split input text into a list of tokens.
    Tries to use the loaded model's HuggingFace tokenizer (via LylexModelHandler,
    specifically if it appears to be patched).
    If that fails or is not applicable, will attempt tiktoken fallback;
    otherwise defaults to whitespace splitting.
    """
    if not text:  # Handles None and empty string by returning an empty list
        return []

    # Try HuggingFace tokenizer via LylexModelHandler from lylex.ai
    # This part specifically looks for a "patched" Handler, where HandlerClass.__module__
    # is different from the module name of ai_mod ('lylex.ai').
    try:
        # Dynamically import lylex.ai to check its current state for LylexModelHandler
        ai_mod = importlib.import_module("lylex.ai")
        HandlerClass = getattr(ai_mod, "LylexModelHandler", None)

        if (
            HandlerClass
            and hasattr(HandlerClass, "__module__")
            and HandlerClass.__module__ != ai_mod.__name__
            and callable(HandlerClass)
        ):
            # Custom/Patched Handler provided
            handler_instance = HandlerClass(
                backend="pt"
            )  # Assumes this constructor API
            tokenizer_obj = getattr(handler_instance, "tokenizer", None)

            if tokenizer_obj and hasattr(tokenizer_obj, "tokenize"):
                return tokenizer_obj.tokenize(text)
            else:
                logging.debug(
                    "Patched LylexModelHandler found but no 'tokenizer' attribute "
                    "or 'tokenize' method on tokenizer."
                )
        else:
            logging.debug(
                "LylexModelHandler not found in lylex.ai, or not patched, or not callable."
            )

    except ImportError:
        logging.debug(
            "lylex.ai module not found, cannot use LylexModelHandler for tokenization."
        )
    except Exception as e:
        logging.warning(
            f"Error attempting to use LylexModelHandler tokenizer: {e}", exc_info=True
        )
        # Proceed to tiktoken fallback

    # Try tiktoken if HuggingFace tokenizer attempt failed, was skipped, or wasn't applicable
    if tiktoken:
        try:
            # Consider making the encoding model configurable if necessary
            enc = tiktoken.get_encoding("cl100k_base")
            ids = enc.encode(text)
            # Decode each token ID individually into a string
            return [enc.decode([i]) for i in ids]
        except Exception as e:
            logging.warning(f"Error using tiktoken: {e}", exc_info=True)
            # Proceed to whitespace splitting fallback
    else:
        logging.debug(
            "tiktoken library not available or import failed; cannot use for tokenization."
        )

    # Fallback to simple whitespace splitting
    logging.debug(f"Falling back to whitespace tokenization for: {text[:30]}...")
    return text.split()
