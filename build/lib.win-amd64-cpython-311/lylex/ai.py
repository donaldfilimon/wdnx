"""
ai.py - Self-learning AI agent using PyTorch and JAX local models.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

import jax

try:
    from jax import config as jax_config
except ImportError:
    jax_config = None
import torch

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        FlaxAutoModelForCausalLM,
        Trainer,
        TrainingArguments,
    )
except Exception:
    AutoModelForCausalLM = AutoTokenizer = FlaxAutoModelForCausalLM = Trainer = (
        TrainingArguments
    ) = None

from .db import LylexDB

logger = logging.getLogger(__name__)

__all__ = ["LylexModelHandler", "LylexAgent"]


class LylexModelHandler:
    """
    Singleton handler to load and generate from local language models in PyTorch or JAX.
    """

    _instances = {}

    def __new__(cls, backend: str = "pt") -> LylexModelHandler:
        key = backend.lower()
        if key not in cls._instances:
            cls._instances[key] = super().__new__(cls)
        return cls._instances[key]

    def __init__(self, backend: str = "pt"):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self.backend = backend.lower()
        if self.backend not in ["pt", "jax"]:
            logger.warning(f"Unsupported backend '{self.backend}', defaulting to 'pt'.")
            self.backend = "pt"
        self.current_model = None
        self.tokenizer = None
        self.model = None
        self._initialized = True
        logger.info(f"LylexModelHandler initialized with backend: {self.backend}")
        # Configure JAX to prioritize GPU if available
        if self.backend == "jax" and jax_config is not None:
            devices = jax.devices()
            platform_name = (
                "gpu" if any("gpu" in str(d).lower() for d in devices) else "cpu"
            )
            jax_config.update("jax_platform_name", platform_name)
        elif self.backend == "jax":
            logger.warning("jax.config not available; skipping platform configuration")

    def load_model(self, model_name: str) -> None:
        """
        Load a specified language model and its tokenizer, caching for future use.

        Parameters:
            model_name: Hugging Face model identifier (e.g., 'gpt2').
        """
        if self.current_model == model_name and self.model is not None:
            return
        logger.info(f"Loading model '{model_name}' (backend: {self.backend})")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.backend == "jax":
            self.model = FlaxAutoModelForCausalLM.from_pretrained(model_name)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            if torch.cuda.is_available():
                self.model.to("cuda")
        self.current_model = model_name
        logger.info(f"Model '{model_name}' loaded successfully.")

    def generate(self, prompt: str, max_length: int = 128, **kwargs: Any) -> str:
        """
        Generate text from the loaded model given a prompt.

        Parameters:
            prompt: Input text prompt.
            max_length: Maximum number of tokens to generate.
            kwargs: Additional generation parameters (e.g., temperature).

        Returns:
            Generated text string.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("No model loaded; call load_model() first.")
        inputs = self.tokenizer(
            prompt, return_tensors=("jax" if self.backend == "jax" else "pt")
        )
        if self.backend == "jax":
            outputs = self.model.generate(**inputs, max_length=max_length, **kwargs)
            generated = self.tokenizer.decode(
                outputs.sequences[0], skip_special_tokens=True
            )
        else:
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            # Use mixed precision on GPU for faster inference
            ctx = torch.cuda.amp.autocast() if device.type == "cuda" else nullcontext()
            with torch.no_grad(), ctx:
                outputs = self.model.generate(**inputs, max_length=max_length, **kwargs)
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated

    def train(
        self, train_dataset: Any, output_dir: str, **training_args_kwargs: Any
    ) -> None:
        """
        Fine-tune the loaded language model on a given dataset.

        Parameters:
            train_dataset: A Dataset or pathlib to training data compatible with HF Trainer.
            output_dir: Directory to save trained model and tokenizer.
            training_args_kwargs: Keyword arguments for TrainingArguments (e.g., num_train_epochs).

        Raises:
            RuntimeError: If no model is loaded.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("No model loaded; call load_model() first.")
        # Prepare training arguments
        args = TrainingArguments(output_dir=output_dir, **training_args_kwargs)
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
        )
        # Train and save
        trainer.train()
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model fine-tuned and saved to {output_dir}")

    def get_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding vector for input text using the model's token embeddings and mean pooling.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("No model loaded; call load_model() first.")
        # Tokenize with attention mask
        if self.backend == "jax":
            inputs = self.tokenizer(
                text, return_tensors="jax", return_attention_mask=True
            )
            # Ensure hidden states are returned
            self.model.config.output_hidden_states = True
            outputs = self.model(
                **inputs, params=self.model.params, output_hidden_states=True
            )
            hidden_states = outputs.hidden_states[-1]  # [1, seq_len, dim]
            mask = inputs["attention_mask"]
            summed = (hidden_states * mask[..., None]).sum(axis=1)
            counts = mask.sum(axis=1, keepdims=True)
            pooled = summed / counts
            return list(pooled[0].tolist())
        else:
            inputs = self.tokenizer(
                text, return_tensors="pt", return_attention_mask=True
            )
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            embeds = self.model.get_input_embeddings()(
                inputs["input_ids"]
            )  # [1, seq_len, dim]
            mask = inputs["attention_mask"].unsqueeze(-1)
            masked_embeds = embeds * mask
            summed = masked_embeds.sum(dim=1)  # [1, dim]
            counts = mask.sum(dim=1)  # [1,1]
            pooled = summed / counts
            return pooled.squeeze(0).detach().cpu().tolist()


class LylexAgent:
    """
    High-level agent to interface with LylexModelHandler for self-learning.
    """

    def __init__(
        self,
        model_name: str,
        backend: str = "pt",
        memory_db: Optional[LylexDB] = None,
        memory_limit: int = 5,
    ) -> None:
        """
        Initialize the LylexAgent with a language model.

        Parameters:
            model_name: Hugging Face model identifier.
            backend: 'pt' for PyTorch or 'jax' for JAX backend.
            memory_db: Optional LylexDB instance for conversational memory.
            memory_limit: Number of past interactions to retrieve as context.
        """
        self.handler = LylexModelHandler(backend)
        self.handler.load_model(model_name)
        # Initialize conversational memory
        self.memory_db = (
            memory_db if memory_db is not None else LylexDB(vector_dimension=384)
        )
        self.memory_limit = memory_limit

    def ask(self, text: str, max_length: int = 128) -> str:
        """
        Generate a response from the agent based on input text.

        Parameters:
            text: Input prompt string.
            max_length: Maximum tokens to generate.

        Returns:
            Response string generated by the model.
        """
        return self.handler.generate(text, max_length=max_length)

    def chat(self, prompt: str, max_length: int = 128, **kwargs: Any) -> str:
        """
        Conversational chat with retrieval-augmented memory.

        Retrieves past interactions from memory, builds context, generates a response,
        and stores the new interaction.

        Parameters:
            prompt: User input prompt.
            max_length: Maximum generation tokens.
            **kwargs: Additional generation parameters.

        Returns:
            The generated response.
        """
        # Retrieve relevant past memories
        memories = self.memory_db.search_interactions(prompt, limit=self.memory_limit)
        # Build context dialogue
        context = ""
        for _, _, meta in memories:
            context += (
                f"User: {meta.get('prompt')}\nAssistant: {meta.get('response')}\n"
            )
        # Compose full prompt with context
        full_prompt = context + f"User: {prompt}\nAssistant:"
        # Generate response using model handler
        response = self.handler.generate(full_prompt, max_length=max_length, **kwargs)
        # Store the new interaction in memory
        self.memory_db.store_interaction(prompt, response)
        return response

    def packages_to_upgrade(self) -> list[Dict[str, Any]]:
        """
        Retrieve metadata for packages needing upgrade by querying the vector store.
        """
        try:
            return self.memory_db.search_outdated_packages()
        except AttributeError:
            return []
