"""
model.py - Neural network and distributed compute utilities for the lylex package.
"""
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import jax
import jax.numpy as jnp
from flax import linen as nnx

__all__ = [
    "TorchEmbeddingModel",
    "init_torch_distributed",
    "wrap_torch_ddp",
    "JAXEmbeddingModel",
    "jax_distributed",
]

class TorchEmbeddingModel(nn.Module):
    """
    Simple MLP embedding model in PyTorch.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def init_torch_distributed(backend: str = "gloo", init_method: str = "env://") -> None:
    """
    Initialize PyTorch distributed process group.
    """
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method=init_method)


def wrap_torch_ddp(model: nn.Module) -> DDP:
    """
    Wrap a PyTorch model in DistributedDataParallel.
    """
    if not dist.is_initialized():
        raise RuntimeError("Distributed not initialized. Call init_torch_distributed first.")
    return DDP(model)

class JAXEmbeddingModel(nnx.Module):
    """
    Simple MLP embedding model in Flax (JAX).
    """
    hidden_dim: int
    output_dim: int

    @nnx.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nnx.Dense(self.hidden_dim)(x)
        x = nnx.relu(x)
        x = nnx.Dense(self.output_dim)(x)
        return x


def jax_distributed(fn):
    """
    Parallelize a JAX function across devices using pmap.
    """
    return jax.pmap(fn) 