import jax
import jax.numpy as jnp
import pytest
import torch

from lylex.model import (
    JAXEmbeddingModel,
    TorchEmbeddingModel,
    jax_distributed,
    wrap_torch_ddp,
)


def test_torch_embedding_forward():
    input_dim, hidden_dim, output_dim = 10, 8, 4
    model = TorchEmbeddingModel(input_dim, hidden_dim, output_dim)
    x = torch.randn(2, input_dim)
    y = model(x)
    assert y.shape == (2, output_dim)


@pytest.mark.skipif(not torch.distributed.is_available(), reason="Distributed backend not available")
def test_wrap_ddp_without_init(monkeypatch):
    model = TorchEmbeddingModel(5, 5, 5)
    # Ensure distributed is not initialized
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    with pytest.raises(RuntimeError):
        wrap_torch_ddp(model)


def test_jax_embedding_forward():
    input_dim, hidden_dim, output_dim = 6, 5, 3
    model = JAXEmbeddingModel(hidden_dim=hidden_dim, output_dim=output_dim)
    params = model.init(jax.random.PRNGKey(0), jnp.ones((2, input_dim)))
    y = model.apply(params, jnp.ones((2, input_dim)))
    assert y.shape == (2, output_dim)


def test_jax_pmap():
    def fn(x):
        return x * 2

    pfn = jax_distributed(fn)
    # pmap over devices: input must match number of devices
    ndevices = jax.device_count()
    args = jnp.arange(ndevices)
    out = pfn(args)
    assert isinstance(out, jnp.ndarray)
    assert out.shape[0] == ndevices
    assert all(out[i] == args[i] * 2 for i in range(ndevices))
