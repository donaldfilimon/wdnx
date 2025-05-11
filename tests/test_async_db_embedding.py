import pytest
import asyncio
import lylex.async_db as mod
from lylex.async_db import AsyncLylexDB

class DummyAsyncClient:
    def __init__(self, *args, **kwargs):
        pass
    async def initialize(self):
        pass
    async def shutdown(self):
        pass

@pytest.fixture(autouse=True)
def patch_async_wdbx(monkeypatch):
    # Monkeypatch AsyncWDBX client to avoid real initialization
    monkeypatch.setattr(mod, 'AsyncWDBX', DummyAsyncClient)
    return monkeypatch

@pytest.mark.asyncio
async def test_async_get_vector_with_custom_embed_fn():
    db = AsyncLylexDB(vector_dimension=2, embed_fn=lambda x: [1.0, 2.0])
    vector = await db._get_vector("prompt")
    assert vector == [1.0, 2.0]

@pytest.mark.asyncio
async def test_async_get_vector_with_model_handler():
    class DummyMH:
        def get_embedding(self, text):
            return [0.7, 0.8]
    mh = DummyMH()
    db = AsyncLylexDB(vector_dimension=2, model_handler=mh)
    vector = await db._get_vector("hi")
    assert vector == [0.7, 0.8]

@pytest.mark.asyncio
async def test_async_get_vector_fallback():
    db = AsyncLylexDB(vector_dimension=3)
    vector = await db._get_vector("hello")
    assert vector == [0.0, 0.0, 0.0] 