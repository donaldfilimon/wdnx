import pytest

from lylex.async_db import AsyncLylexDB


class DummyAsyncClient:
    def __init__(self, *args, **kwargs):
        self.shard_manager = None
        self.shard_clients = {}
        self.stored = []

    async def initialize(self):
        pass

    async def shutdown(self):
        pass

    async def store(self, vector, metadata):
        self.stored.append((vector, metadata))
        return 456

    async def search(self, vector, limit=5):
        return [(2, 0.8, {"prompt": "hi", "response": "hello async"})]


@pytest.fixture(autouse=True)
def patch_async_wdbx(monkeypatch):
    import lylex.async_db as mod

    monkeypatch.setattr(mod, "AsyncWDBX", DummyAsyncClient)
    return monkeypatch


@pytest.mark.asyncio
async def test_async_store_and_search(tmp_path):
    db = AsyncLylexDB(vector_dimension=3, embed_fn=lambda x: [1.0, 2.0, 3.0])
    await db.initialize()
    vid = await db.store_interaction("test?", "resp")
    assert vid == 456
    results = await db.search_interactions("test?", limit=1)
    assert isinstance(results, list)
    assert results[0][0] == 2
    await db.shutdown()
