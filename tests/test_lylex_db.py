import pytest

import lylex.db as db_module

class DummyClient:
    def __init__(self, vector_dimension, enable_plugins, thread_safe, **kwargs):
        self._client = self
        self.vector_dimension = vector_dimension
        self.stored = []
    def initialize(self):
        pass
    def store(self, vector, metadata):
        self.stored.append((vector, metadata))
        return 123
    def search(self, vector, limit):
        return [(1, 0.9, {"prompt":"hi","response":"hello"})]
    def shutdown(self):
        pass

@pytest.fixture(autouse=True)
def patch_wdbx(monkeypatch):
    # Monkeypatch WDBX in db_module
    monkeypatch.setattr(db_module, 'WDBX', DummyClient)
    return monkeypatch


def test_store_interaction(monkeypatch):
    db = db_module.LylexDB(vector_dimension=5)
    vid = db.store_interaction("hi", "hello")
    assert vid == 123
    assert db.client.stored[0][1]["prompt"] == "hi"
    assert db.client.stored[0][1]["response"] == "hello"


def test_search_interactions(monkeypatch):
    db = db_module.LylexDB(vector_dimension=5)
    results = db.search_interactions("test", limit=1)
    assert isinstance(results, list)
    assert results[0][2]["response"] == "hello"


def test_shutdown(monkeypatch):
    db = db_module.LylexDB(vector_dimension=5)
    # Should not raise
    db.shutdown() 