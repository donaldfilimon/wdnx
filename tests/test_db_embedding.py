import pytest

import lylex.db as db_module
from lylex.db import LylexDB


class DummyClient:
    def __init__(self, vector_dimension, enable_plugins, thread_safe, **kwargs):
        pass

    def initialize(self):
        pass


@pytest.fixture(autouse=True)
def patch_wdbx(monkeypatch):
    # Monkeypatch WDBX client to avoid real WDBX initialization
    monkeypatch.setattr(db_module, "WDBX", DummyClient)
    return monkeypatch


# Helper embedding handler classes
def test_get_vector_with_custom_embed_fn():
    vector = [1.0, 2.0]
    db = LylexDB(vector_dimension=2, embed_fn=lambda x: vector)
    assert db._get_vector("prompt") == vector


@pytest.fixture(autouse=False)
def patch_env(monkeypatch):
    # Fixture to set OPENAI_API_KEY
    monkeypatch.setenv("OPENAI_API_KEY", "fake_key")
    yield
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_get_vector_with_embedding_handler(monkeypatch, patch_env):
    class GoodHandler:
        def __init__(self):
            self.model = "text-embedding-ada-002"

        def embed(self, texts):
            return [[0.5, 0.5]]

    handler = GoodHandler()
    db = LylexDB(vector_dimension=2, embedding_handler=handler)
    vec = db._get_vector("hello")
    assert vec == [0.5, 0.5]

    class BadHandler:
        def __init__(self):
            self.model = "text-embedding-ada-002"

        def embed(self, texts):
            return [[0.1]]

    bad_handler = BadHandler()
    db2 = LylexDB(vector_dimension=2, embedding_handler=bad_handler)
    vec2 = db2._get_vector("hello")
    assert vec2 == [0.0, 0.0]


def test_get_vector_with_model_handler():
    class DummyMH:
        def get_embedding(self, text):
            return [0.2, 0.3]

    mh = DummyMH()
    db = LylexDB(vector_dimension=2, model_handler=mh)
    vec = db._get_vector("hi there")
    assert vec == [0.2, 0.3]


def test_get_vector_fallback():
    db = LylexDB(vector_dimension=3)
    vec = db._get_vector("anything")
    assert vec == [0.0, 0.0, 0.0]
