import sys
import types

from lylex import tokenize


def test_tokenize_with_hf_tokenizer(monkeypatch):
    # Simulate a HuggingFace tokenizer available
    class DummyTokenizer:
        def tokenize(self, text):
            return ["hf_tok1", "hf_tok2"]

    class DummyHandler:
        def __init__(self, backend):
            self.tokenizer = DummyTokenizer()

    # Patch LylexModelHandler to use DummyHandler
    monkeypatch.setattr("lylex.ai.LylexModelHandler", DummyHandler)
    tokens = tokenize("testing hf")
    assert tokens == ["hf_tok1", "hf_tok2"]


def test_tokenize_with_tiktoken(monkeypatch):
    # Simulate HF tokenizer failure
    class BadHandler:
        def __init__(self, backend):
            raise Exception("no hf")

    monkeypatch.setattr("lylex.ai.LylexModelHandler", BadHandler)
    # Insert dummy tiktoken module
    dummy_enc = types.SimpleNamespace(
        encode=lambda text: [0, 1], decode=lambda token_list: f"tok{token_list[0]}"
    )
    dummy_tiktoken = types.SimpleNamespace(get_encoding=lambda name: dummy_enc)
    monkeypatch.setitem(sys.modules, "tiktoken", dummy_tiktoken)
    tokens = tokenize("ab")
    assert tokens == ["tok0", "tok1"]


def test_tokenize_fallback(monkeypatch):
    # Simulate no HF tokenizer and no tiktoken
    class BadHandler:
        def __init__(self, backend):
            raise Exception("no hf")

    monkeypatch.setattr("lylex.ai.LylexModelHandler", BadHandler)
    # Ensure tiktoken import fails or decodes incorrectly
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    text = "one two three"
    tokens = tokenize(text)
    assert tokens == text.split()
