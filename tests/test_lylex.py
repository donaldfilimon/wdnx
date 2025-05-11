from lylex import tokenize


def test_tokenize_basic():
    text = "Hello, world! This is a test."
    tokens = tokenize(text)
    assert isinstance(tokens, list)
    assert "Hello," in tokens and "world!" in tokens


def test_tokenize_empty():
    assert tokenize("") == []


def test_tokenize_none():
    assert tokenize(None) == []
