from lylex import tokenize


def test_tokenize_basic():
    text = "Hello, world! This is a test."
    tokens = tokenize(text)
    assert isinstance(tokens, list)
    # Strip whitespace from each token for matching
    stripped = [tok.strip() for tok in tokens]
    assert "Hello" in stripped and "," in stripped and "world" in stripped and "!" in stripped


def test_tokenize_empty():
    assert tokenize("") == []


def test_tokenize_none():
    assert tokenize(None) == []
