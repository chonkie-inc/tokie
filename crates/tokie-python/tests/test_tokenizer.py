import pytest
import tokie


def test_from_pretrained_bert():
    t = tokie.Tokenizer.from_pretrained("bert-base-uncased")
    assert t.vocab_size == 30522
    assert repr(t) == "Tokenizer(vocab_size=30522)"


def test_encode_decode_roundtrip():
    t = tokie.Tokenizer.from_pretrained("bert-base-uncased")
    text = "Hello, world!"
    tokens = t.encode(text)
    decoded = t.decode(tokens)
    # BERT adds [CLS] and [SEP], decode includes them
    assert isinstance(tokens, list)
    assert all(isinstance(tok, int) for tok in tokens)
    assert isinstance(decoded, str)


def test_encode_without_special_tokens():
    t = tokie.Tokenizer.from_pretrained("bert-base-uncased")
    with_special = t.encode("hello", add_special_tokens=True)
    without_special = t.encode("hello", add_special_tokens=False)
    # With special tokens should be longer ([CLS] + tokens + [SEP])
    assert len(with_special) == len(without_special) + 2


def test_count_tokens():
    t = tokie.Tokenizer.from_pretrained("bert-base-uncased")
    count = t.count_tokens("Hello, world!")
    tokens = t.encode("Hello, world!", add_special_tokens=False)
    assert count == len(tokens)


def test_encode_pair():
    t = tokie.Tokenizer.from_pretrained("bert-base-uncased")
    pair = t.encode_pair("How are you?", "I am fine.")
    assert isinstance(pair.ids, list)
    assert isinstance(pair.attention_mask, list)
    assert isinstance(pair.type_ids, list)
    assert len(pair) == len(pair.ids)
    assert len(pair.attention_mask) == len(pair.ids)
    assert len(pair.type_ids) == len(pair.ids)
    # type_ids should have 0s for first seq, 1s for second
    assert 0 in pair.type_ids
    assert 1 in pair.type_ids


def test_encode_bytes():
    t = tokie.Tokenizer.from_pretrained("bert-base-uncased")
    tokens = t.encode_bytes(b"hello")
    assert isinstance(tokens, list)
    assert len(tokens) > 0


def test_decode_bytes():
    t = tokie.Tokenizer.from_pretrained("bert-base-uncased")
    tokens = t.encode("hello")
    raw = t.decode_bytes(tokens)
    assert isinstance(raw, bytes)


def test_save_load_roundtrip(tmp_path):
    t = tokie.Tokenizer.from_pretrained("bert-base-uncased")
    path = str(tmp_path / "test.tkz")
    t.save(path)
    t2 = tokie.Tokenizer.from_file(path)
    assert t2.vocab_size == t.vocab_size
    assert t2.encode("test") == t.encode("test")


def test_error_handling():
    with pytest.raises(tokie.TokieError):
        tokie.Tokenizer.from_file("/nonexistent.tkz")


def test_gpt2():
    t = tokie.Tokenizer.from_pretrained("openai-community/gpt2")
    tokens = t.encode("Hello, world!", add_special_tokens=False)
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    decoded = t.decode(tokens)
    assert decoded == "Hello, world!"


def test_encode_batch_basic():
    t = tokie.Tokenizer.from_pretrained("bert-base-uncased")
    texts = ["Hello world", "How are you?", "This is a test", "Goodbye", "One more"]
    batch = t.encode_batch(texts)
    assert len(batch) == 5
    for i, text in enumerate(texts):
        assert batch[i] == t.encode(text)


def test_encode_batch_empty():
    t = tokie.Tokenizer.from_pretrained("bert-base-uncased")
    assert t.encode_batch([]) == []


def test_encode_batch_preserves_order():
    t = tokie.Tokenizer.from_pretrained("bert-base-uncased")
    texts = [f"sentence number {i} with some content" for i in range(50)]
    batch = t.encode_batch(texts)
    assert len(batch) == 50
    for i, text in enumerate(texts):
        assert batch[i] == t.encode(text), f"Mismatch at index {i}"


def test_encode_batch_without_special_tokens():
    t = tokie.Tokenizer.from_pretrained("bert-base-uncased")
    texts = ["hello", "world"]
    with_special = t.encode_batch(texts, add_special_tokens=True)
    without_special = t.encode_batch(texts, add_special_tokens=False)
    for ws, wos in zip(with_special, without_special):
        assert len(ws) == len(wos) + 2  # [CLS] + tokens + [SEP]


def test_count_tokens_batch():
    t = tokie.Tokenizer.from_pretrained("bert-base-uncased")
    texts = ["Hello world", "How are you?", "Test"]
    counts = t.count_tokens_batch(texts)
    assert len(counts) == 3
    for i, text in enumerate(texts):
        assert counts[i] == t.count_tokens(text)


def test_count_tokens_batch_empty():
    t = tokie.Tokenizer.from_pretrained("bert-base-uncased")
    assert t.count_tokens_batch([]) == []
