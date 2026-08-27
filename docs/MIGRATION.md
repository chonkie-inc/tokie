# Migrating from HuggingFace Tokenizers to tokie

## Python: 5-Line Migration

```diff
- from tokenizers import Tokenizer
+ import tokie

- tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
+ tokenizer = tokie.Tokenizer.from_pretrained("bert-base-uncased")

  # encode() returns the same structure
  encoding = tokenizer.encode("Hello, world!")
  encoding.ids             # identical token IDs
  encoding.attention_mask   # identical attention mask
  encoding.type_ids         # identical type IDs
```

## Rust: Same Change

```diff
- use tokenizers::Tokenizer;
+ use tokie::Tokenizer;

- let tokenizer = Tokenizer::from_pretrained("bert-base-uncased", None)?;
+ let tokenizer = Tokenizer::from_pretrained("bert-base-uncased")?;

  let encoding = tokenizer.encode("Hello, world!", true);
  // encoding.ids, encoding.attention_mask, encoding.type_ids — all identical
```

## API Comparison

| HuggingFace | tokie | Notes |
|---|---|---|
| `Tokenizer.from_pretrained(repo)` | `Tokenizer.from_pretrained(repo)` | Loads `.tkz` first, falls back to `tokenizer.json` |
| `tokenizer.encode(text)` | `tokenizer.encode(text)` | Returns `Encoding` with `ids`, `attention_mask`, `type_ids` |
| `tokenizer.encode(text, pair)` | `tokenizer.encode_pair(text, pair)` | Separate method for pairs |
| `tokenizer.encode_batch(texts)` | `tokenizer.encode_batch(texts)` | Parallel across all CPU cores |
| `tokenizer.decode(ids)` | `tokenizer.decode(ids)` | Returns `str` or `None` |
| `tokenizer.decode_batch(seqs)` | `tokenizer.decode_batch(seqs)` | |
| `tokenizer.token_to_id(token)` | `tokenizer.token_to_id(token)` | |
| `tokenizer.id_to_token(id)` | `tokenizer.id_to_token(id)` | |
| `tokenizer.get_vocab()` | `tokenizer.get_vocab()` | `dict[str, int]` |
| `tokenizer.get_vocab_size()` | `tokenizer.vocab_size` | Property instead of method |
| `tokenizer.enable_padding(...)` | `tokenizer.enable_padding(...)` | Same kwargs |
| `tokenizer.enable_truncation(...)` | `tokenizer.enable_truncation(...)` | Same kwargs |
| `tokenizer.no_padding()` | `tokenizer.no_padding()` | |
| `tokenizer.no_truncation()` | `tokenizer.no_truncation()` | |
| `tokenizer.save(path)` | `tokenizer.save(path)` | Saves as `.tkz` (10x smaller) |
| `tokenizer.padding` | `tokenizer.pad_token_id` | Pad token ID property |
| N/A | `tokenizer.count_tokens(text)` | Fast token counting without full encoding |
| N/A | `tokenizer.count_tokens_batch(texts)` | Parallel token counting |

## What's Different

**Pair encoding** uses a separate method:
```python
# HuggingFace
encoding = tokenizer.encode("query", "document")

# tokie
encoding = tokenizer.encode_pair("query", "document")
```

**`add_special_tokens` defaults to `True`** (same as HuggingFace):
```python
# Both libraries
encoding = tokenizer.encode("Hello", add_special_tokens=True)   # default
encoding = tokenizer.encode("Hello", add_special_tokens=False)  # raw tokens only
```

**Save format** is `.tkz` binary (not JSON):
```python
# Save as .tkz (10x smaller, 5ms load time)
tokenizer.save("model.tkz")
tokenizer = tokie.Tokenizer.from_file("model.tkz")

# Can still load from tokenizer.json
tokenizer = tokie.Tokenizer.from_json("tokenizer.json")
```

## What's NOT Supported

tokie is an **inference-only** tokenizer. These training/modification features are not available:

- `add_tokens()` / `add_special_tokens()` — vocabulary modification
- `train()` / `train_from_iterator()` — training new tokenizers
- `Tokenizer.from_buffer()` — loading from bytes
- Custom normalizers/pre-tokenizers/decoders at runtime
- `post_process()` as a standalone method

If you need these features, use HuggingFace tokenizers for training, then load the resulting `tokenizer.json` with tokie for inference.

## Supported Models

tokie supports any model with a `tokenizer.json` on HuggingFace, including:

- **BPE**: GPT-2, GPT-4 (cl100k), GPT-4o (o200k), Llama 3/4, Mistral, Phi, Qwen, CodeLlama
- **WordPiece**: BERT, MiniLM, BGE, GTE, ModernBERT, cross-encoders
- **SentencePiece BPE**: T5, XLM-R, Gemma
- **Unigram**: XLM-R (via SentencePiece)

Pre-built `.tkz` files for 60+ popular models are available at [tokiers on HuggingFace](https://huggingface.co/tokiers).
