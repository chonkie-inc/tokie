<div align="center">

![tokie](assets/tokie.png)

*50x faster tokenization, 10x smaller models, 100% accurate drop-in for HuggingFace*

[![Crates.io](https://img.shields.io/crates/v/tokie)](https://crates.io/crates/tokie)
[![PyPI](https://img.shields.io/pypi/v/tokie)](https://pypi.org/project/tokie/)
[![License](https://img.shields.io/crates/l/tokie)](LICENSE-MIT)

</div>

## Why tokie?

When I started building [Chonkie](https://github.com/chonkie-inc/chonkie), the biggest bottleneck wasn't chunking — it was tokenization. Fast, effective chunking is a function of fast, effective tokenization, and we were spending more time counting tokens than actually doing the work.

I love and respect HuggingFace tokenizers. But they can't be *that* fast because of fundamentals — most of the slowdown comes from regex, and no amount of optimization on top of regex will fix that. So we wrote custom parsers that replace the regex entirely, and got order-of-magnitude speedups.

The other problem: there was no single tokenizer format that could load both tiktoken-style tokenizers (GPT, Claude) *and* everything else on HuggingFace (BERT, Llama, T5, Mistral). That's crazy ambitious for one project — you need algorithms fundamentally designed for each method of tokenization, and you need to figure out individually how to make them all fast.

Not many know this, but before Chonkie I actually released [AutoTikTokenizer](https://github.com/bhavnick/autotiktokenizer) because I believed tiktoken's BPE engine could load everything on HuggingFace. I was wrong. You need backtracking BPE, simple BPE, SentencePiece BPE, WordPiece, and Unigram — each with their own tricks.

And then there's the fun part: tokenization is embarrassingly parallel if you chunk text properly, so we used our own chunking algorithm to distribute tokenization across all cores.

The result is **tokie** — one tokenizer to rule them all.

## Features

- **50x faster** than HuggingFace tokenizers on common workloads
- **10x smaller** model files with the `.tkz` binary format (~5ms load time)
- **100% accurate** — identical output to HuggingFace, token for token
- **One format** — loads tiktoken, HuggingFace tokenizer.json, and .tkz files
- **Every algorithm** — BPE (backtracking, simple, SentencePiece), WordPiece, Unigram
- **Custom parsers** — no regex engine, just purpose-built tokenization parsers
- **Rust + Python** — native Rust library with PyO3 bindings

## Quick Start

### Python

```bash
pip install tokie
```

```python
import tokie

tokenizer = tokie.Tokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.encode("Hello, world!")
text = tokenizer.decode(tokens)
print(tokenizer.vocab_size)  # 30522
```

### Rust

```toml
[dependencies]
tokie = { version = "0.0.3", features = ["hf"] }
```

```rust
use tokie::Tokenizer;

let tokenizer = Tokenizer::from_pretrained("bert-base-uncased")?;
let tokens = tokenizer.encode("Hello, world!", true);
let text = tokenizer.decode(&tokens).unwrap();
```

## Supported Models

tokie works with any HuggingFace tokenizer. Some highlights:

| Encoder | Models |
|---------|--------|
| **BPE (Backtracking)** | GPT-2, cl100k, o200k, Llama 3/4, Mistral, Phi, Qwen |
| **BPE (SentencePiece)** | T5, XLM-RoBERTa, CodeLlama, Gemma |
| **WordPiece** | BERT, MiniLM, BGE, GTE, E5, ModernBERT |
| **Unigram** | ALBERT, XLNet, Marian |

## The `.tkz` Format

tokie's binary format pre-builds the Aho-Corasick automaton and stores it directly — no parsing, no regex compilation on load:

```python
# Save
tokenizer.save("model.tkz")

# Load (~5ms, 10x smaller than tokenizer.json)
tokenizer = tokie.Tokenizer.from_file("model.tkz")
```

`from_pretrained()` automatically tries `.tkz` first, falling back to `tokenizer.json`.

## Pair Encoding

For cross-encoder and reranker models:

```python
pair = tokenizer.encode_pair("How are you?", "I am fine.")
pair.ids             # [101, 2129, 2024, ..., 102]
pair.attention_mask  # [1, 1, 1, ..., 1]
pair.type_ids        # [0, 0, 0, ..., 1, 1, 1]
```
