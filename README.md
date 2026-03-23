<div align="center">

![tokie](assets/tokie.png)

*50x faster tokenization, 10x smaller models, 100% accurate drop-in for HuggingFace*

[![Crates.io](https://img.shields.io/crates/v/tokie)](https://crates.io/crates/tokie)
[![PyPI](https://img.shields.io/pypi/v/tokie)](https://pypi.org/project/tokie/)
[![License](https://img.shields.io/crates/l/tokie)](LICENSE-MIT)

</div>

**tokie** is a fast, correct tokenizer library built in Rust with Python bindings. It loads any tokenizer from HuggingFace — BPE (GPT, Llama, Mistral), WordPiece (BERT), Unigram (T5) — and tokenizes text up to 50x faster by replacing regex with custom parsers and parallelizing across all cores.

![benchmark](assets/benchmark.png)

## Install

### Python

```bash
uv pip install tokie
# or
pip install tokie
```

### Rust

```toml
[dependencies]
tokie = { version = "0.0.3", features = ["hf"] }
```

## Quick Start

### Python

```python
import tokie

# Load any HuggingFace tokenizer
tokenizer = tokie.Tokenizer.from_pretrained("bert-base-uncased")

# Encode and decode
tokens = tokenizer.encode("Hello, world!")
text = tokenizer.decode(tokens)

# Count tokens without allocating
count = tokenizer.count_tokens("Hello, world!")

# Vocabulary size
print(tokenizer.vocab_size)  # 30522
```

### Rust

```rust
use tokie::Tokenizer;

let tokenizer = Tokenizer::from_pretrained("bert-base-uncased")?;
let tokens = tokenizer.encode("Hello, world!", true);
let text = tokenizer.decode(&tokens).unwrap();
```

## Examples

### Cross-Encoder Pair Encoding

For rerankers and cross-encoders that need sentence pairs with token type IDs:

```python
pair = tokenizer.encode_pair("How are you?", "I am fine.")
pair.ids             # [101, 2129, 2024, 2017, 1029, 102, 1045, 2572, 2986, 1012, 102]
pair.attention_mask  # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
pair.type_ids        # [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
```

### Save and Load `.tkz` Files

tokie's binary format is ~10x smaller than `tokenizer.json` and loads in ~5ms:

```python
tokenizer.save("model.tkz")
tokenizer = tokie.Tokenizer.from_file("model.tkz")
```

`from_pretrained()` automatically tries `.tkz` first, falling back to `tokenizer.json`.

## Why tokie?

When I started building [Chonkie](https://github.com/chonkie-inc/chonkie), the biggest bottleneck wasn't chunking — it was tokenization. Fast, effective chunking is a function of fast, effective tokenization, and we were spending more time counting tokens than actually doing the work.

I love and respect HuggingFace tokenizers. But they can't be *that* fast because of fundamentals — most of the slowdown comes from regex, and no amount of optimization on top of regex will fix that. So we wrote custom parsers that replace the regex entirely, and got order-of-magnitude speedups.

The other problem: there was no single tokenizer format that could load both tiktoken-style tokenizers (GPT, Claude) *and* everything else on HuggingFace (BERT, Llama, T5, Mistral). That's crazy ambitious for one project — you need algorithms fundamentally designed for each method of tokenization, and you need to figure out individually how to make them all fast.

Not many know this, but before Chonkie I actually released [AutoTikTokenizer](https://github.com/bhavnick/autotiktokenizer) because I believed tiktoken's BPE engine could load everything on HuggingFace. I was wrong. You need backtracking BPE, simple BPE, SentencePiece BPE, WordPiece, and Unigram — each with their own tricks.

And then there's the fun part: tokenization is embarrassingly parallel if you chunk text properly, so we used our own chunking algorithm to distribute tokenization across all cores.

The result is **tokie** — one tokenizer to rule them all.

## Acknowledgements

tokie builds on ideas and techniques from several excellent projects:

- [HuggingFace tokenizers](https://github.com/huggingface/tokenizers) — the gold standard for tokenizer correctness
- [tiktoken](https://github.com/openai/tiktoken) — OpenAI's fast BPE implementation
- [GitHub's rust-gems](https://github.com/github/rust-gems) — the backtracking BPE approach using Aho-Corasick automata
- [memchunk](https://github.com/chonkie-inc/chunk) — SIMD-accelerated text chunking for parallel tokenization

## Citation

If you use tokie in your research, please cite it as follows:

```bibtex
@software{tokie2025,
  author = {Minhas, Bhavnick},
  title = {tokie: Fast, correct tokenizer library for every HuggingFace model},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/chonkie-inc/tokie}},
}
```
