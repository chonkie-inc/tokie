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

tokenizer = tokie.Tokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.encode("Hello, world!")
text = tokenizer.decode(tokens)
print(tokenizer.vocab_size)  # 30522
```

### Rust

```rust
use tokie::Tokenizer;

let tokenizer = Tokenizer::from_pretrained("bert-base-uncased")?;
let tokens = tokenizer.encode("Hello, world!", true);
let text = tokenizer.decode(&tokens).unwrap();
```
