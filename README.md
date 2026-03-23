<div align="center">

![tokie](assets/tokie.png)

# tokie

[![Crates.io](https://img.shields.io/crates/v/tokie)](https://crates.io/crates/tokie)
[![PyPI](https://img.shields.io/pypi/v/tokie)](https://pypi.org/project/tokie/)
[![License](https://img.shields.io/crates/l/tokie)](LICENSE-MIT)

*50x faster tokenization, 10x smaller models, 100% accurate drop-in for HuggingFace*

[Install](#install) •
[Quick Start](#quick-start) •
[Examples](#examples) •
[Benchmarks](#benchmarks) •
[Why tokie?](#why-tokie)

</div>

**tokie** is a Rust tokenizer library (with Python bindings) that can load any tokenizer on HuggingFace and tokenize up to 80x faster. It supports every major algorithm — BPE, WordPiece, SentencePiece, and Unigram — and is 100% token-accurate, every time.

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

## Benchmarks

All benchmarks run on War and Peace (3.36 MB) on an Apple M3 Pro. tokie produces **identical output** to HuggingFace tokenizers — every token matches, every time.

### BPE Encoding (GPT-2, Llama, Mistral)

For tiktoken-style BPE models (GPT-2, cl100k, o200k, Llama 3), tokie uses a backtracking encoder built on an Aho-Corasick automaton. Instead of iteratively merging byte pairs, it does a greedy longest-match in O(n) time, with backtracking only when adjacent tokens form invalid pairs. Combined with parallel chunking across all cores, this gives **298 MB/s** — 51x faster than HuggingFace and 21x faster than tiktoken.

![BPE encoding speed](assets/benchmark.png)

### WordPiece (BERT, MiniLM, BGE, GTE)

WordPiece tokenizers use a different algorithm — greedy longest-match prefix search over a vocabulary trie. tokie uses a pre-built Double-Array trie for O(n) lookup with excellent cache locality, combined with a custom BERT pretokenizer that avoids regex entirely. The result is **82x faster** than HuggingFace tokenizers on BERT, with identical output (737,710 tokens match exactly).

![WordPiece encoding speed](assets/benchmark_wordpiece.png)

### SentencePiece BPE (T5, XLM-R, Gemma)

SentencePiece-style models use a different merge algorithm with non-topological rank orders. tokie uses a radix heap with O(1) amortized operations that exploits BPE's monotonic rank property. Text is chunked at metaspace boundaries using SIMD-accelerated splitting, then encoded in parallel. This gives **7.7x** faster throughput than HuggingFace tokenizers.

![SentencePiece BPE speed](assets/benchmark_sentencepiece.png)

### Tokenizer Loading

Loading a tokenizer from `tokenizer.json` requires JSON parsing, vocabulary construction, and — for BPE models — building the Aho-Corasick automaton from scratch. tiktoken similarly has to parse its BPE data and compile regex patterns on every load. tokie's `.tkz` binary format stores all of this pre-built: the Double-Array Aho-Corasick (DAAC) automaton state, the normalized vocabulary, and the encoder configuration are serialized directly. Loading becomes a near-zero-cost deserialization — no parsing, no construction — achieving **4x–9x faster** cold load times than HuggingFace and **2x–3x faster** than tiktoken.

![Tokenizer loading time](assets/benchmark_loading.png)

## Why tokie?

When I started building [Chonkie](https://github.com/chonkie-inc/chonkie), the biggest bottleneck wasn't chunking — it was tokenization. We were spending more time counting tokens than actually chunking text. I love and respect HuggingFace tokenizers, but the fundamental problem is regex: most of the time in tokenization is spent in the pretokenizer's regex engine, and no amount of optimization on top of regex will fix that.

So we replaced regex entirely. tokie uses hand-written parsers for each pretokenization pattern — GPT-2, cl100k, o200k, BERT — that understand the exact character classes needed without the overhead of a general-purpose regex engine. That alone gets you a 10–20x speedup on pretokenization.

The second problem was that no single library could load everything. tiktoken handles GPT-style tokenizers but nothing else. HuggingFace tokenizers supports everything but is built around a one-size-fits-all architecture. I actually tried to solve this before with [AutoTikTokenizer](https://github.com/bhavnick/autotiktokenizer), believing tiktoken's BPE engine could handle all of HuggingFace. I was wrong — you need fundamentally different algorithms for each encoder type: backtracking BPE for tiktoken-style models, heap-based BPE for models with non-topological merge orders, radix-heap BPE for SentencePiece, plus WordPiece and Unigram each with their own tricks.

The third insight was parallelism. Tokenization is embarrassingly parallel if you split text at the right boundaries. We use [memchunk](https://github.com/chonkie-inc/chunk) to SIMD-split text into chunks that respect token boundaries, then encode each chunk on a separate core and concatenate. This gives near-linear scaling — about 5x on 8 cores.

Finally, we built the `.tkz` format to eliminate load-time overhead. A `tokenizer.json` file has to be parsed, validated, and used to reconstruct all the internal data structures (including the Aho-Corasick automaton, which is expensive to build for large vocabularies). The `.tkz` format stores the pre-built DAAC automaton, vocabulary, and configuration as a flat binary — loading is just deserialization, no construction required. This cuts load times from 150ms to 15ms for large models like O200K.

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
