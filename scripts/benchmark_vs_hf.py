"""Benchmark tokie vs HuggingFace tokenizers.

Run: python scripts/benchmark_vs_hf.py

Produces a comparison table and optional JSON output for visualization.
"""

import json
import sys
import time
from dataclasses import dataclass

import tokie
from tokenizers import Tokenizer as HFTokenizer


@dataclass
class BenchResult:
    name: str
    library: str
    time_ms: float
    throughput_mbs: float
    tokens: int


MODELS = [
    ("bert-base-uncased", "WordPiece"),
    ("openai-community/gpt2", "BPE"),
    ("meta-llama/Llama-3.2-1B", "BPE"),
    ("Qwen/Qwen3-0.6B", "BPE"),
    ("google/gemma-3-1b-pt", "SentencePiece"),
]

SHORT_TEXT = "The quick brown fox jumps over the lazy dog. " * 10  # ~450 bytes
MEDIUM_TEXT = SHORT_TEXT * 100  # ~45 KB
LONG_TEXT = SHORT_TEXT * 2000  # ~900 KB


def bench_encode(tokenizer, text: str, warmup: int = 3, iters: int = 10):
    """Benchmark single encode, return (time_ms, token_count)."""
    for _ in range(warmup):
        enc = tokenizer.encode(text) if hasattr(enc := tokenizer.encode(text), 'ids') else tokenizer.encode(text)

    times = []
    token_count = 0
    for _ in range(iters):
        t0 = time.perf_counter()
        enc = tokenizer.encode(text)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        token_count = len(enc.ids) if hasattr(enc, 'ids') else len(enc.ids)

    median = sorted(times)[len(times) // 2]
    return median * 1000, token_count


def bench_encode_batch(tokenizer, texts: list[str], warmup: int = 2, iters: int = 5):
    """Benchmark batch encode, return (time_ms, total_tokens)."""
    for _ in range(warmup):
        tokenizer.encode_batch(texts)

    times = []
    total_tokens = 0
    for _ in range(iters):
        t0 = time.perf_counter()
        results = tokenizer.encode_batch(texts)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        total_tokens = sum(len(r.ids) for r in results)

    median = sorted(times)[len(times) // 2]
    return median * 1000, total_tokens


def bench_decode(tokenizer, token_ids: list[int], warmup: int = 3, iters: int = 10):
    """Benchmark decode, return time_ms."""
    for _ in range(warmup):
        tokenizer.decode(token_ids)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        tokenizer.decode(token_ids)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return sorted(times)[len(times) // 2] * 1000


def load_tokie(repo_id: str):
    return tokie.Tokenizer.from_pretrained(repo_id)


def load_hf(repo_id: str):
    return HFTokenizer.from_pretrained(repo_id)


def verify_correctness(tokie_tok, hf_tok, text: str, model_name: str):
    """Verify tokie produces the same token IDs as HF."""
    tokie_ids = tokie_tok.encode(text, add_special_tokens=False).ids
    hf_ids = hf_tok.encode(text, add_special_tokens=False).ids
    if tokie_ids != hf_ids:
        print(f"  WARNING: Token mismatch for {model_name}!")
        print(f"    tokie: {len(tokie_ids)} tokens, first 10: {tokie_ids[:10]}")
        print(f"    hf:    {len(hf_ids)} tokens, first 10: {hf_ids[:10]}")
        return False
    return True


def run_benchmarks():
    results = []
    text_sizes = [
        ("short (~450B)", SHORT_TEXT),
        ("medium (~45KB)", MEDIUM_TEXT),
        ("long (~900KB)", LONG_TEXT),
    ]

    print("=" * 80)
    print("tokie vs HuggingFace tokenizers benchmark")
    print("=" * 80)
    print()

    for repo_id, model_type in MODELS:
        print(f"Model: {repo_id} ({model_type})")
        print("-" * 60)

        try:
            t0 = time.perf_counter()
            tokie_tok = load_tokie(repo_id)
            tokie_load_ms = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            hf_tok = load_hf(repo_id)
            hf_load_ms = (time.perf_counter() - t0) * 1000
        except Exception as e:
            print(f"  SKIP: {e}")
            print()
            continue

        print(f"  Load time: tokie={tokie_load_ms:.1f}ms  hf={hf_load_ms:.1f}ms  ({hf_load_ms/tokie_load_ms:.1f}x)")

        # Correctness check
        correct = verify_correctness(tokie_tok, hf_tok, MEDIUM_TEXT, repo_id)
        print(f"  Correctness: {'PASS' if correct else 'FAIL'}")

        # Encode benchmarks at different sizes
        for size_name, text in text_sizes:
            text_bytes = len(text.encode("utf-8"))

            tokie_ms, tokie_tokens = bench_encode(tokie_tok, text)
            hf_ms, hf_tokens = bench_encode(hf_tok, text)

            tokie_mbs = text_bytes / (tokie_ms / 1000) / 1e6
            hf_mbs = text_bytes / (hf_ms / 1000) / 1e6
            speedup = hf_ms / tokie_ms

            print(f"  Encode {size_name}:")
            print(f"    tokie: {tokie_ms:8.2f}ms ({tokie_mbs:7.1f} MB/s, {tokie_tokens} tokens)")
            print(f"    hf:    {hf_ms:8.2f}ms ({hf_mbs:7.1f} MB/s, {hf_tokens} tokens)")
            print(f"    speedup: {speedup:.1f}x")

            results.append({
                "model": repo_id, "model_type": model_type,
                "benchmark": f"encode_{size_name}", "text_bytes": text_bytes,
                "tokie_ms": tokie_ms, "hf_ms": hf_ms, "speedup": speedup,
                "tokie_tokens": tokie_tokens, "hf_tokens": hf_tokens,
            })

        # Batch encode (100 medium texts)
        batch_texts = [MEDIUM_TEXT] * 100
        tokie_ms, tokie_total = bench_encode_batch(tokie_tok, batch_texts, warmup=1, iters=3)
        hf_ms, hf_total = bench_encode_batch(hf_tok, batch_texts, warmup=1, iters=3)
        total_bytes = sum(len(t.encode("utf-8")) for t in batch_texts)
        speedup = hf_ms / tokie_ms
        print(f"  Batch encode (100 x 45KB):")
        print(f"    tokie: {tokie_ms:8.1f}ms ({total_bytes / (tokie_ms / 1000) / 1e6:.1f} MB/s)")
        print(f"    hf:    {hf_ms:8.1f}ms ({total_bytes / (hf_ms / 1000) / 1e6:.1f} MB/s)")
        print(f"    speedup: {speedup:.1f}x")

        results.append({
            "model": repo_id, "model_type": model_type,
            "benchmark": "batch_encode_100x45KB", "text_bytes": total_bytes,
            "tokie_ms": tokie_ms, "hf_ms": hf_ms, "speedup": speedup,
        })

        # Decode
        enc = tokie_tok.encode(LONG_TEXT, add_special_tokens=False)
        token_ids = enc.ids
        tokie_dec_ms = bench_decode(tokie_tok, token_ids)
        hf_dec_ms = bench_decode(hf_tok, token_ids)
        dec_speedup = hf_dec_ms / tokie_dec_ms
        print(f"  Decode ({len(token_ids)} tokens):")
        print(f"    tokie: {tokie_dec_ms:8.2f}ms")
        print(f"    hf:    {hf_dec_ms:8.2f}ms")
        print(f"    speedup: {dec_speedup:.1f}x")

        results.append({
            "model": repo_id, "model_type": model_type,
            "benchmark": "decode", "tokens": len(token_ids),
            "tokie_ms": tokie_dec_ms, "hf_ms": hf_dec_ms, "speedup": dec_speedup,
        })

        print()

    # Summary table
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Model':<35} {'Benchmark':<25} {'Speedup':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['model']:<35} {r['benchmark']:<25} {r['speedup']:>7.1f}x")

    # Save JSON
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to benchmark_results.json")


if __name__ == "__main__":
    run_benchmarks()
