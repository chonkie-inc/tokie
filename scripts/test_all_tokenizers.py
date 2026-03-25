#!/usr/bin/env python3
"""Test all tokiers/ models against HuggingFace reference on enwik8.

For each model:
1. Load from tokiers/ via tokie (from_pretrained)
2. Load original model via HuggingFace tokenizers
3. Encode enwik8 (first 1MB for speed) with both
4. Compare token-by-token
5. Report results as JSON for README generation
"""

import json
import sys
import time
from pathlib import Path

# Model mapping: tokiers/ repo -> original HF model
MODELS = {
    # BPE byte-level
    "tokiers/gpt2": "openai-community/gpt2",
    "tokiers/cl100k": None,  # tiktoken, no HF model
    "tokiers/o200k": None,  # tiktoken, no HF model
    "tokiers/roberta-base": "FacebookAI/roberta-base",
    "tokiers/phi-2": "microsoft/phi-2",
    "tokiers/Phi-3-mini-4k-instruct": "microsoft/Phi-3-mini-4k-instruct",
    "tokiers/ModernBERT-base": "answerdotai/ModernBERT-base",
    "tokiers/CodeLlama-7b-hf": "codellama/CodeLlama-7b-hf",
    "tokiers/Llama-3.2-1B": "meta-llama/Llama-3.2-1B",
    "tokiers/Llama-4-Scout-17B-16E": "meta-llama/Llama-4-Scout-17B-16E",
    "tokiers/Mistral-7B-v0.1": "mistralai/Mistral-7B-v0.1",
    "tokiers/Mistral-Nemo-Base-2407": "mistralai/Mistral-Nemo-Base-2407",
    "tokiers/Mixtral-8x7B-v0.1": "mistralai/Mixtral-8x7B-v0.1",
    "tokiers/Qwen2-7B": "Qwen/Qwen2-7B",
    "tokiers/Qwen3-Embedding-0.6B": "Qwen/Qwen3-Embedding-0.6B",
    "tokiers/Qwen3-Embedding-4B": "Qwen/Qwen3-Embedding-4B",
    "tokiers/Qwen3-Embedding-8B": "Qwen/Qwen3-Embedding-8B",
    "tokiers/nomic-embed-text-v1": "nomic-ai/nomic-embed-text-v1",

    # WordPiece (BERT-family)
    "tokiers/bert-base-uncased": "google-bert/bert-base-uncased",
    "tokiers/all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "tokiers/all-MiniLM-L12-v2": "sentence-transformers/all-MiniLM-L12-v2",
    "tokiers/all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    "tokiers/bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
    "tokiers/bge-large-en-v1.5": "BAAI/bge-large-en-v1.5",
    "tokiers/bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
    "tokiers/bge-en-icl": "BAAI/bge-en-icl",
    "tokiers/e5-base-v2": "intfloat/e5-base-v2",
    "tokiers/e5-large-v2": "intfloat/e5-large-v2",
    "tokiers/e5-small-v2": "intfloat/e5-small-v2",
    "tokiers/gte-base": "thenlper/gte-base",
    "tokiers/gte-large": "thenlper/gte-large",
    "tokiers/gte-small": "thenlper/gte-small",
    "tokiers/gte-Qwen2-7B-instruct": "Alibaba-NLP/gte-Qwen2-7B-instruct",
    "tokiers/ms-marco-MiniLM-L-4-v2": "cross-encoder/ms-marco-MiniLM-L-4-v2",
    "tokiers/ms-marco-MiniLM-L-6-v2": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "tokiers/mxbai-embed-large-v1": "mixedbread-ai/mxbai-embed-large-v1",
    "tokiers/mxbai-embed-2d-large-v1": "mixedbread-ai/mxbai-embed-2d-large-v1",
    "tokiers/mxbai-embed-xsmall-v1": "mixedbread-ai/mxbai-embed-xsmall-v1",
    "tokiers/deepset-mxbai-embed-de-large-v1": "mixedbread-ai/deepset-mxbai-embed-de-large-v1",

    # Jina (BPE byte-level)
    "tokiers/jina-embeddings-v2-base-en": "jinaai/jina-embeddings-v2-base-en",
    "tokiers/jina-embeddings-v2-base-code": "jinaai/jina-embeddings-v2-base-code",
    "tokiers/jina-embeddings-v3": "jinaai/jina-embeddings-v3",
    "tokiers/jina-embeddings-v4": "jinaai/jina-embeddings-v4",

    # Cohere (BPE)
    "tokiers/Cohere-embed-english-v3.0": "Cohere/Cohere-embed-english-v3.0",
    "tokiers/Cohere-embed-english-light-v3.0": "Cohere/Cohere-embed-english-light-v3.0",
    "tokiers/Cohere-embed-multilingual-v3.0": "Cohere/Cohere-embed-multilingual-v3.0",
    "tokiers/Cohere-embed-multilingual-light-v3.0": "Cohere/Cohere-embed-multilingual-light-v3.0",

    # Voyage (BPE)
    "tokiers/voyage-3": "voyage-ai/voyage-3",
    "tokiers/voyage-3-large": "voyage-ai/voyage-3-large",
    "tokiers/voyage-3-lite": "voyage-ai/voyage-3-lite",
    "tokiers/voyage-3.5": "voyage-ai/voyage-3.5",
    "tokiers/voyage-3.5-lite": "voyage-ai/voyage-3.5-lite",
    "tokiers/voyage-code-2": "voyage-ai/voyage-code-2",
    "tokiers/voyage-code-3": "voyage-ai/voyage-code-3",
    "tokiers/voyage-finance-2": "voyage-ai/voyage-finance-2",
    "tokiers/voyage-law-2": "voyage-ai/voyage-law-2",
    "tokiers/voyage-multilingual-2": "voyage-ai/voyage-multilingual-2",
    "tokiers/voyage-multimodal-3": "voyage-ai/voyage-multimodal-3",

    # Unigram / SentencePiece
    "tokiers/t5-base": "google-t5/t5-base",
    "tokiers/xlm-roberta-base": "FacebookAI/xlm-roberta-base",
}


def load_text(path: str, max_bytes: int = 1_000_000) -> str:
    """Load test text, truncated to max_bytes for speed."""
    with open(path, "rb") as f:
        data = f.read(max_bytes)
    # Ensure valid UTF-8
    return data.decode("utf-8", errors="replace")


def test_model(tokiers_repo: str, hf_model: str | None, text: str) -> dict:
    """Test a single model, return result dict."""
    import tokie

    result = {
        "tokiers_repo": tokiers_repo,
        "hf_model": hf_model,
        "status": "unknown",
        "tokie_tokens": 0,
        "hf_tokens": 0,
        "match": False,
        "error": None,
    }

    # Load tokie
    try:
        tok = tokie.Tokenizer.from_pretrained(tokiers_repo)
    except Exception as e:
        result["status"] = "tokie_load_fail"
        result["error"] = str(e)
        return result

    # Encode with tokie
    try:
        t0 = time.time()
        tokie_ids = tok.encode(text, add_special_tokens=False).ids
        result["tokie_time"] = time.time() - t0
        result["tokie_tokens"] = len(tokie_ids)
    except Exception as e:
        result["status"] = "tokie_encode_fail"
        result["error"] = str(e)
        return result

    if hf_model is None:
        result["status"] = "tokie_only"
        result["match"] = True  # no HF reference to compare
        return result

    # Load HF
    try:
        from tokenizers import Tokenizer as HfTokenizer
        hf_tok = HfTokenizer.from_pretrained(hf_model)
        # Disable truncation
        hf_tok.no_truncation()
    except Exception as e:
        result["status"] = "hf_load_fail"
        result["error"] = str(e)
        return result

    # Encode with HF
    try:
        t0 = time.time()
        hf_enc = hf_tok.encode(text, add_special_tokens=False)
        result["hf_time"] = time.time() - t0
        hf_ids = hf_enc.ids
        result["hf_tokens"] = len(hf_ids)
    except Exception as e:
        result["status"] = "hf_encode_fail"
        result["error"] = str(e)
        return result

    # Compare
    if tokie_ids == hf_ids:
        result["status"] = "pass"
        result["match"] = True
    else:
        result["status"] = "fail"
        result["match"] = False
        # Find first difference
        for i in range(min(len(tokie_ids), len(hf_ids))):
            if tokie_ids[i] != hf_ids[i]:
                result["first_diff_idx"] = i
                result["first_diff_tokie"] = tokie_ids[max(0, i-2):i+3]
                result["first_diff_hf"] = hf_ids[max(0, i-2):i+3]
                break
        else:
            result["first_diff_idx"] = min(len(tokie_ids), len(hf_ids))

    return result


def main():
    enwik8_path = "benches/data/enwik8"
    if not Path(enwik8_path).exists():
        print(f"ERROR: {enwik8_path} not found", file=sys.stderr)
        sys.exit(1)

    # Use first 1MB for speed (still ~300K tokens)
    text = load_text(enwik8_path, max_bytes=1_000_000)
    print(f"Test data: {len(text)} bytes ({len(text)/1000:.0f} KB)")
    print(f"Testing {len(MODELS)} models...\n")

    results = []
    pass_count = 0
    fail_count = 0
    skip_count = 0

    for tokiers_repo, hf_model in sorted(MODELS.items()):
        short_name = tokiers_repo.replace("tokiers/", "")
        print(f"  {short_name:45s} ", end="", flush=True)

        result = test_model(tokiers_repo, hf_model, text)
        results.append(result)

        if result["status"] == "pass":
            speedup = result.get("hf_time", 0) / max(result.get("tokie_time", 0.001), 0.001)
            print(f"PASS  ({result['tokie_tokens']:>8} tokens, {speedup:.1f}x faster)")
            pass_count += 1
        elif result["status"] == "tokie_only":
            print(f"OK    ({result['tokie_tokens']:>8} tokens, no HF reference)")
            skip_count += 1
        elif result["status"] == "fail":
            diff_idx = result.get("first_diff_idx", "?")
            print(f"FAIL  (diff at token {diff_idx}, tokie={result['tokie_tokens']} vs hf={result['hf_tokens']})")
            fail_count += 1
        else:
            print(f"ERROR ({result['status']}: {result.get('error', '?')[:60]})")
            fail_count += 1

    print(f"\n{'='*70}")
    print(f"Results: {pass_count} pass, {fail_count} fail, {skip_count} skip")
    print(f"{'='*70}")

    # Save results
    output_path = "test_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {output_path}")

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
