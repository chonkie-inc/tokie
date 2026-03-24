#!/usr/bin/env python3
"""Populate newly created tokiers repos with original model files + banner + README."""

from huggingface_hub import HfApi, hf_hub_download
import os
import tempfile

api = HfApi()

# Map: tokiers repo name -> original HF repo
NEW_REPOS = {
    # Sentence Transformers / Embedding models
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "all-MiniLM-L12-v2": "sentence-transformers/all-MiniLM-L12-v2",
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    "e5-small-v2": "intfloat/e5-small-v2",
    "e5-base-v2": "intfloat/e5-base-v2",
    "e5-large-v2": "intfloat/e5-large-v2",
    "gte-small": "thenlper/gte-small",
    "gte-base": "thenlper/gte-base",
    "gte-large": "thenlper/gte-large",
    "nomic-embed-text-v1": "nomic-ai/nomic-embed-text-v1",
    "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
    "bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
    "bge-large-en-v1.5": "BAAI/bge-large-en-v1.5",
    "jina-embeddings-v2-base-en": "jinaai/jina-embeddings-v2-base-en",
    "jina-embeddings-v2-base-code": "jinaai/jina-embeddings-v2-base-code",
    "voyage-3-large": "voyageai/voyage-3-large",
    # BERT family
    "bert-base-uncased": "bert-base-uncased",
    "roberta-base": "FacebookAI/roberta-base",
    "ModernBERT-base": "answerdotai/ModernBERT-base",
    # GPT-2 / tiktoken
    "gpt2": "openai-community/gpt2",
    "cl100k": "Xenova/gpt-4",
    "o200k": "Xenova/gpt-4o",
    # LLaMA
    "Llama-3.2-1B": "meta-llama/Llama-3.2-1B",
    "Llama-4-Scout-17B-16E": "meta-llama/Llama-4-Scout-17B-16E",
    "CodeLlama-7b-hf": "codellama/CodeLlama-7b-hf",
    # Mistral
    "Mistral-7B-v0.1": "mistralai/Mistral-7B-v0.1",
    "Mistral-Nemo-Base-2407": "mistralai/Mistral-Nemo-Base-2407",
    "Mixtral-8x7B-v0.1": "mistralai/Mixtral-8x7B-v0.1",
    # Phi
    "phi-2": "microsoft/phi-2",
    "Phi-3-mini-4k-instruct": "microsoft/Phi-3-mini-4k-instruct",
    # Qwen
    "Qwen2-7B": "Qwen/Qwen2-7B",
    # T5 / XLM-R
    "t5-base": "google-t5/t5-base",
    "xlm-roberta-base": "FacebookAI/xlm-roberta-base",
}

# Files to try downloading from the original repo
TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]

BANNER_PATH = "assets/tokie-banner.png"
ORG = "tokiers"


def make_readme(repo_name, original_repo):
    return f"""---
tags:
- tokie
library_name: tokie
---

<p align="center">
  <img src="tokie-banner.png" alt="tokie" width="600">
</p>

# {repo_name}

Pre-built [tokie](https://github.com/chonkie-inc/tokie) tokenizer for [{original_repo}](https://huggingface.co/{original_repo}).

## Quick Start (Python)

```bash
pip install tokie
```

```python
import tokie

tokenizer = tokie.Tokenizer.from_pretrained("tokiers/{repo_name}")
encoding = tokenizer.encode("Hello, world!")
print(encoding.ids)
print(encoding.attention_mask)
```

## Quick Start (Rust)

```toml
[dependencies]
tokie = {{ version = "0.0.4", features = ["hf"] }}
```

```rust
use tokie::Tokenizer;

let tokenizer = Tokenizer::from_pretrained("tokiers/{repo_name}").unwrap();
let encoding = tokenizer.encode("Hello, world!", true);
println!("{{:?}}", encoding.ids);
```

## Files

- `tokenizer.tkz` — tokie binary format (~10x smaller, loads in ~5ms)
- `tokenizer.json` — original HuggingFace tokenizer

## About tokie

**50x faster tokenization, 10x smaller model files, 100% accurate.**

tokie is a drop-in replacement for HuggingFace tokenizers, built in Rust. See [GitHub](https://github.com/chonkie-inc/tokie) for benchmarks and documentation.

## License

MIT OR Apache-2.0 (tokie library). Original model files retain their original license from [{original_repo}](https://huggingface.co/{original_repo}).
"""


success = 0
failed = []

for repo_name, original_repo in sorted(NEW_REPOS.items()):
    repo_id = f"{ORG}/{repo_name}"
    print(f"\n{'='*60}")
    print(f"  {repo_id}  (from {original_repo})")
    print(f"{'='*60}")

    try:
        # 1. Upload banner
        print(f"  Uploading banner...", end=" ", flush=True)
        api.upload_file(
            path_or_fileobj=BANNER_PATH,
            path_in_repo="tokie-banner.png",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add tokie banner",
        )
        print("OK")

        # 2. Download and upload tokenizer files from original repo
        for fname in TOKENIZER_FILES:
            print(f"  Fetching {fname}...", end=" ", flush=True)
            try:
                local_path = hf_hub_download(
                    repo_id=original_repo,
                    filename=fname,
                )
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=fname,
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"Add {fname} from {original_repo}",
                )
                print("OK")
            except Exception as e:
                if "404" in str(e) or "EntryNotFound" in str(e) or "does not exist" in str(e).lower():
                    print("not found (skipped)")
                else:
                    print(f"WARN: {e}")

        # 3. Upload README
        print(f"  Uploading README...", end=" ", flush=True)
        readme_content = make_readme(repo_name, original_repo)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(readme_content)
            f.flush()
            api.upload_file(
                path_or_fileobj=f.name,
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="model",
                commit_message="Add model card README",
            )
        os.unlink(f.name)
        print("OK")

        success += 1

    except Exception as e:
        print(f"  FAIL: {e}")
        failed.append((repo_name, str(e)))

print(f"\n{'='*60}")
print(f"=== Summary ===")
print(f"  Success: {success}/{len(NEW_REPOS)}")
print(f"  Failed:  {len(failed)}/{len(NEW_REPOS)}")
if failed:
    print("\n  Failures:")
    for name, err in failed:
        print(f"    {name} -- {err}")
