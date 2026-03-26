#!/usr/bin/env python3
"""Upload all regenerated v11 .tkz files to the tokiers HuggingFace org."""

from huggingface_hub import HfApi, create_repo
import os

api = HfApi()

# Map: .tkz filename (without extension) -> tokiers repo name
# Only models that should be on the tokiers org (embedding/reranker models, not base LLMs)
TKZ_TO_REPO = {
    # Cross-encoders
    "cross_encoder_ms_marco_minilm_l4_v2": "ms-marco-MiniLM-L-4-v2",
    "cross_encoder_ms_marco_minilm_l6_v2": "ms-marco-MiniLM-L-6-v2",
    # Voyage AI
    "voyageai_voyage_3": "voyage-3",
    "voyageai_voyage_3_lite": "voyage-3-lite",
    "voyageai_voyage_3.5": "voyage-3.5",
    "voyageai_voyage_3.5_lite": "voyage-3.5-lite",
    "voyageai_voyage_code_2": "voyage-code-2",
    "voyageai_voyage_code_3": "voyage-code-3",
    "voyageai_voyage_finance_2": "voyage-finance-2",
    "voyageai_voyage_law_2": "voyage-law-2",
    "voyageai_voyage_multilingual_2": "voyage-multilingual-2",
    "voyageai_voyage_multimodal_3": "voyage-multimodal-3",
    "voyage3_large": "voyage-3-large",
    # Cohere
    "cohere_embed_english_v3.0": "Cohere-embed-english-v3.0",
    "cohere_embed_english_light_v3.0": "Cohere-embed-english-light-v3.0",
    "cohere_embed_multilingual_v3.0": "Cohere-embed-multilingual-v3.0",
    "cohere_embed_multilingual_light_v3.0": "Cohere-embed-multilingual-light-v3.0",
    # Jina AI
    "jinaai_jina_embeddings_v2_base_en": "jina-embeddings-v2-base-en",
    "jinaai_jina_embeddings_v2_base_code": "jina-embeddings-v2-base-code",
    "jinaai_jina_embeddings_v3": "jina-embeddings-v3",
    "jinaai_jina_embeddings_v4": "jina-embeddings-v4",
    # Mixedbread
    "mixedbread_ai_mxbai_embed_large_v1": "mxbai-embed-large-v1",
    "mixedbread_ai_mxbai_embed_2d_large_v1": "mxbai-embed-2d-large-v1",
    "mixedbread_ai_mxbai_embed_xsmall_v1": "mxbai-embed-xsmall-v1",
    "mixedbread_ai_deepset_mxbai_embed_de_large_v1": "deepset-mxbai-embed-de-large-v1",
    # Qwen embeddings
    "qwen3_embedding_0.6b": "Qwen3-Embedding-0.6B",
    "qwen3_embedding_4b": "Qwen3-Embedding-4B",
    "qwen3_embedding_8b": "Qwen3-Embedding-8B",
    "alibaba_nlp_gte_qwen2_7b_instruct": "gte-Qwen2-7B-instruct",
    # BAAI
    "baai_bge_small_en_v1.5": "bge-small-en-v1.5",
    "baai_bge_base_en_v1.5": "bge-base-en-v1.5",
    "baai_bge_large_en_v1.5": "bge-large-en-v1.5",
    "baai_bge_en_icl": "bge-en-icl",
    # Sentence Transformers
    "sentence_transformers_all_minilm_l6_v2": "all-MiniLM-L6-v2",
    "sentence_transformers_all_minilm_l12_v2": "all-MiniLM-L12-v2",
    "sentence_transformers_all_mpnet_base_v2": "all-mpnet-base-v2",
    # E5
    "intfloat_e5_small_v2": "e5-small-v2",
    "intfloat_e5_base_v2": "e5-base-v2",
    "intfloat_e5_large_v2": "e5-large-v2",
    # GTE
    "thenlper_gte_small": "gte-small",
    "thenlper_gte_base": "gte-base",
    "thenlper_gte_large": "gte-large",
    # Nomic
    "nomic_ai_nomic_embed_text_v1": "nomic-embed-text-v1",
    # BERT/RoBERTa/ModernBERT
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
    "modernbert": "ModernBERT-base",
    # Base models (useful for downstream)
    "gpt2": "gpt2",
    "cl100k": "cl100k",
    "o200k": "o200k",
    "llama3": "Llama-3.2-1B",
    "llama4": "Llama-4-Scout-17B-16E",
    "codellama": "CodeLlama-7b-hf",
    "mistral_7b": "Mistral-7B-v0.1",
    "mistral_nemo": "Mistral-Nemo-Base-2407",
    "mixtral_8x7b": "Mixtral-8x7B-v0.1",
    "phi2": "phi-2",
    "phi3": "Phi-3-mini-4k-instruct",
    "qwen2": "Qwen2-7B",
    "t5": "t5-base",
    "xlm_roberta": "xlm-roberta-base",
    # New models
    "DeepSeek-V3": "DeepSeek-V3",
    "DeepSeek-R1": "DeepSeek-R1",
    "gemma-2-2b": "gemma-2-2b",
    "gemma-3-4b-it": "gemma-3-4b-it",
    "bge-m3": "bge-m3",
    "snowflake-arctic-embed-l-v2.0": "snowflake-arctic-embed-l-v2.0",
    "NV-Embed-v2": "NV-Embed-v2",
    "SmolLM2-135M": "SmolLM2-135M",
    "stablelm-2-1_6b": "stablelm-2-1_6b",
}

MODELS_DIR = "models"
ORG = "tokiers"

success = 0
failed = []

for tkz_name, repo_name in sorted(TKZ_TO_REPO.items()):
    tkz_path = os.path.join(MODELS_DIR, f"{tkz_name}.tkz")
    repo_id = f"{ORG}/{repo_name}"

    if not os.path.exists(tkz_path):
        print(f"  SKIP  {tkz_name:55s} -- file not found")
        continue

    print(f"  {tkz_name:55s} -> {repo_id:45s} ", end="", flush=True)

    try:
        # Create repo if it doesn't exist
        create_repo(repo_id, repo_type="model", exist_ok=True)

        # Upload the .tkz file
        api.upload_file(
            path_or_fileobj=tkz_path,
            path_in_repo="tokenizer.tkz",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Update tokenizer.tkz to v11 (with pad_token_id)",
        )
        size_kb = os.path.getsize(tkz_path) / 1024
        print(f"OK ({size_kb:.0f} KB)")
        success += 1
    except Exception as e:
        print(f"FAIL: {e}")
        failed.append((tkz_name, str(e)))

print(f"\n=== Summary ===")
print(f"  Success: {success}/{len(TKZ_TO_REPO)}")
print(f"  Failed:  {len(failed)}/{len(TKZ_TO_REPO)}")
if failed:
    print("\n  Failures:")
    for name, err in failed:
        print(f"    {name} -- {err}")
