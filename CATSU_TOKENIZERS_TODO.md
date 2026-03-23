# Catsu Tokenizers - TODO

Tokenizers used in catsu's models.json that are not yet supported in tokie.

## Priority: High (Multiple models use these)

### Voyage AI (10 tokenizers)
- [ ] `voyageai/voyage-3`
- [ ] `voyageai/voyage-3-lite`
- [ ] `voyageai/voyage-3.5`
- [ ] `voyageai/voyage-3.5-lite`
- [ ] `voyageai/voyage-code-2`
- [ ] `voyageai/voyage-code-3`
- [ ] `voyageai/voyage-finance-2`
- [ ] `voyageai/voyage-law-2`
- [ ] `voyageai/voyage-multilingual-2`
- [ ] `voyageai/voyage-multimodal-3`

### Cohere (5 tokenizers)
- [ ] `Cohere/Cohere-embed-v4`
- [ ] `Cohere/Cohere-embed-english-v3.0`
- [ ] `Cohere/Cohere-embed-english-light-v3.0`
- [ ] `Cohere/Cohere-embed-multilingual-v3.0`
- [ ] `Cohere/Cohere-embed-multilingual-light-v3.0`

### Jina AI v3/v4 (4 tokenizers)
- [ ] `jinaai/jina-embeddings-v3`
- [ ] `jinaai/jina-embeddings-v4`
- [ ] `jinaai/jina-code-embeddings-0.5b`
- [ ] `jinaai/jina-code-embeddings-1.5b`

### Mixedbread (4 tokenizers)
- [ ] `mixedbread-ai/mxbai-embed-large-v1`
- [ ] `mixedbread-ai/mxbai-embed-2d-large-v1`
- [ ] `mixedbread-ai/mxbai-embed-xsmall-v1`
- [ ] `mixedbread-ai/deepset-mxbai-embed-de-large-v1`

## Priority: Medium

### Qwen3 Embeddings (3 tokenizers)
- [ ] `Qwen/Qwen3-Embedding-0.6B`
- [ ] `Qwen/Qwen3-Embedding-4B`
- [ ] `Qwen/Qwen3-Embedding-8B`

### Alibaba NLP (1 tokenizer)
- [ ] `Alibaba-NLP/gte-Qwen2-7B-instruct`

### BAAI (1 tokenizer)
- [ ] `BAAI/bge-en-icl`

## Priority: Lower

### Google (1 tokenizer)
- [ ] `google/embeddinggemma-300m`

### PFNet (1 tokenizer)
- [ ] `pfnet/plamo-1.0-embedding`

---

## Already Supported (21 tokenizers)

### Direct Support (15)
- [x] `BAAI/bge-base-en-v1.5` -> baai_bge_base_en_v1.5.tkz
- [x] `BAAI/bge-large-en-v1.5` -> baai_bge_large_en_v1.5.tkz
- [x] `BAAI/bge-small-en-v1.5` -> baai_bge_small_en_v1.5.tkz
- [x] `intfloat/e5-base-v2` -> intfloat_e5_base_v2.tkz
- [x] `intfloat/e5-large-v2` -> intfloat_e5_large_v2.tkz
- [x] `jinaai/jina-embeddings-v2-base-code` -> jinaai_jina_embeddings_v2_base_code.tkz
- [x] `jinaai/jina-embeddings-v2-base-en` -> jinaai_jina_embeddings_v2_base_en.tkz
- [x] `nomic-ai/nomic-embed-text-v1` -> nomic_ai_nomic_embed_text_v1.tkz
- [x] `sentence-transformers/all-MiniLM-L12-v2` -> sentence_transformers_all_minilm_l12_v2.tkz
- [x] `sentence-transformers/all-MiniLM-L6-v2` -> sentence_transformers_all_minilm_l6_v2.tkz
- [x] `sentence-transformers/all-mpnet-base-v2` -> sentence_transformers_all_mpnet_base_v2.tkz
- [x] `thenlper/gte-base` -> thenlper_gte_base.tkz
- [x] `thenlper/gte-large` -> thenlper_gte_large.tkz
- [x] `voyageai/voyage-3-large` -> voyage3_large.tkz
- [x] `cl100k_base` (tiktoken) -> cl100k.tkz

### Compatible (6)
- [x] `nomic-ai/nomic-embed-text-v1.5` ~ nomic_ai_nomic_embed_text_v1.tkz
- [x] `Alibaba-NLP/gte-modernbert-base` ~ modernbert.tkz
- [x] `BAAI/bge-m3` ~ xlm_roberta.tkz
- [x] `intfloat/multilingual-e5-large` ~ xlm_roberta.tkz
- [x] `WhereIsAI/UAE-Large-V1` ~ bert.tkz
- [x] `togethercomputer/m2-bert-80M-8k-retrieval` ~ bert.tkz

---

## Progress

- **Supported:** 21/51 (41.2%)
- **Remaining:** 30/51 (58.8%)
