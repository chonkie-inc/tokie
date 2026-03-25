//! Token-accuracy tests: compare tokie against HuggingFace tokenizers on enwik8.
//!
//! Run with: cargo test -p tokie --test accuracy --features hf -- --ignored
//!
//! Requires network access and benches/data/enwik8 (1MB used).

use std::path::Path;

use tokenizers::Tokenizer as HfTokenizer;
use tokie::Tokenizer;

/// Load first `max_bytes` of enwik8, returning valid UTF-8.
fn load_enwik8(max_bytes: usize) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("benches/data/enwik8");
    let data = std::fs::read(&path).unwrap_or_else(|e| panic!("Failed to read {}: {e}", path.display()));
    let truncated = &data[..data.len().min(max_bytes)];
    String::from_utf8_lossy(truncated).into_owned()
}

/// Compare tokie (from tokiers/) against HuggingFace on enwik8.
/// Returns (pass, first_diff_index) — None means all tokens match.
fn compare_model(tokiers_repo: &str, hf_model: &str, text: &str) -> (bool, Option<usize>) {
    let tok = Tokenizer::from_pretrained(tokiers_repo)
        .unwrap_or_else(|e| panic!("Failed to load tokie {tokiers_repo}: {e}"));
    let hf = HfTokenizer::from_pretrained(hf_model, None)
        .unwrap_or_else(|e| panic!("Failed to load HF {hf_model}: {e}"));

    let tokie_ids = tok.encode(text, false).ids;
    let hf_enc = hf.encode(text, false)
        .unwrap_or_else(|e| panic!("HF encode failed for {hf_model}: {e}"));
    let hf_ids = hf_enc.get_ids();

    if tokie_ids.as_slice() == hf_ids {
        (true, None)
    } else {
        let diff = tokie_ids.iter().zip(hf_ids.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(tokie_ids.len().min(hf_ids.len()));
        (false, Some(diff))
    }
}

// ============================================================================
// WordPiece models (BERT-family)
// ============================================================================

macro_rules! accuracy_test {
    ($name:ident, $tokiers:expr, $hf:expr) => {
        #[test]
        #[ignore] // Requires network + enwik8
        fn $name() {
            let text = load_enwik8(1_000_000);
            let (pass, diff) = compare_model($tokiers, $hf, &text);
            assert!(pass, "Token mismatch at index {:?}", diff);
        }
    };
}

// WordPiece
accuracy_test!(bert_base_uncased,       "tokiers/bert-base-uncased",       "google-bert/bert-base-uncased");
accuracy_test!(all_minilm_l6_v2,        "tokiers/all-MiniLM-L6-v2",       "sentence-transformers/all-MiniLM-L6-v2");
accuracy_test!(all_minilm_l12_v2,       "tokiers/all-MiniLM-L12-v2",      "sentence-transformers/all-MiniLM-L12-v2");
accuracy_test!(all_mpnet_base_v2,       "tokiers/all-mpnet-base-v2",      "sentence-transformers/all-mpnet-base-v2");
accuracy_test!(bge_base_en_v1_5,        "tokiers/bge-base-en-v1.5",       "BAAI/bge-base-en-v1.5");
accuracy_test!(bge_large_en_v1_5,       "tokiers/bge-large-en-v1.5",      "BAAI/bge-large-en-v1.5");
accuracy_test!(bge_small_en_v1_5,       "tokiers/bge-small-en-v1.5",      "BAAI/bge-small-en-v1.5");
accuracy_test!(bge_en_icl,              "tokiers/bge-en-icl",             "BAAI/bge-en-icl");
accuracy_test!(e5_base_v2,              "tokiers/e5-base-v2",             "intfloat/e5-base-v2");
accuracy_test!(e5_large_v2,             "tokiers/e5-large-v2",            "intfloat/e5-large-v2");
accuracy_test!(e5_small_v2,             "tokiers/e5-small-v2",            "intfloat/e5-small-v2");
accuracy_test!(gte_base,                "tokiers/gte-base",               "thenlper/gte-base");
accuracy_test!(gte_large,               "tokiers/gte-large",              "thenlper/gte-large");
accuracy_test!(gte_small,               "tokiers/gte-small",              "thenlper/gte-small");
accuracy_test!(gte_qwen2_7b_instruct,   "tokiers/gte-Qwen2-7B-instruct", "Alibaba-NLP/gte-Qwen2-7B-instruct");
accuracy_test!(ms_marco_minilm_l_4_v2,  "tokiers/ms-marco-MiniLM-L-4-v2","cross-encoder/ms-marco-MiniLM-L-4-v2");
accuracy_test!(ms_marco_minilm_l_6_v2,  "tokiers/ms-marco-MiniLM-L-6-v2","cross-encoder/ms-marco-MiniLM-L-6-v2");
accuracy_test!(mxbai_embed_large_v1,    "tokiers/mxbai-embed-large-v1",   "mixedbread-ai/mxbai-embed-large-v1");
accuracy_test!(mxbai_embed_2d_large_v1, "tokiers/mxbai-embed-2d-large-v1","mixedbread-ai/mxbai-embed-2d-large-v1");
accuracy_test!(mxbai_embed_xsmall_v1,   "tokiers/mxbai-embed-xsmall-v1",  "mixedbread-ai/mxbai-embed-xsmall-v1");
accuracy_test!(deepset_mxbai_embed_de,  "tokiers/deepset-mxbai-embed-de-large-v1", "mixedbread-ai/deepset-mxbai-embed-de-large-v1");
accuracy_test!(nomic_embed_text_v1,     "tokiers/nomic-embed-text-v1",    "nomic-ai/nomic-embed-text-v1");

// BPE (byte-level)
accuracy_test!(gpt2,                    "tokiers/gpt2",                   "openai-community/gpt2");
accuracy_test!(roberta_base,            "tokiers/roberta-base",           "FacebookAI/roberta-base");
accuracy_test!(phi_2,                   "tokiers/phi-2",                  "microsoft/phi-2");
accuracy_test!(phi_3_mini,              "tokiers/Phi-3-mini-4k-instruct", "microsoft/Phi-3-mini-4k-instruct");
accuracy_test!(modernbert_base,         "tokiers/ModernBERT-base",        "answerdotai/ModernBERT-base");
accuracy_test!(codellama_7b,            "tokiers/CodeLlama-7b-hf",       "codellama/CodeLlama-7b-hf");
accuracy_test!(llama_3_2_1b,            "tokiers/Llama-3.2-1B",          "meta-llama/Llama-3.2-1B");
accuracy_test!(llama_4_scout,           "tokiers/Llama-4-Scout-17B-16E", "meta-llama/Llama-4-Scout-17B-16E");
accuracy_test!(mistral_7b,              "tokiers/Mistral-7B-v0.1",       "mistralai/Mistral-7B-v0.1");
accuracy_test!(mistral_nemo,            "tokiers/Mistral-Nemo-Base-2407","mistralai/Mistral-Nemo-Base-2407");
accuracy_test!(mixtral_8x7b,            "tokiers/Mixtral-8x7B-v0.1",    "mistralai/Mixtral-8x7B-v0.1");
accuracy_test!(qwen2_7b,               "tokiers/Qwen2-7B",              "Qwen/Qwen2-7B");
accuracy_test!(qwen3_embed_0_6b,        "tokiers/Qwen3-Embedding-0.6B",  "Qwen/Qwen3-Embedding-0.6B");
accuracy_test!(qwen3_embed_4b,          "tokiers/Qwen3-Embedding-4B",    "Qwen/Qwen3-Embedding-4B");
accuracy_test!(qwen3_embed_8b,          "tokiers/Qwen3-Embedding-8B",    "Qwen/Qwen3-Embedding-8B");

// Jina (BPE)
accuracy_test!(jina_v2_base_en,         "tokiers/jina-embeddings-v2-base-en",   "jinaai/jina-embeddings-v2-base-en");
accuracy_test!(jina_v2_base_code,       "tokiers/jina-embeddings-v2-base-code", "jinaai/jina-embeddings-v2-base-code");
accuracy_test!(jina_v3,                 "tokiers/jina-embeddings-v3",           "jinaai/jina-embeddings-v3");
accuracy_test!(jina_v4,                 "tokiers/jina-embeddings-v4",           "jinaai/jina-embeddings-v4");

// Cohere (BPE)
accuracy_test!(cohere_english_v3,       "tokiers/Cohere-embed-english-v3.0",            "Cohere/Cohere-embed-english-v3.0");
accuracy_test!(cohere_english_light_v3, "tokiers/Cohere-embed-english-light-v3.0",      "Cohere/Cohere-embed-english-light-v3.0");
accuracy_test!(cohere_multi_v3,         "tokiers/Cohere-embed-multilingual-v3.0",       "Cohere/Cohere-embed-multilingual-v3.0");
accuracy_test!(cohere_multi_light_v3,   "tokiers/Cohere-embed-multilingual-light-v3.0", "Cohere/Cohere-embed-multilingual-light-v3.0");

// SentencePiece / Unigram
accuracy_test!(t5_base,                 "tokiers/t5-base",              "google-t5/t5-base");
accuracy_test!(xlm_roberta_base,        "tokiers/xlm-roberta-base",     "FacebookAI/xlm-roberta-base");
