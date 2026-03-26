//! Full enwik8 test.
//! cargo run --example check_models --release --features hf

use std::path::Path;
use tokie::Tokenizer;
use tokenizers::Tokenizer as HfTokenizer;

fn load_enwik8(max_bytes: usize) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .parent().unwrap()
        .join("benches/data/enwik8");
    let data = std::fs::read(&path).unwrap();
    let truncated = &data[..data.len().min(max_bytes)];
    String::from_utf8_lossy(truncated).into_owned()
}

fn main() {
    let text = load_enwik8(1_000_000);
    eprintln!("Testing on {:.1} MB of enwik8\n", text.len() as f64 / 1_000_000.0);

    let models: Vec<(&str, &str)> = vec![
        // New models
        ("DeepSeek-V3", "deepseek-ai/DeepSeek-V3"),
        ("Gemma-3-4B", "google/gemma-3-4b-it"),
        ("Gemma-2-2B", "google/gemma-2-2b"),
        ("bge-m3", "BAAI/bge-m3"),
        ("Snowflake Arctic v2", "Snowflake/snowflake-arctic-embed-l-v2.0"),
        ("NV-Embed-v2", "nvidia/NV-Embed-v2"),
        // Regression check
        ("XLM-RoBERTa", "tokiers/xlm-roberta-base"),
        ("T5", "tokiers/t5-base"),
        ("BERT", "tokiers/bert-base-uncased"),
        ("GPT-2", "tokiers/gpt2"),
        ("Llama-3.2", "tokiers/Llama-3.2-1B"),
        ("Phi-2", "tokiers/phi-2"),
        ("ModernBERT", "tokiers/ModernBERT-base"),
        ("Voyage-code-2", "tokiers/voyage-code-2"),
        ("Cohere-multi-v3", "tokiers/Cohere-embed-multilingual-v3.0"),
        ("Jina-v3", "tokiers/jina-embeddings-v3"),
        ("deepset-mxbai", "tokiers/deepset-mxbai-embed-de-large-v1"),
    ];

    let mut pass = 0;
    let mut fail = 0;

    for (name, repo) in &models {
        eprint!("{:<25} ", name);
        let tok = match Tokenizer::from_pretrained(repo) {
            Ok(t) => t,
            Err(e) => { eprintln!("⚠️  {}", e); fail += 1; continue; }
        };
        let mut hf = match HfTokenizer::from_pretrained(repo, None) {
            Ok(t) => t,
            Err(e) => { eprintln!("⚠️  {}", e); fail += 1; continue; }
        };
        let _ = hf.with_truncation(None);
        let _ = hf.with_padding(None);
        let tokie_ids = tok.encode(&text, false).ids;
        let hf_ids = hf.encode(text.as_str(), false).unwrap().get_ids().to_vec();
        if tokie_ids == hf_ids {
            eprintln!("✅ PASS ({} tokens)", tokie_ids.len());
            pass += 1;
        } else {
            let d = tokie_ids.iter().zip(hf_ids.iter()).position(|(t,h)| t!=h).unwrap_or(0);
            eprintln!("❌ FAIL (tokie={}, hf={}, diff@{})", tokie_ids.len(), hf_ids.len(), d);
            fail += 1;
        }
    }
    eprintln!("\n{pass} pass, {fail} fail out of {}", pass + fail);
}
