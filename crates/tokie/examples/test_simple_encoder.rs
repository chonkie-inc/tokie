//! Test Simple BPE encoder for SentencePiece
//!
//! Run with: cargo run --release --example test_simple_encoder

use std::path::PathBuf;

fn get_hf_tokenizer_path(model_id: &str) -> Option<PathBuf> {
    let cache_dir = dirs::home_dir()?.join(".cache/huggingface/hub");
    let model_dir_name = format!("models--{}", model_id.replace('/', "--"));
    let model_dir = cache_dir.join(&model_dir_name).join("snapshots");
    let snapshot = model_dir.read_dir().ok()?.next()?.ok()?.path();
    let tokenizer_path = snapshot.join("tokenizer.json");
    if tokenizer_path.exists() {
        Some(tokenizer_path)
    } else {
        None
    }
}

fn main() {
    let tokenizer_path = get_hf_tokenizer_path("mistralai/Mistral-7B-v0.1").unwrap();

    // Load with HF tokenizers lib for reference
    let hf_tok = tokenizers::Tokenizer::from_file(&tokenizer_path).unwrap();

    // Load with tokie using Simple encoder
    let tokie = tokie::Tokenizer::from_json_with_encoder(
        &tokenizer_path,
        tokie::EncoderType::Simple
    ).unwrap();

    println!("Encoder: {:?}", tokie.encoder_type());
    println!("Normalizer: {:?}", tokie.normalizer());

    let test_cases = [
        "Hello, world!",
        "  multiple   spaces  ",
        "The quick brown fox",
    ];

    for text in test_cases {
        println!("\n=== Test: {:?} ===", text);

        let hf_enc = hf_tok.encode(text, false).unwrap();
        let tokie_tokens = tokie.encode(text, false);

        println!("HF tokens:    {:?}", hf_enc.get_ids());
        println!("tokie tokens: {:?}", tokie_tokens);

        if hf_enc.get_ids() == &tokie_tokens {
            println!("✓ MATCH");
        } else {
            println!("✗ MISMATCH");
            println!("  HF decoded:    {:?}", hf_tok.decode(hf_enc.get_ids(), false).unwrap());
            println!("  tokie decoded: {:?}", tokie.decode(&tokie_tokens).unwrap_or_default());
        }
    }
}
