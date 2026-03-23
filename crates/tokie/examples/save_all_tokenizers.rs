//! Save all tokenizers to binary format with updated serde (v7)

use tokie::{EncoderType, PretokType, Tokenizer};
use tokie::hf::from_json_with_options;

fn main() {
    let home = dirs::home_dir().unwrap();
    let hf_cache = home.join(".cache/huggingface/hub");

    // GPT-2
    println!("=== GPT-2 ===");
    let gpt2_paths = [
        hf_cache.join("models--openai-community--gpt2/snapshots"),
        hf_cache.join("models--gpt2/snapshots"),
    ];

    for gpt2_path in &gpt2_paths {
        if let Some(json_path) = find_tokenizer_json(gpt2_path) {
            let tok = from_json_with_options(&json_path, EncoderType::Backtracking, PretokType::Gpt2).unwrap();
            println!("  Encoder: {:?}, Pretok: {:?}, Vocab: {}", tok.encoder_type(), tok.pretokenizer_type(), tok.vocab_size());
            tok.to_file("models/gpt2.tkz").unwrap();
            println!("  Saved to models/gpt2.tkz");
            verify("models/gpt2.tkz");
            break;
        }
    }

    // cl100k (GPT-4)
    println!("\n=== cl100k (GPT-4) ===");
    let cl100k_path = hf_cache.join("models--Xenova--gpt-4/snapshots");
    if let Some(json_path) = find_tokenizer_json(&cl100k_path) {
        let tok = from_json_with_options(&json_path, EncoderType::Backtracking, PretokType::Cl100k).unwrap();
        println!("  Encoder: {:?}, Pretok: {:?}, Vocab: {}", tok.encoder_type(), tok.pretokenizer_type(), tok.vocab_size());
        tok.to_file("models/cl100k.tkz").unwrap();
        println!("  Saved to models/cl100k.tkz");
        verify("models/cl100k.tkz");
    } else {
        println!("  Skipping - tokenizer not found in cache");
    }

    // o200k (GPT-4o)
    println!("\n=== o200k (GPT-4o) ===");
    let o200k_path = hf_cache.join("models--Xenova--gpt-4o/snapshots");
    if let Some(json_path) = find_tokenizer_json(&o200k_path) {
        let tok = from_json_with_options(&json_path, EncoderType::Backtracking, PretokType::O200k).unwrap();
        println!("  Encoder: {:?}, Pretok: {:?}, Vocab: {}", tok.encoder_type(), tok.pretokenizer_type(), tok.vocab_size());
        tok.to_file("models/o200k.tkz").unwrap();
        println!("  Saved to models/o200k.tkz");
        verify("models/o200k.tkz");
    } else {
        println!("  Skipping - tokenizer not found in cache");
    }

    println!("\n=== Done ===");
    println!("\nSaved models:");
    for entry in std::fs::read_dir("models").unwrap() {
        let entry = entry.unwrap();
        let metadata = entry.metadata().unwrap();
        let size = metadata.len();
        println!("  {} ({:.2} MB)", entry.file_name().to_string_lossy(), size as f64 / 1_000_000.0);
    }
}

fn find_tokenizer_json(snapshots_path: &std::path::Path) -> Option<std::path::PathBuf> {
    if !snapshots_path.exists() {
        return None;
    }

    std::fs::read_dir(snapshots_path).ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path().join("tokenizer.json"))
        .find(|p| p.exists())
}

fn verify(path: &str) {
    let loaded = Tokenizer::from_file(path).unwrap();
    let text = "Hello, world! Testing tokenizer.";
    let tokens = loaded.encode(text, false);
    let decoded = loaded.decode(&tokens).unwrap();
    if decoded == text {
        println!("  Verified: encode/decode roundtrip OK ({} tokens)", tokens.len());
    } else {
        println!("  WARNING: decode mismatch!");
    }
}
