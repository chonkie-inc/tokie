//! Test loading various tokenizers from HuggingFace JSON

use std::path::Path;
use tokie::Tokenizer;

fn test_tokenizer(name: &str, path: &str, verbose: bool) {
    let path = Path::new(path);
    if !path.exists() {
        println!("{:<25} SKIPPED (file not found)", name);
        return;
    }

    // Use from_file for .tkz, from_json for .json
    let result = if path.extension().map_or(false, |e| e == "tkz") {
        Tokenizer::from_file(path).map_err(|e| tokie::hf::JsonLoadError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))
    } else {
        Tokenizer::from_json(path)
    };

    match result {
        Ok(t) => {
            let test = "Hello, world! This is a test.";
            let tokens = t.encode(test, false);
            let decoded = t.decode(&tokens).unwrap_or_else(|| "[decode failed]".to_string());
            let roundtrip = if decoded == test { "✓" } else { "≈" };
            println!("{:<25} OK  vocab={:<6} tokens={:<3} pretok={:?} encoder={:?} {}",
                     name, t.vocab_size(), tokens.len(), t.pretokenizer_type(), t.encoder_type(), roundtrip);
            if verbose && roundtrip == "≈" {
                println!("                          in:  {:?}", test);
                println!("                          out: {:?}", decoded);
            }
        }
        Err(e) => {
            println!("{:<25} ERR {:?}", name, e);
        }
    }
}

fn main() {
    let home = std::env::var("HOME").unwrap();
    let cache = format!("{home}/.cache/huggingface/hub");
    let verbose = std::env::args().any(|a| a == "-v" || a == "--verbose");

    let models = [
        // WordPiece models
        ("GTE Base", "models--thenlper--gte-base/snapshots/c078288308d8dee004ab72c6191778064285ec0c/tokenizer.json"),
        ("MiniLM", "models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/tokenizer.json"),
        ("BGE", "models--BAAI--bge-base-en-v1.5/snapshots/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a/tokenizer.json"),
        ("E5", "models--intfloat--e5-base-v2/snapshots/f52bf8ec8c7124536f0efb74aca902b2995e5bcd/tokenizer.json"),
        ("Nomic", "models--nomic-ai--nomic-embed-text-v1/snapshots/eb6b20cd65fcbdf7a2bc4ebac97908b3b21da981/tokenizer.json"),
        // BPE models
        ("ModernBERT", "models--answerdotai--ModernBERT-base/snapshots/8949b909ec900327062f0ebf497f51aef5e6f0c8/tokenizer.json"),
        ("RoBERTa", "models--FacebookAI--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b/tokenizer.json"),
        // Local .tkz models
        ("BERT (local)", "models/bert.tkz"),
        ("GPT2 (local)", "models/gpt2.tkz"),
    ];

    println!("Testing tokenizer loading:\n");

    for (name, rel_path) in models {
        let path = if rel_path.starts_with("models/") {
            rel_path.to_string()
        } else {
            format!("{cache}/{rel_path}")
        };
        test_tokenizer(name, &path, verbose);
    }
}
