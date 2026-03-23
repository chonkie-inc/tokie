//! Benchmark WordPiece (BERT) encoding: tokie vs HuggingFace tokenizers
//!
//! Run with: cargo run --example bench_wordpiece --release

use std::time::Instant;

const ITERATIONS: usize = 10;

fn main() {
    let text = std::fs::read_to_string("benches/data/war_and_peace.txt")
        .expect("Failed to read war_and_peace.txt");
    let mb = text.len() as f64 / 1_000_000.0;
    println!("Text size: {:.2} MB\n", mb);

    // Load tokie from .tkz
    let tokie_tok = tokie::Tokenizer::from_file("../../models/bert.tkz")
        .expect("Failed to load models/bert.tkz");

    // Load HF tokenizers from JSON
    let hf_tok = tokenizers::Tokenizer::from_file("benches/data/gpt2_tokenizer.json")
        .ok();

    // Try loading BERT JSON from /tmp/tokenizers/
    let hf_bert = tokenizers::Tokenizer::from_file("/tmp/tokenizers/bert_tokenizer.json")
        .expect("Failed to load /tmp/tokenizers/bert_tokenizer.json - download with:\n  curl -sL 'https://huggingface.co/bert-base-uncased/raw/main/tokenizer.json' -o /tmp/tokenizers/bert_tokenizer.json");

    // Warmup
    let _ = tokie_tok.encode(&text[..1000], false);
    let _ = hf_bert.encode(text.as_str(), false);

    // Benchmark tokie
    println!("=== tokie (WordPiece) ===");
    let start = Instant::now();
    let mut tokie_count = 0;
    for _ in 0..ITERATIONS {
        let tokens = tokie_tok.encode(&text, false);
        tokie_count = tokens.len();
    }
    let tokie_elapsed = start.elapsed() / ITERATIONS as u32;
    let tokie_tp = mb / tokie_elapsed.as_secs_f64();
    println!("  Tokens: {}", tokie_count);
    println!("  Encode time: {:.2} ms/iter", tokie_elapsed.as_secs_f64() * 1000.0);
    println!("  Throughput: {:.1} MB/s", tokie_tp);

    // Benchmark HF tokenizers
    println!("\n=== HF tokenizers (WordPiece) ===");
    let start = Instant::now();
    let mut hf_count = 0;
    for _ in 0..ITERATIONS {
        let encoding = hf_bert.encode(text.as_str(), false).unwrap();
        hf_count = encoding.get_ids().len();
    }
    let hf_elapsed = start.elapsed() / ITERATIONS as u32;
    let hf_tp = mb / hf_elapsed.as_secs_f64();
    println!("  Tokens: {}", hf_count);
    println!("  Encode time: {:.2} ms/iter", hf_elapsed.as_secs_f64() * 1000.0);
    println!("  Throughput: {:.1} MB/s", hf_tp);

    println!("\nSpeedup: {:.1}x", tokie_tp / hf_tp);
    println!("Token match: {}", if tokie_count == hf_count { "YES" } else { "NO" });
}
