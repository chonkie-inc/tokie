//! Compare tokie vs huggingface tokenizers vs tiktoken
//!
//! Run with: cargo run --release --example compare_libs

use std::fs;
use std::time::Instant;

const TEXT_PATH: &str = "benches/data/war_and_peace.txt";
const TOKENIZER_PATH: &str = "benches/data/gpt2_tokenizer.json";
const ITERATIONS: usize = 20;

fn main() {
    let text = fs::read_to_string(TEXT_PATH).expect("Failed to read text file");
    let bytes = text.len();

    println!("Tokenizer Comparison Benchmark");
    println!("==============================");
    println!("Text: War and Peace ({:.2} MB)\n", bytes as f64 / 1024.0 / 1024.0);

    // 1. tokie
    let tokie = tokie::Tokenizer::from_json(TOKENIZER_PATH).expect("Failed to load tokie");

    // Warmup
    let _ = tokie.encode(&text, false);

    let start = Instant::now();
    let mut tokie_tokens = 0;
    for _ in 0..ITERATIONS {
        tokie_tokens = tokie.encode(&text, false).len();
    }
    let tokie_time = start.elapsed();
    let tokie_tp = (bytes * ITERATIONS) as f64 / tokie_time.as_secs_f64() / 1024.0 / 1024.0;

    println!("tokie");
    println!("  Tokens: {}", tokie_tokens);
    println!("  Time:   {:.2} ms", tokie_time.as_secs_f64() * 1000.0 / ITERATIONS as f64);
    println!("  Throughput: {:.1} MiB/s\n", tokie_tp);

    // 2. huggingface tokenizers
    let hf = tokenizers::Tokenizer::from_file(TOKENIZER_PATH).expect("Failed to load HF tokenizer");

    // Warmup
    let _ = hf.encode(text.as_str(), false);

    let start = Instant::now();
    let mut hf_tokens = 0;
    for _ in 0..ITERATIONS {
        hf_tokens = hf.encode(text.as_str(), false).unwrap().get_ids().len();
    }
    let hf_time = start.elapsed();
    let hf_tp = (bytes * ITERATIONS) as f64 / hf_time.as_secs_f64() / 1024.0 / 1024.0;

    println!("huggingface tokenizers");
    println!("  Tokens: {}", hf_tokens);
    println!("  Time:   {:.2} ms", hf_time.as_secs_f64() * 1000.0 / ITERATIONS as f64);
    println!("  Throughput: {:.1} MiB/s\n", hf_tp);

    // 3. tiktoken
    let tiktoken = tiktoken_rs::get_bpe_from_model("gpt2").expect("Failed to load tiktoken");

    // Warmup
    let _ = tiktoken.encode_ordinary(&text);

    let start = Instant::now();
    let mut tiktoken_tokens = 0;
    for _ in 0..ITERATIONS {
        tiktoken_tokens = tiktoken.encode_ordinary(&text).len();
    }
    let tiktoken_time = start.elapsed();
    let tiktoken_tp = (bytes * ITERATIONS) as f64 / tiktoken_time.as_secs_f64() / 1024.0 / 1024.0;

    println!("tiktoken-rs");
    println!("  Tokens: {}", tiktoken_tokens);
    println!("  Time:   {:.2} ms", tiktoken_time.as_secs_f64() * 1000.0 / ITERATIONS as f64);
    println!("  Throughput: {:.1} MiB/s\n", tiktoken_tp);

    // Summary
    println!("==============================");
    println!("Summary (throughput):");
    println!("  tokie:      {:>7.1} MiB/s", tokie_tp);
    println!("  HF:         {:>7.1} MiB/s ({:.2}x vs tokie)", hf_tp, tokie_tp / hf_tp);
    println!("  tiktoken:   {:>7.1} MiB/s ({:.2}x vs tokie)", tiktoken_tp, tokie_tp / tiktoken_tp);

    // Verify token counts match
    println!("\nToken count verification:");
    if tokie_tokens == hf_tokens && tokie_tokens == tiktoken_tokens {
        println!("  All tokenizers produce {} tokens ✓", tokie_tokens);
    } else {
        println!("  tokie: {}, HF: {}, tiktoken: {} ✗", tokie_tokens, hf_tokens, tiktoken_tokens);
    }
}
