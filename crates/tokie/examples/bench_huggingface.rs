//! Benchmark HuggingFace tokenizers (Rust) on War and Peace
//!
//! Run with: cargo run --example bench_huggingface --release

use std::time::Instant;
use tokenizers::Tokenizer;

const TEXT_PATH: &str = "benches/data/war_and_peace.txt";
const TOKENIZER_PATH: &str = "benches/data/gpt2_tokenizer.json";
const ITERATIONS: usize = 10;

fn main() {
    let text = std::fs::read_to_string(TEXT_PATH).expect("Failed to read text file");
    let mb = text.len() as f64 / (1024.0 * 1024.0);
    println!("Text size: {:.2} MB\n", mb);

    let tokenizer = Tokenizer::from_file(TOKENIZER_PATH).expect("Failed to load tokenizer");

    // Warmup
    let _ = tokenizer.encode(text.as_str(), false);

    println!("=== HuggingFace tokenizers (Rust) ===");

    // Benchmark encode
    let start = Instant::now();
    let mut token_count = 0;
    for _ in 0..ITERATIONS {
        let encoding = tokenizer.encode(text.as_str(), false).unwrap();
        token_count = encoding.get_ids().len();
    }
    let encode_elapsed = start.elapsed().as_secs_f64();
    let encode_throughput = (mb * ITERATIONS as f64) / encode_elapsed;

    println!("  Tokens: {}", token_count);
    println!("  Encode time: {:.2} ms/iter", (encode_elapsed / ITERATIONS as f64) * 1000.0);
    println!("  Encode throughput: {:.1} MiB/s", encode_throughput);

    // Get tokens for decode benchmark
    let encoding = tokenizer.encode(text.as_str(), false).unwrap();
    let tokens: Vec<u32> = encoding.get_ids().to_vec();

    // Benchmark decode
    let start = Instant::now();
    let mut decoded = String::new();
    for _ in 0..ITERATIONS {
        decoded = tokenizer.decode(&tokens, false).unwrap();
    }
    let decode_elapsed = start.elapsed().as_secs_f64();
    let decode_throughput = (mb * ITERATIONS as f64) / decode_elapsed;

    println!("  Decode time: {:.2} ms/iter", (decode_elapsed / ITERATIONS as f64) * 1000.0);
    println!("  Decode throughput: {:.1} MiB/s", decode_throughput);

    // Verify roundtrip
    let roundtrip_ok = decoded == text;
    println!("  Roundtrip OK: {}", roundtrip_ok);
}
