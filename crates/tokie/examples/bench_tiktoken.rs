//! Benchmark tiktoken-rs on War and Peace
//!
//! Run with: cargo run --example bench_tiktoken --release

use std::time::Instant;
use tiktoken_rs::{r50k_base, cl100k_base, o200k_base, CoreBPE};

const TEXT_PATH: &str = "benches/data/war_and_peace.txt";
const ITERATIONS: usize = 10;

fn main() {
    let text = std::fs::read_to_string(TEXT_PATH).expect("Failed to read text file");
    let mb = text.len() as f64 / (1024.0 * 1024.0);
    println!("Text size: {:.2} MB\n", mb);

    // GPT-2 encoding (r50k_base)
    println!("=== tiktoken-rs r50k_base (GPT-2) ===");
    let bpe = r50k_base().unwrap();
    bench_encoding(&bpe, &text, mb);

    // cl100k_base (GPT-3.5/4)
    println!("\n=== tiktoken-rs cl100k_base (GPT-3.5/4) ===");
    let bpe = cl100k_base().unwrap();
    bench_encoding(&bpe, &text, mb);

    // o200k_base (GPT-4o)
    println!("\n=== tiktoken-rs o200k_base (GPT-4o) ===");
    let bpe = o200k_base().unwrap();
    bench_encoding(&bpe, &text, mb);
}

fn bench_encoding(bpe: &CoreBPE, text: &str, mb: f64) {
    // Warmup
    let _ = bpe.encode_ordinary(text);

    // Benchmark encode
    let start = Instant::now();
    let mut token_count = 0;
    for _ in 0..ITERATIONS {
        let tokens = bpe.encode_ordinary(text);
        token_count = tokens.len();
    }
    let encode_elapsed = start.elapsed().as_secs_f64();
    let encode_throughput = (mb * ITERATIONS as f64) / encode_elapsed;

    println!("  Tokens: {}", token_count);
    println!("  Encode time: {:.2} ms/iter", (encode_elapsed / ITERATIONS as f64) * 1000.0);
    println!("  Encode throughput: {:.1} MiB/s", encode_throughput);

    // Get tokens for decode benchmark
    let tokens = bpe.encode_ordinary(text);

    // Benchmark decode
    let start = Instant::now();
    let mut decoded = String::new();
    for _ in 0..ITERATIONS {
        decoded = bpe.decode(tokens.clone()).unwrap();
    }
    let decode_elapsed = start.elapsed().as_secs_f64();
    let decode_throughput = (mb * ITERATIONS as f64) / decode_elapsed;

    println!("  Decode time: {:.2} ms/iter", (decode_elapsed / ITERATIONS as f64) * 1000.0);
    println!("  Decode throughput: {:.1} MiB/s", decode_throughput);

    // Verify roundtrip
    let roundtrip_ok = decoded == text;
    println!("  Roundtrip OK: {}", roundtrip_ok);
}
