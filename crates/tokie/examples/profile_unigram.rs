//! Profile Unigram encoder to identify bottlenecks.
//!
//! Usage: cargo run --release --example profile_unigram

use std::fs;
use std::time::Instant;
use tokie::Tokenizer;

fn main() {
    println!("=== Unigram Encoder Profiling ===\n");

    // Load T5 tokenizer (Unigram)
    let tokenizer_path = "models/t5.tkz";
    println!("Loading tokenizer from {}...", tokenizer_path);
    let tokenizer = Tokenizer::from_file(tokenizer_path).expect("Failed to load tokenizer");
    println!("Encoder type: {:?}", tokenizer.encoder_type());
    println!("Vocab size: {}\n", tokenizer.vocab_size());

    // Test with enwik8 samples of varying sizes
    println!("=== Enwik8 Test (varying sizes) ===\n");
    if let Ok(enwik8) = fs::read_to_string("benches/data/enwik8") {
        for &size in &[10_000, 100_000, 1_000_000, 10_000_000] {
            let sample = &enwik8[..size.min(enwik8.len())];

            // Warmup
            let _ = tokenizer.encode(sample, false);

            // Benchmark
            let iterations = if size >= 1_000_000 { 3 } else { 10 };
            let start = Instant::now();
            let mut total_tokens = 0;
            for _ in 0..iterations {
                let tokens = tokenizer.encode(sample, false);
                total_tokens = tokens.len();
            }
            let elapsed = start.elapsed();

            let throughput = (sample.len() * iterations) as f64 / elapsed.as_secs_f64() / 1_000_000.0;
            let size_str = if size >= 1_000_000 {
                format!("{}MB", size / 1_000_000)
            } else {
                format!("{}KB", size / 1_000)
            };
            println!("Enwik8 {:>5}: {:>8.2} MB/s | {} tokens", size_str, throughput, total_tokens);
        }
    }

    // Test with FULL enwik8 (100MB) - compare regular vs chunked
    println!("\n=== Full Enwik8 (100MB) - Regular vs Chunked ===\n");
    if let Ok(enwik8) = fs::read_to_string("benches/data/enwik8") {
        println!("Full enwik8 size: {} bytes ({:.2} MB)", enwik8.len(), enwik8.len() as f64 / 1_000_000.0);

        // Get the Unigram encoder directly
        let unigram = tokenizer.encoder().as_unigram().expect("Expected Unigram encoder");
        let enwik8_bytes = enwik8.as_bytes();

        // Warmup
        let _ = unigram.encode(&enwik8_bytes[..100_000]);

        // Test 1: encode_single on 10MB (non-chunked baseline)
        println!("\nBaseline: encode_single on 10MB sample:");
        let sample_10mb = &enwik8_bytes[..10_000_000];
        let start = Instant::now();
        let tokens_single = unigram.encode_single(sample_10mb);
        let elapsed = start.elapsed();
        let throughput = sample_10mb.len() as f64 / elapsed.as_secs_f64() / 1_000_000.0;
        println!("  Throughput: {:>8.2} MB/s | {} tokens | {:>6}ms", throughput, tokens_single.len(), elapsed.as_millis());

        // Test 2: encode_chunked on same 10MB
        println!("\nChunked: encode_chunked on 10MB sample:");
        let start = Instant::now();
        let tokens_chunked = unigram.encode_chunked(sample_10mb, 64 * 1024);
        let elapsed = start.elapsed();
        let throughput = sample_10mb.len() as f64 / elapsed.as_secs_f64() / 1_000_000.0;
        println!("  Throughput: {:>8.2} MB/s | {} tokens | {:>6}ms", throughput, tokens_chunked.len(), elapsed.as_millis());

        // Verify exact match
        if tokens_single == tokens_chunked {
            println!("\n✓ EXACT MATCH: {} tokens", tokens_single.len());
        } else {
            println!("\n✗ MISMATCH: single={}, chunked={}", tokens_single.len(), tokens_chunked.len());
            // Find first difference
            for (i, (a, b)) in tokens_single.iter().zip(tokens_chunked.iter()).enumerate() {
                if a != b {
                    println!("  First diff at position {}: single={}, chunked={}", i, a, b);
                    break;
                }
            }
        }
    }

    println!("\n=== Simple Text Test ===\n");

    // Test cases of varying sizes
    let test_sizes = [10, 100, 1000, 10000];
    let base_text = "The quick brown fox jumps over the lazy dog. ";

    for &size in &test_sizes {
        let text = base_text.repeat(size / base_text.len() + 1);
        let text = &text[..size.min(text.len())];

        // Warmup
        for _ in 0..3 {
            let _ = tokenizer.encode(text, false);
        }

        // Benchmark
        let iterations = if size < 1000 { 1000 } else { 100 };
        let start = Instant::now();
        let mut total_tokens = 0;
        for _ in 0..iterations {
            let tokens = tokenizer.encode(text, false);
            total_tokens = tokens.len();
        }
        let elapsed = start.elapsed();

        let total_bytes = text.len() * iterations;
        let throughput = total_bytes as f64 / elapsed.as_secs_f64() / 1_000_000.0;
        let us_per_call = elapsed.as_micros() as f64 / iterations as f64;

        println!(
            "Size {:>5} bytes: {:>8.2} MB/s | {:>8.1} µs/call | {} tokens",
            size, throughput, us_per_call, total_tokens
        );
    }

    // Profile allocation overhead by comparing first call vs subsequent calls
    println!("\n=== Allocation Profiling ===\n");

    let test_text = base_text.repeat(100);
    let test_text = &test_text[..4000];

    // First call (cold)
    let start = Instant::now();
    let _ = tokenizer.encode(test_text, false);
    let first_call = start.elapsed();

    // Subsequent calls (but still allocating)
    let iterations = 1000;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = tokenizer.encode(test_text, false);
    }
    let avg_call = start.elapsed().as_nanos() as f64 / iterations as f64;

    println!("First call:      {:>10.1} µs", first_call.as_micros());
    println!("Avg subsequent:  {:>10.1} µs", avg_call / 1000.0);

    println!("\n=== Done ===");
}
