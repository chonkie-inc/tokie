//! Profile BERT tokenizer stages

use std::time::Instant;
use tokie::{Tokenizer, clean_text, strip_accents};
use tokie::normalizer::bert_uncased_normalize;
use tokie::pretok::Pretok;

fn main() {
    let text = std::fs::read_to_string("benches/data/war_and_peace.txt").unwrap();
    let text_bytes = text.len();

    println!("Text size: {:.2} MB ({} bytes)\n", text_bytes as f64 / 1_000_000.0, text_bytes);

    let tokie = Tokenizer::from_file("models/bert.tkz").unwrap();
    let iterations = 10;

    // Stage 1: Separate normalization steps
    println!("=== Separate Normalization (3 passes) ===");
    let start = Instant::now();
    let mut normalized = String::new();
    for _ in 0..iterations {
        let cleaned = clean_text(&text);
        let stripped = strip_accents(&cleaned);
        normalized = stripped.to_lowercase();
    }
    let elapsed = start.elapsed();
    let separate_time = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    println!("  Time: {:.2} ms/iter", separate_time);
    println!("  Throughput: {:.1} MB/s", (text_bytes * iterations) as f64 / elapsed.as_secs_f64() / 1_000_000.0);
    println!("  Normalized size: {} bytes", normalized.len());

    // Stage 1b: Fused normalization
    println!("\n=== Fused Normalization (1 pass) ===");
    let start = Instant::now();
    let mut fused_normalized = std::borrow::Cow::Borrowed("");
    for _ in 0..iterations {
        fused_normalized = bert_uncased_normalize(&text);
    }
    let elapsed = start.elapsed();
    let fused_time = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    println!("  Time: {:.2} ms/iter", fused_time);
    println!("  Throughput: {:.1} MB/s", (text_bytes * iterations) as f64 / elapsed.as_secs_f64() / 1_000_000.0);
    println!("  Normalized size: {} bytes", fused_normalized.len());
    println!("  Speedup: {:.2}x", separate_time / fused_time);

    // Verify outputs match
    if normalized != fused_normalized.as_ref() {
        println!("  WARNING: Outputs differ!");
        // Find first difference
        for (i, (a, b)) in normalized.chars().zip(fused_normalized.chars()).enumerate() {
            if a != b {
                println!("    First diff at char {}: {:?} vs {:?}", i, a, b);
                break;
            }
        }
    } else {
        println!("  Outputs match!");
    }

    let normalized = fused_normalized.into_owned();

    // Stage 2: Pretokenization
    println!("\n=== Stage 2: Pretokenization ===");
    let start = Instant::now();
    let mut pretok_count = 0;
    for _ in 0..iterations {
        pretok_count = Pretok::BERT.split(&normalized).count();
    }
    let elapsed = start.elapsed();
    println!("  Time: {:.2} ms/iter", elapsed.as_secs_f64() * 1000.0 / iterations as f64);
    println!("  Throughput: {:.1} MB/s", (normalized.len() * iterations) as f64 / elapsed.as_secs_f64() / 1_000_000.0);
    println!("  Pretokens: {}", pretok_count);

    // Stage 2b: Pretokenization with collection
    println!("\n=== Stage 2b: Pretokenization (collect to Vec) ===");
    let start = Instant::now();
    let mut pretokens: Vec<&str> = Vec::new();
    for _ in 0..iterations {
        pretokens = Pretok::BERT.split(&normalized).collect();
    }
    let elapsed = start.elapsed();
    println!("  Time: {:.2} ms/iter", elapsed.as_secs_f64() * 1000.0 / iterations as f64);
    println!("  Throughput: {:.1} MB/s", (normalized.len() * iterations) as f64 / elapsed.as_secs_f64() / 1_000_000.0);
    println!("  Pretokens: {}", pretokens.len());

    // Stage 3: WordPiece encoding only
    println!("\n=== Stage 3: WordPiece Encoding ===");
    let encoder = match tokie.encoder() {
        tokie::encoder::Encoder::WordPiece(wp) => wp,
        _ => panic!("Expected WordPiece encoder"),
    };

    let start = Instant::now();
    let mut token_count = 0;
    for _ in 0..iterations {
        token_count = 0;
        for pretoken in &pretokens {
            token_count += encoder.encode(pretoken.as_bytes()).len();
        }
    }
    let elapsed = start.elapsed();
    let total_pretoken_bytes: usize = pretokens.iter().map(|s| s.len()).sum();
    println!("  Time: {:.2} ms/iter", elapsed.as_secs_f64() * 1000.0 / iterations as f64);
    println!("  Throughput: {:.1} MB/s", (total_pretoken_bytes * iterations) as f64 / elapsed.as_secs_f64() / 1_000_000.0);
    println!("  Tokens: {}", token_count);

    // Stage 4: Full pipeline
    println!("\n=== Full Pipeline (encode) ===");
    let start = Instant::now();
    let mut full_tokens = 0;
    for _ in 0..iterations {
        full_tokens = tokie.encode(&text, false).len();
    }
    let elapsed = start.elapsed();
    println!("  Time: {:.2} ms/iter", elapsed.as_secs_f64() * 1000.0 / iterations as f64);
    println!("  Throughput: {:.1} MB/s", (text_bytes * iterations) as f64 / elapsed.as_secs_f64() / 1_000_000.0);
    println!("  Tokens: {}", full_tokens);

    // Summary
    println!("\n=== Summary ===");
    println!("Breakdown of full pipeline time:");
}
