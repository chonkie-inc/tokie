//! Compare SentencePieceBPE (v1 - heap) vs SentencePieceBPEv2 (simple O(n²))
//!
//! This test loads a Mistral tokenizer, builds both encoder versions,
//! and compares correctness and speed.
//!
//! Usage: cargo run --release --example compare_sentencepiece

use std::fs;
use std::time::Instant;
use tokie::encoder::{SentencePieceBPE, SentencePieceBPEv2};

fn main() {
    println!("=== SentencePiece v1 vs v2 Comparison ===\n");

    // Load tokenizer.json for Mistral 7B
    let json_path = "benches/data/mistral_tokenizer.json";

    // Check if we have the tokenizer.json (need to download if not)
    if !std::path::Path::new(json_path).exists() {
        println!("Downloading Mistral tokenizer.json...");
        let output = std::process::Command::new("curl")
            .args([
                "-L",
                "-o",
                json_path,
                "https://huggingface.co/mistralai/Mistral-7B-v0.1/raw/main/tokenizer.json",
            ])
            .output()
            .expect("Failed to download");
        if !output.status.success() {
            eprintln!("Failed to download tokenizer");
            return;
        }
    }

    println!("Loading tokenizer from {}...", json_path);
    let json_str = fs::read_to_string(json_path).expect("Failed to read tokenizer.json");
    let data: serde_json::Value = serde_json::from_str(&json_str).expect("Failed to parse JSON");

    // Extract vocab and merges
    let model = &data["model"];
    let vocab_obj = model["vocab"].as_object().expect("No vocab");
    let merges_arr = model["merges"].as_array().expect("No merges");

    println!("Vocab size: {}", vocab_obj.len());
    println!("Merges: {}\n", merges_arr.len());

    // Check max token length
    let max_token_len = vocab_obj.iter()
        .map(|(s, _)| decode_sentencepiece_token(s).len())
        .max()
        .unwrap_or(0);
    println!("Max token length: {} bytes\n", max_token_len);

    // Build vocab list sorted by ID
    let mut vocab: Vec<(u32, Vec<u8>)> = vocab_obj
        .iter()
        .map(|(token_str, id_val)| {
            let id = id_val.as_u64().unwrap() as u32;
            let bytes = decode_sentencepiece_token(token_str);
            (id, bytes)
        })
        .collect();
    vocab.sort_by_key(|(id, _)| *id);

    // Parse merges
    let merges: Vec<(u32, u32)> = merges_arr
        .iter()
        .filter_map(|m| {
            let s = m.as_str()?;
            let mut parts = s.split(' ');
            let left_str = parts.next()?;
            let right_str = parts.next()?;
            let left = vocab_obj.get(left_str)?.as_u64()? as u32;
            let right = vocab_obj.get(right_str)?.as_u64()? as u32;
            Some((left, right))
        })
        .collect();

    println!("Parsed {} merges\n", merges.len());

    // Build both encoders
    println!("Building encoders...");

    let start = Instant::now();
    let (v1_encoder, _) = SentencePieceBPE::from_vocab_and_merges(&vocab, &merges, 256);
    let v1_build_time = start.elapsed();

    let start = Instant::now();
    let (v2_encoder, _) = SentencePieceBPEv2::from_vocab_and_merges(&vocab, &merges, 256);
    let v2_build_time = start.elapsed();

    println!("  v1 (heap) build time:   {:>6}ms", v1_build_time.as_millis());
    println!("  v2 (simple) build time: {:>6}ms\n", v2_build_time.as_millis());

    // Test cases (already normalized with metaspace)
    let test_cases = vec![
        "▁Hello,▁world!",
        "▁The▁quick▁brown▁fox▁jumps▁over▁the▁lazy▁dog.",
        "▁This▁is▁a▁test▁with▁numbers:▁12345▁and▁symbols:▁@#$%",
        "▁Bonjour▁le▁monde!▁Привет▁мир!▁你好世界!",
        "▁def▁fibonacci(n):\n▁▁▁▁if▁n▁<=▁1:\n▁▁▁▁▁▁▁▁return▁n",
        "▁a",
        "▁ab",
        "▁abc",
        "▁The▁Eiffel▁Tower▁is▁330▁meters▁tall.",
    ];

    println!("=== Correctness Test ===\n");

    let mut mismatches = 0;
    for (i, text) in test_cases.iter().enumerate() {
        let bytes = text.as_bytes();
        let v1_tokens = v1_encoder.encode(bytes);
        let v2_tokens = v2_encoder.encode(bytes);

        let status = if v1_tokens == v2_tokens { "✓" } else { "✗" };
        print!("Test {}: {} ", i + 1, status);

        if v1_tokens != v2_tokens {
            mismatches += 1;
            println!("MISMATCH!");
            // Use chars().take() for safe UTF-8 handling
            let input_preview: String = text.chars().take(50).collect();
            println!("  Input: {:?}", input_preview);
            println!("  v1: {:?}", &v1_tokens[..v1_tokens.len().min(10)]);
            println!("  v2: {:?}", &v2_tokens[..v2_tokens.len().min(10)]);
        } else {
            println!("({} tokens)", v1_tokens.len());
        }
    }

    println!("\nCorrectness: {}/{} tests passed\n", test_cases.len() - mismatches, test_cases.len());

    // Speed benchmark (using chunked encoding for v2)
    println!("=== Speed Benchmark (chunked, chunk_size=256) ===\n");

    let bench_text = "▁The▁quick▁brown▁fox▁jumps▁over▁the▁lazy▁dog.▁".repeat(100);
    let bench_bytes = bench_text.as_bytes();
    let chunk_size = 256; // Small chunks for efficient O(n²)
    println!("Benchmark text: {} bytes, chunk_size: {}\n", bench_bytes.len(), chunk_size);

    // Warmup
    for _ in 0..10 {
        let _ = v1_encoder.encode(bench_bytes);
        let _ = v2_encoder.encode_chunked(bench_bytes, chunk_size);
    }

    // Benchmark v1
    let iterations = 100;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = v1_encoder.encode(bench_bytes);
    }
    let v1_time = start.elapsed();
    let v1_throughput = (bench_bytes.len() as f64 * iterations as f64) / v1_time.as_secs_f64() / 1_000_000.0;

    // Benchmark v2 (chunked)
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = v2_encoder.encode_chunked(bench_bytes, chunk_size);
    }
    let v2_time = start.elapsed();
    let v2_throughput = (bench_bytes.len() as f64 * iterations as f64) / v2_time.as_secs_f64() / 1_000_000.0;

    println!("SentencePieceBPE (v1 - heap):          {:>8.2} MB/s", v1_throughput);
    println!("SentencePieceBPEv2 (v2 - chunked O(n²)): {:>8.2} MB/s", v2_throughput);
    println!("\nSpeedup: {:.1}x", v2_throughput / v1_throughput);

    // Verify chunked correctness
    let v1_tokens_bench = v1_encoder.encode(bench_bytes);
    let v2_tokens_bench = v2_encoder.encode_chunked(bench_bytes, chunk_size);
    if v1_tokens_bench == v2_tokens_bench {
        println!("✓ Chunked tokens match!");
    } else {
        println!("✗ Chunked tokens MISMATCH! v1={}, v2={}", v1_tokens_bench.len(), v2_tokens_bench.len());
    }

    // Larger benchmark
    println!("\n=== Large Text Benchmark ===\n");

    let large_text = "▁The▁quick▁brown▁fox▁jumps▁over▁the▁lazy▁dog.▁".repeat(10000);
    let large_bytes = large_text.as_bytes();
    println!("Large text: {} bytes ({:.2} MB)\n", large_bytes.len(), large_bytes.len() as f64 / 1_000_000.0);

    // Warmup
    let _ = v1_encoder.encode(large_bytes);
    let _ = v2_encoder.encode_chunked(large_bytes, chunk_size);

    // Benchmark v1
    let start = Instant::now();
    let v1_tokens = v1_encoder.encode(large_bytes);
    let v1_time = start.elapsed();
    let v1_throughput = large_bytes.len() as f64 / v1_time.as_secs_f64() / 1_000_000.0;

    // Benchmark v2 (chunked)
    let start = Instant::now();
    let v2_tokens = v2_encoder.encode_chunked(large_bytes, chunk_size);
    let v2_time = start.elapsed();
    let v2_throughput = large_bytes.len() as f64 / v2_time.as_secs_f64() / 1_000_000.0;

    println!("v1 (heap):             {:>8.2} MB/s ({:>6}ms, {} tokens)", v1_throughput, v1_time.as_millis(), v1_tokens.len());
    println!("v2 (chunked O(n²)):    {:>8.2} MB/s ({:>6}ms, {} tokens)", v2_throughput, v2_time.as_millis(), v2_tokens.len());
    println!("\nSpeedup: {:.1}x", v2_throughput / v1_throughput);

    if v1_tokens == v2_tokens {
        println!("\n✓ Large text tokens match!");
    } else {
        println!("\n✗ Large text tokens MISMATCH!");
    }

    println!("\n=== Done ===");
}

/// Decode a SentencePiece token string to bytes.
/// Handles metaspace (▁) and byte fallbacks (<0xXX>).
fn decode_sentencepiece_token(s: &str) -> Vec<u8> {
    // Handle byte fallback tokens like <0x0A>
    if s.starts_with("<0x") && s.ends_with('>') && s.len() == 6 {
        if let Ok(byte) = u8::from_str_radix(&s[3..5], 16) {
            return vec![byte];
        }
    }

    // Handle special tokens
    if s.starts_with('<') && s.ends_with('>') {
        return s.as_bytes().to_vec();
    }

    // Regular token - just return bytes
    // Note: metaspace (▁) is already a UTF-8 character
    s.as_bytes().to_vec()
}
