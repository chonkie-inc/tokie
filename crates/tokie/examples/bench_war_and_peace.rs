//! Benchmark on War and Peace

use std::time::Instant;
use tokenizers::Tokenizer as HfTokenizer;
use tokie::hf;
use tokie::PretokType;

fn main() {
    let path = dirs::home_dir()
        .unwrap()
        .join(".cache/huggingface/hub/models--meta-llama--llama-3.2-1b/snapshots");
    let snapshot = std::fs::read_dir(&path)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let json_path = snapshot.join("tokenizer.json");

    // Load War and Peace
    let text = std::fs::read_to_string("benches/data/war_and_peace.txt").unwrap();
    let bytes = text.len();

    println!("=== War and Peace Benchmark ===\n");
    println!("Text size: {:.2} MB ({} bytes)\n", bytes as f64 / 1_000_000.0, bytes);

    println!("Loading tokenizers...");
    let hf_tok = HfTokenizer::from_file(&json_path).unwrap();
    let tokie_tok = hf::from_json_with_pretokenizer(&json_path, PretokType::Cl100k).unwrap();

    // Warm up
    let _ = hf_tok.encode(&text[..1000], false);
    let _ = tokie_tok.encode(&text, false);

    // Benchmark HuggingFace
    println!("\nBenchmarking HuggingFace...");
    let start = Instant::now();
    let hf_result = hf_tok.encode(text.as_str(), false).unwrap();
    let hf_elapsed = start.elapsed();
    let hf_tokens = hf_result.get_ids().len();
    let hf_throughput = bytes as f64 / hf_elapsed.as_secs_f64() / 1_000_000.0;

    // Benchmark Tokie
    println!("Benchmarking Tokie...");
    let start = Instant::now();
    let tokie_result = tokie_tok.encode(&text, false);
    let tokie_elapsed = start.elapsed();
    let tokie_tokens = tokie_result.len();
    let tokie_throughput = bytes as f64 / tokie_elapsed.as_secs_f64() / 1_000_000.0;

    let speedup = tokie_throughput / hf_throughput;

    println!("\n=== Results ===\n");
    println!("HuggingFace: {:>7} tokens in {:>8.2?} ({:>6.2} MB/s)", hf_tokens, hf_elapsed, hf_throughput);
    println!("Tokie:       {:>7} tokens in {:>8.2?} ({:>6.2} MB/s)", tokie_tokens, tokie_elapsed, tokie_throughput);
    println!("\nSpeedup: {:.2}x", speedup);

    // Verify correctness
    let hf_ids: Vec<u32> = hf_result.get_ids().to_vec();
    if hf_ids == tokie_result {
        println!("Output matches HuggingFace!");
    } else {
        println!("WARNING: Output mismatch!");
        println!("  HF tokens:    {}", hf_ids.len());
        println!("  Tokie tokens: {}", tokie_result.len());
    }
}
