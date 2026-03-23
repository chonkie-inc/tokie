//! Benchmark Unigram tokenizers on War and Peace
//!
//! Tests T5 and XLM-RoBERTa throughput against HuggingFace.
//!
//! Run with: cargo run --release --example bench_unigram

use std::path::PathBuf;
use std::time::Instant;
use tokenizers::Tokenizer as HfTokenizer;

fn get_hf_tokenizer_path(model_id: &str) -> Option<PathBuf> {
    let cache_dir = dirs::home_dir()?.join(".cache/huggingface/hub");
    let model_dir_name = format!("models--{}", model_id.replace('/', "--"));
    let model_dir = cache_dir.join(&model_dir_name).join("snapshots");
    let snapshot = model_dir.read_dir().ok()?.next()?.ok()?.path();
    let tokenizer_path = snapshot.join("tokenizer.json");
    if tokenizer_path.exists() {
        Some(tokenizer_path)
    } else {
        None
    }
}

struct BenchResult {
    name: &'static str,
    hf_tokens: usize,
    tokie_tokens: usize,
    hf_throughput: f64,
    tokie_throughput: f64,
    matches: bool,
}

fn bench_model(name: &'static str, model_id: &str, text: &str) -> Option<BenchResult> {
    let path = get_hf_tokenizer_path(model_id)?;

    // Load HuggingFace tokenizer
    let mut hf_tok = HfTokenizer::from_file(&path).ok()?;
    let _ = hf_tok.with_truncation(None);

    // Load Tokie tokenizer
    let tokie_tok = tokie::Tokenizer::from_json(&path).ok()?;

    let bytes = text.len();

    // Warm up
    let _ = hf_tok.encode(&text[..text.len().min(1000)], false);
    let _ = tokie_tok.encode(&text[..text.len().min(1000)], false);

    // Benchmark HuggingFace
    let start = Instant::now();
    let hf_result = hf_tok.encode(text, false).ok()?;
    let hf_elapsed = start.elapsed();
    let hf_tokens = hf_result.get_ids().len();
    let hf_throughput = bytes as f64 / hf_elapsed.as_secs_f64() / 1_000_000.0;

    // Benchmark Tokie
    let start = Instant::now();
    let tokie_result = tokie_tok.encode(text, false);
    let tokie_elapsed = start.elapsed();
    let tokie_tokens = tokie_result.len();
    let tokie_throughput = bytes as f64 / tokie_elapsed.as_secs_f64() / 1_000_000.0;

    // Check match
    let hf_ids: Vec<u32> = hf_result.get_ids().to_vec();
    let matches = hf_ids == tokie_result;

    Some(BenchResult {
        name,
        hf_tokens,
        tokie_tokens,
        hf_throughput,
        tokie_throughput,
        matches,
    })
}

fn main() {
    // Load War and Peace
    let text = match std::fs::read_to_string("benches/data/war_and_peace.txt") {
        Ok(t) => t,
        Err(_) => {
            eprintln!("Error: benches/data/war_and_peace.txt not found");
            eprintln!("Please ensure the benchmark data file exists.");
            return;
        }
    };
    let bytes = text.len();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║        Unigram Tokenizer Benchmark (War and Peace)           ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");
    println!("Text size: {:.2} MB ({} bytes)\n", bytes as f64 / 1_000_000.0, bytes);

    let models = [
        ("T5 Small", "google-t5/t5-small"),
        ("T5 Base", "google-t5/t5-base"),
        ("ALBERT Base", "albert/albert-base-v2"),
        ("XLM-RoBERTa", "FacebookAI/xlm-roberta-base"),
    ];

    let mut results = Vec::new();

    for (name, model_id) in models {
        print!("Testing {}... ", name);
        match bench_model(name, model_id, &text) {
            Some(result) => {
                println!("done");
                results.push(result);
            }
            None => {
                println!("skipped (not cached)");
            }
        }
    }

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    println!("{:<15} {:>10} {:>10} {:>12} {:>12} {:>8} {:>8}",
        "Model", "HF Tokens", "TK Tokens", "HF MB/s", "TK MB/s", "Speedup", "Match");
    println!("{}", "─".repeat(77));

    for r in &results {
        let speedup = r.tokie_throughput / r.hf_throughput;
        let match_str = if r.matches { "✓" } else { "✗" };
        println!("{:<15} {:>10} {:>10} {:>12.2} {:>12.2} {:>7.1}x {:>8}",
            r.name,
            r.hf_tokens,
            r.tokie_tokens,
            r.hf_throughput,
            r.tokie_throughput,
            speedup,
            match_str
        );
    }

    if !results.is_empty() {
        let avg_speedup: f64 = results.iter()
            .map(|r| r.tokie_throughput / r.hf_throughput)
            .sum::<f64>() / results.len() as f64;

        println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Average speedup: {:.1}x", avg_speedup);

        let all_match = results.iter().all(|r| r.matches);
        if all_match {
            println!("All outputs match HuggingFace!");
        } else {
            println!("WARNING: Some outputs don't match!");
        }
    }
}
