//! Download, convert, and benchmark all supported tokenizers
//!
//! Run with:
//!   cargo run --example generate_all_models --release
//!
//! This will:
//! 1. Download tokenizer.json files from HuggingFace
//! 2. Convert them to .tkz format in models/
//! 3. Benchmark load time and encoding throughput

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

/// Model configuration: (hf_repo, output_name, display_name)
const MODELS: &[(&str, &str, &str)] = &[
    // Already have these but need benchmarks:
    // ("openai/gpt2", "gpt2", "GPT-2"), // Uses vocab.json not tokenizer.json

    // Unigram models
    ("google/albert-base-v2", "albert", "ALBERT"),
    ("google-t5/t5-base", "t5", "T5"),
    ("google/mt5-base", "mt5", "mT5"),
    ("FacebookAI/xlm-roberta-base", "xlm_roberta", "XLM-RoBERTa"),

    // SentencePiece BPE models
    ("google/gemma-2b", "gemma", "Gemma"),
    ("google/gemma-2-2b", "gemma2", "Gemma 2"),
    ("meta-llama/Llama-2-7b-hf", "llama2", "Llama 2"),
    ("microsoft/deberta-v3-base", "deberta_v3", "DeBERTa v3"),

    // Backtracking BPE models
    ("answerdotai/ModernBERT-base", "modernbert", "ModernBERT"),
    ("FacebookAI/roberta-base", "roberta", "RoBERTa"),
    ("microsoft/deberta-base", "deberta", "DeBERTa"),
    ("microsoft/phi-2", "phi2", "Phi-2"),
    ("mosaicml/mpt-7b", "mpt", "MPT"),
    ("bigcode/starcoder", "starcoder", "StarCoder"),

    // Simple BPE models
    ("mistralai/Mistral-Nemo-Instruct-2407", "mistral_nemo", "Mistral Nemo"),
];

fn download_tokenizer(repo: &str, cache_dir: &Path) -> Option<String> {
    let safe_name = repo.replace('/', "_");
    let cache_path = cache_dir.join(format!("{}_tokenizer.json", safe_name));

    if cache_path.exists() {
        return Some(cache_path.to_string_lossy().to_string());
    }

    println!("  Downloading from {}...", repo);
    let url = format!(
        "https://huggingface.co/{}/resolve/main/tokenizer.json",
        repo
    );

    let output = Command::new("curl")
        .args(["-sL", "-o", cache_path.to_str().unwrap(), &url])
        .output()
        .ok()?;

    if output.status.success() && cache_path.exists() {
        // Check if it's a valid JSON (not an error page)
        let content = fs::read_to_string(&cache_path).ok()?;
        if content.starts_with('{') && content.contains("\"model\"") {
            return Some(cache_path.to_string_lossy().to_string());
        } else {
            fs::remove_file(&cache_path).ok();
        }
    }

    None
}

fn main() {
    let cache_dir = Path::new("/tmp/tokenizer_cache");
    let models_dir = Path::new("models");

    fs::create_dir_all(cache_dir).expect("Failed to create cache dir");
    fs::create_dir_all(models_dir).expect("Failed to create models dir");

    // Load test data
    let text = include_str!("../benches/data/war_and_peace.txt");
    let text_mb = text.len() as f64 / 1_000_000.0;

    println!("=== Tokenizer Model Generator ===\n");
    println!("Test data: War and Peace ({:.2} MB)\n", text_mb);

    // Results storage
    let mut results: HashMap<String, (f64, f64)> = HashMap::new(); // name -> (load_ms, throughput)

    // First benchmark existing models
    println!("--- Benchmarking existing models ---\n");

    let existing_models = [
        ("gpt2.tkz", "GPT-2"),
        ("cl100k.tkz", "GPT-4"),
        ("o200k.tkz", "GPT-4o"),
        ("llama3.tkz", "Llama 3"),
        ("qwen2.tkz", "Qwen 2"),
        ("bert.tkz", "BERT"),
        ("mistral_7b.tkz", "Mistral 7B"),
        ("mixtral_8x7b.tkz", "Mixtral 8x7B"),
        ("codellama.tkz", "Code Llama"),
        ("phi3.tkz", "Phi-3"),
        ("voyage3_large.tkz", "Voyage 3"),
    ];

    for (filename, name) in existing_models {
        let path = models_dir.join(filename);
        if path.exists() {
            if let Some((load_ms, throughput)) = benchmark_tkz(&path, text) {
                println!("  {}: {:.1}ms load, {:.0} MB/s", name, load_ms, throughput);
                results.insert(name.to_string(), (load_ms, throughput));
            }
        }
    }

    println!("\n--- Generating new models ---\n");

    for (repo, output_name, display_name) in MODELS {
        println!("Processing {}...", display_name);

        let tkz_path = models_dir.join(format!("{}.tkz", output_name));

        // Skip if already exists
        if tkz_path.exists() {
            println!("  Already exists, benchmarking...");
            if let Some((load_ms, throughput)) = benchmark_tkz(&tkz_path, text) {
                println!("  {}: {:.1}ms load, {:.0} MB/s", display_name, load_ms, throughput);
                results.insert(display_name.to_string(), (load_ms, throughput));
            }
            continue;
        }

        // Download tokenizer.json
        let json_path = match download_tokenizer(repo, cache_dir) {
            Some(p) => p,
            None => {
                println!("  Failed to download, skipping");
                continue;
            }
        };

        // Load and convert
        println!("  Loading from JSON...");
        let tokenizer = match tokie::hf::from_json(&json_path) {
            Ok(t) => t,
            Err(e) => {
                println!("  Failed to load: {:?}", e);
                continue;
            }
        };

        // Save as .tkz
        println!("  Saving as .tkz...");
        if let Err(e) = tokenizer.to_file(&tkz_path) {
            println!("  Failed to save: {:?}", e);
            continue;
        }

        // Benchmark
        if let Some((load_ms, throughput)) = benchmark_tkz(&tkz_path, text) {
            println!("  {}: {:.1}ms load, {:.0} MB/s", display_name, load_ms, throughput);
            results.insert(display_name.to_string(), (load_ms, throughput));
        }
    }

    // Print summary table
    println!("\n=== Summary ===\n");
    println!("| Model | Load (ms) | Throughput (MB/s) |");
    println!("|-------|-----------|-------------------|");

    let mut sorted: Vec<_> = results.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(b.0));

    for (name, (load_ms, throughput)) in sorted {
        println!("| {} | {:.1} | {:.0} |", name, load_ms, throughput);
    }
}

fn benchmark_tkz(path: &Path, text: &str) -> Option<(f64, f64)> {
    let text_mb = text.len() as f64 / 1_000_000.0;

    // Measure load time
    let load_start = Instant::now();
    let tokenizer = tokie::Tokenizer::from_file(path).ok()?;
    let load_time = load_start.elapsed();

    // Warm up
    let _ = tokenizer.encode(text, false);

    // Benchmark encoding (3 iterations)
    let enc_start = Instant::now();
    for _ in 0..3 {
        let _ = tokenizer.encode(text, false);
    }
    let enc_time = enc_start.elapsed() / 3;
    let throughput = text_mb / enc_time.as_secs_f64();

    Some((load_time.as_secs_f64() * 1000.0, throughput))
}
