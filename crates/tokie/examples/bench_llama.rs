//! Benchmark LLaMA 3 tokenizer performance

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

    println!("=== LLaMA 3 Tokenizer Benchmark ===\n");
    println!("Loading tokenizers...");

    let load_start = Instant::now();
    let hf_tok = HfTokenizer::from_file(&json_path).unwrap();
    let hf_load_time = load_start.elapsed();

    let load_start = Instant::now();
    let tokie_tok = hf::from_json_with_pretokenizer(&json_path, PretokType::Cl100k).unwrap();
    let tokie_load_time = load_start.elapsed();

    println!("  HuggingFace load: {:?}", hf_load_time);
    println!("  Tokie load:       {:?}", tokie_load_time);

    // Test data
    let short_text = "Hello, world!";
    let medium_text = "The quick brown fox jumps over the lazy dog. ".repeat(100);
    let long_text = "The quick brown fox jumps over the lazy dog. ".repeat(10000);

    println!("\n=== Short Text ({} bytes) ===\n", short_text.len());
    benchmark(&hf_tok, &tokie_tok, short_text, 10000);

    println!("\n=== Medium Text ({} bytes) ===\n", medium_text.len());
    benchmark(&hf_tok, &tokie_tok, &medium_text, 1000);

    println!("\n=== Long Text ({} bytes) ===\n", long_text.len());
    benchmark(&hf_tok, &tokie_tok, &long_text, 10);

    // Verify correctness
    println!("\n=== Correctness Check ===\n");
    let test_texts = [
        "Hello, world!",
        "trailing whitespace   ",
        "The quick brown fox jumps over the lazy dog.",
        &medium_text,
    ];

    let mut all_match = true;
    for text in test_texts {
        let hf_tokens: Vec<u32> = hf_tok.encode(text, false).unwrap().get_ids().to_vec();
        let tokie_tokens = tokie_tok.encode(text, false);
        if hf_tokens != tokie_tokens {
            println!("MISMATCH for {:?}", &text[..text.len().min(50)]);
            println!("  HF:    {:?}", &hf_tokens[..hf_tokens.len().min(10)]);
            println!("  Tokie: {:?}", &tokie_tokens[..tokie_tokens.len().min(10)]);
            all_match = false;
        }
    }
    if all_match {
        println!("All outputs match HuggingFace reference!");
    }
}

fn benchmark(hf_tok: &HfTokenizer, tokie_tok: &tokie::Tokenizer, text: &str, iterations: usize) {
    let bytes = text.as_bytes();

    // Warm up
    let _ = hf_tok.encode(text, false);
    let _ = tokie_tok.encode(text, false);

    // Benchmark HuggingFace
    let start = Instant::now();
    let mut hf_tokens = 0;
    for _ in 0..iterations {
        hf_tokens = hf_tok.encode(text, false).unwrap().get_ids().len();
    }
    let hf_elapsed = start.elapsed();
    let hf_throughput = (bytes.len() * iterations) as f64 / hf_elapsed.as_secs_f64() / 1_000_000.0;

    // Benchmark Tokie
    let start = Instant::now();
    let mut tokie_tokens = 0;
    for _ in 0..iterations {
        tokie_tokens = tokie_tok.encode(text, false).len();
    }
    let tokie_elapsed = start.elapsed();
    let tokie_throughput =
        (bytes.len() * iterations) as f64 / tokie_elapsed.as_secs_f64() / 1_000_000.0;

    let speedup = tokie_throughput / hf_throughput;

    println!(
        "HuggingFace: {:>6} tokens, {:>8.2} MB/s ({:?})",
        hf_tokens, hf_throughput, hf_elapsed
    );
    println!(
        "Tokie:       {:>6} tokens, {:>8.2} MB/s ({:?})",
        tokie_tokens, tokie_throughput, tokie_elapsed
    );
    println!("Speedup:     {:.2}x", speedup);
}
