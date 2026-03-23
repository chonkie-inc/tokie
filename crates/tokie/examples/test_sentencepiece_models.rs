//! Test SentencePiece tokenizer models against HuggingFace and create .tkz files.
//!
//! Run with: cargo run --release --example test_sentencepiece_models

use std::path::Path;
use std::time::Instant;
use tokenizers::Tokenizer as HfTokenizer;
use tokie::hf;

/// Download a tokenizer.json file from HuggingFace if not present.
fn download_if_needed(json_path: &str, url: &str) -> Result<(), String> {
    if !Path::new(json_path).exists() {
        println!("  Downloading from HuggingFace...");

        // Try to get HF token for gated models
        let hf_token = std::env::var("HF_TOKEN").ok()
            .or_else(|| std::fs::read_to_string(
                dirs::home_dir()?.join(".cache/huggingface/token")
            ).ok().map(|s| s.trim().to_string()));

        let mut cmd = std::process::Command::new("curl");
        cmd.args(["-sL", "-o", json_path]);

        if let Some(token) = hf_token {
            cmd.args(["-H", &format!("Authorization: Bearer {}", token)]);
        }

        cmd.arg(url);

        let status = cmd.status().map_err(|e| e.to_string())?;
        if !status.success() {
            return Err(format!("Failed to download from {}", url));
        }

        // Check if we got an error response (HTML instead of JSON)
        let content = std::fs::read_to_string(json_path).unwrap_or_default();
        if content.contains("<!DOCTYPE") || content.contains("<html") {
            std::fs::remove_file(json_path).ok();
            return Err("Received HTML error page - model may require authentication".to_string());
        }
    }
    Ok(())
}

/// Test a tokenizer against HuggingFace and create .tkz file.
fn test_and_create(
    name: &str,
    json_path: &str,
    url: &str,
    test_text: &str,
) -> Result<(bool, usize, f64, f64), String> {
    println!("\n{}", "=".repeat(70));
    println!("Testing: {}", name);
    println!("{}", "=".repeat(70));

    // Download if needed
    download_if_needed(json_path, url)?;

    // Load HuggingFace tokenizer
    let hf = HfTokenizer::from_file(json_path).map_err(|e| e.to_string())?;

    // Load tokie tokenizer
    let tokie = hf::from_json(json_path).map_err(|e| e.to_string())?;

    println!("  Vocab size: {}", tokie.vocab_size());
    println!("  Encoder: {:?}", tokie.encoder_type());
    println!("  Normalizer: {:?}", tokie.normalizer());
    println!("  Pretokenizer: {:?}", tokie.pretokenizer_type());

    // Test on sample text
    let hf_result = hf.encode(test_text, false).map_err(|e| e.to_string())?;
    let hf_tokens: Vec<u32> = hf_result.get_ids().to_vec();

    let tokie_tokens = tokie.encode(test_text, false);

    let matches = hf_tokens == tokie_tokens;

    println!("\n  Sample test ({} chars):", test_text.len());
    println!("    HuggingFace: {} tokens", hf_tokens.len());
    println!("    Tokie:       {} tokens", tokie_tokens.len());
    println!("    Match: {}", if matches { "YES" } else { "NO" });

    if !matches && hf_tokens.len() < 50 {
        println!("    HF:    {:?}", hf_tokens);
        println!("    Tokie: {:?}", tokie_tokens);
    }

    // Benchmark on War and Peace if available
    let (throughput_hf, throughput_tokie) = if let Ok(war_and_peace) = std::fs::read_to_string("benches/data/war_and_peace.txt") {
        let bytes = war_and_peace.len();

        // Benchmark HuggingFace
        let start = Instant::now();
        let hf_wap = hf.encode(war_and_peace.as_str(), false).map_err(|e| e.to_string())?;
        let hf_time = start.elapsed();
        let hf_throughput = bytes as f64 / hf_time.as_secs_f64() / 1_000_000.0;

        // Benchmark tokie
        let start = Instant::now();
        let tokie_wap = tokie.encode(&war_and_peace, false);
        let tokie_time = start.elapsed();
        let tokie_throughput = bytes as f64 / tokie_time.as_secs_f64() / 1_000_000.0;

        let wap_matches = hf_wap.get_ids() == tokie_wap.as_slice();

        println!("\n  War and Peace ({:.2} MB):", bytes as f64 / 1_000_000.0);
        println!("    HuggingFace: {} tokens in {:?} ({:.2} MB/s)",
                 hf_wap.get_ids().len(), hf_time, hf_throughput);
        println!("    Tokie:       {} tokens in {:?} ({:.2} MB/s)",
                 tokie_wap.len(), tokie_time, tokie_throughput);
        println!("    Match: {}", if wap_matches { "YES" } else { "NO" });
        println!("    Speedup: {:.2}x", tokie_throughput / hf_throughput);

        if !wap_matches {
            // Find first difference
            let first_diff = hf_wap.get_ids()
                .iter()
                .zip(tokie_wap.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(hf_wap.get_ids().len().min(tokie_wap.len()));
            println!("    First diff at token {}", first_diff);
        }

        (hf_throughput, tokie_throughput)
    } else {
        (0.0, 0.0)
    };

    // Save .tkz file
    let tkz_path = format!("models/{}.tkz", name);
    tokie.to_file(&tkz_path).map_err(|e| e.to_string())?;
    println!("\n  Saved: {}", tkz_path);

    // Get file size
    let file_size = std::fs::metadata(&tkz_path).map_err(|e| e.to_string())?.len();
    println!("  Size: {:.2} MB", file_size as f64 / 1_000_000.0);

    Ok((matches, tokie.vocab_size(), throughput_hf, throughput_tokie))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("SentencePiece Tokenizer Tests\n");

    // Ensure models directory exists
    std::fs::create_dir_all("models")?;

    let test_text = "Hello, world! This is a test of the tokenizer. \
                     The quick brown fox jumps over the lazy dog. \
                     Special chars: @#$%^&*() and numbers 12345. \
                     Unicode: 你好世界 привет мир";

    let mut results = Vec::new();

    // Mistral 7B - SentencePiece BPE
    results.push((
        "mistral_7b",
        test_and_create(
            "mistral_7b",
            "/tmp/mistral_tokenizer.json",
            "https://huggingface.co/mistralai/Mistral-7B-v0.1/raw/main/tokenizer.json",
            test_text,
        ),
    ));

    // Mixtral 8x7B - SentencePiece BPE (same tokenizer as Mistral)
    results.push((
        "mixtral_8x7b",
        test_and_create(
            "mixtral_8x7b",
            "/tmp/mixtral_tokenizer.json",
            "https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/raw/main/tokenizer.json",
            test_text,
        ),
    ));

    // Qwen2 - ByteLevel vocab-defined BPE (tiktoken-style)
    results.push((
        "qwen2",
        test_and_create(
            "qwen2",
            "/tmp/qwen2_tokenizer.json",
            "https://huggingface.co/Qwen/Qwen2-0.5B/raw/main/tokenizer.json",
            test_text,
        ),
    ));

    // Code Llama - SentencePiece BPE (32K vocab, same tokenizer as Llama 2)
    results.push((
        "codellama",
        test_and_create(
            "codellama",
            "/tmp/codellama_tokenizer.json",
            "https://huggingface.co/codellama/CodeLlama-7b-hf/raw/main/tokenizer.json",
            test_text,
        ),
    ));

    // Phi-3 - SentencePiece BPE (32K vocab)
    results.push((
        "phi3",
        test_and_create(
            "phi3",
            "/tmp/phi3_tokenizer.json",
            "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/raw/main/tokenizer.json",
            test_text,
        ),
    ));

    // Summary
    println!("\n{}", "=".repeat(70));
    println!("SUMMARY");
    println!("{}", "=".repeat(70));
    println!("\n| Model | Vocab | Match | HF MB/s | Tokie MB/s | Speedup |");
    println!("|-------|------:|:-----:|--------:|-----------:|--------:|");

    for (name, result) in &results {
        match result {
            Ok((matches, vocab, hf_tp, tokie_tp)) => {
                let speedup = if *hf_tp > 0.0 { tokie_tp / hf_tp } else { 0.0 };
                println!("| {} | {} | {} | {:.1} | {:.1} | {:.1}x |",
                    name, vocab,
                    if *matches { "YES" } else { "NO" },
                    hf_tp, tokie_tp, speedup);
            }
            Err(e) => {
                println!("| {} | - | ERROR | - | - | - | ({})", name, e);
            }
        }
    }

    // Count successes
    let total = results.len();
    let passed = results.iter().filter(|(_, r)| r.as_ref().map(|(m, _, _, _)| *m).unwrap_or(false)).count();
    let failed = results.iter().filter(|(_, r)| r.as_ref().map(|(m, _, _, _)| !m).unwrap_or(false)).count();
    let errors = results.iter().filter(|(_, r)| r.is_err()).count();

    println!("\nResults: {} passed, {} failed, {} errors (out of {})", passed, failed, errors, total);

    Ok(())
}
