//! Benchmark multiple tokenizers for TOKENIZERS.md
//!
//! Run with:
//!   cargo run --example bench_all --release

use std::path::Path;
use std::time::Instant;

fn main() {
    let text = include_str!("../benches/data/war_and_peace.txt");
    let text_mb = text.len() as f64 / 1_000_000.0;

    println!("Benchmark: War and Peace ({:.2} MB)\n", text_mb);
    println!("| Model | Load (tokie) | Load (HF) | Load Speedup | Throughput (tokie) | Throughput (HF) | Encode Speedup |");
    println!("|-------|--------------|-----------|--------------|--------------------|-----------------| ---------------|");

    // Models to benchmark (tkz_path, json_path, name)
    let models: Vec<(&str, Option<&str>, &str)> = vec![
        ("models/gpt2.tkz", Some("benches/data/gpt2_tokenizer.json"), "GPT-2"),
        ("models/cl100k.tkz", None, "CL100K (GPT-4)"),
        ("models/o200k.tkz", None, "O200K (GPT-4o)"),
        ("models/llama3.tkz", None, "Llama 3"),
        ("models/qwen2.tkz", None, "Qwen 2"),
        ("models/bert.tkz", None, "BERT"),
        ("models/mistral_7b.tkz", None, "Mistral 7B"),
        ("models/voyage3_large.tkz", Some("/tmp/voyage3_large_tokenizer.json"), "Voyage 3"),
    ];

    for (tkz_path, json_path, name) in models {
        if !Path::new(tkz_path).exists() {
            continue;
        }

        // Load tokie
        let tokie_start = Instant::now();
        let tokie_tok = match tokie::Tokenizer::from_file(tkz_path) {
            Ok(t) => t,
            Err(_) => continue, // Skip unsupported formats
        };
        let tokie_load = tokie_start.elapsed();

        // Load HF if json available
        let (hf_load, hf_throughput) = if let Some(jp) = json_path {
            if Path::new(jp).exists() {
                let hf_start = Instant::now();
                let hf_tok = tokenizers::Tokenizer::from_file(jp).expect("HF load failed");
                let hf_load = hf_start.elapsed();

                // Warm up
                let _ = hf_tok.encode(text, false);

                // Benchmark HF
                let hf_enc_start = Instant::now();
                for _ in 0..3 {
                    let _ = hf_tok.encode(text, false);
                }
                let hf_enc_time = hf_enc_start.elapsed() / 3;
                let hf_tp = text_mb / hf_enc_time.as_secs_f64();

                (Some(hf_load), Some(hf_tp))
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        // Warm up tokie
        let _ = tokie_tok.encode(text, false);

        // Benchmark tokie
        let tokie_enc_start = Instant::now();
        for _ in 0..3 {
            let _ = tokie_tok.encode(text, false);
        }
        let tokie_enc_time = tokie_enc_start.elapsed() / 3;
        let tokie_tp = text_mb / tokie_enc_time.as_secs_f64();

        // Format output
        let load_speedup = hf_load.map(|h| h.as_secs_f64() / tokie_load.as_secs_f64());
        let enc_speedup = hf_throughput.map(|h| tokie_tp / h);

        println!(
            "| {} | {:.1}ms | {} | {} | {:.0} MB/s | {} | {} |",
            name,
            tokie_load.as_secs_f64() * 1000.0,
            hf_load.map(|h| format!("{:.1}ms", h.as_secs_f64() * 1000.0)).unwrap_or("-".to_string()),
            load_speedup.map(|s| format!("{:.1}x", s)).unwrap_or("-".to_string()),
            tokie_tp,
            hf_throughput.map(|h| format!("{:.1} MB/s", h)).unwrap_or("-".to_string()),
            enc_speedup.map(|s| format!("{:.1}x", s)).unwrap_or("-".to_string()),
        );
    }
}
