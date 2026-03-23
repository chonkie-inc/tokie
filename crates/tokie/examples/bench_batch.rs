//! Benchmark batch encoding: Approach A (distributed) vs B (sequential) vs HF
//!
//! Run with: cargo run --example bench_batch --release

use std::time::Instant;

const ITERATIONS: usize = 5;

fn main() {
    let text = std::fs::read_to_string("benches/data/war_and_peace.txt")
        .expect("Failed to read war_and_peace.txt");

    // Load tokie from .tkz
    let tokie_tok = tokie::Tokenizer::from_file("../../models/bert.tkz")
        .expect("Failed to load models/bert.tkz");

    // Load HF tokenizers
    let hf_tok = tokenizers::Tokenizer::from_file("/tmp/tokenizers/bert_tokenizer.json")
        .expect("Failed to load bert_tokenizer.json");

    // === Workload 1: Many small texts (lines) ===
    let lines: Vec<&str> = text.lines().filter(|l| !l.is_empty()).collect();
    let total_bytes_lines: usize = lines.iter().map(|l| l.len()).sum();
    let mb_lines = total_bytes_lines as f64 / 1_000_000.0;
    println!("=== Workload 1: Many small texts ({} lines, {:.2} MB) ===\n", lines.len(), mb_lines);
    bench_workload(&tokie_tok, &hf_tok, &lines);

    // === Workload 2: Few large texts (chapters split at double newlines) ===
    let chapters: Vec<&str> = text.split("\n\n\n").filter(|c| c.len() > 1000).collect();
    let total_bytes_chapters: usize = chapters.iter().map(|c| c.len()).sum();
    let mb_chapters = total_bytes_chapters as f64 / 1_000_000.0;
    println!("\n=== Workload 2: Few large texts ({} chunks, {:.2} MB) ===\n", chapters.len(), mb_chapters);
    bench_workload(&tokie_tok, &hf_tok, &chapters);

    // === Workload 3: Mixed sizes ===
    let mut mixed: Vec<&str> = Vec::new();
    // Add some short texts
    for line in lines.iter().take(100) {
        mixed.push(line);
    }
    // Add some large chunks
    for chapter in chapters.iter().take(5) {
        mixed.push(chapter);
    }
    let total_bytes_mixed: usize = mixed.iter().map(|t| t.len()).sum();
    let mb_mixed = total_bytes_mixed as f64 / 1_000_000.0;
    println!("\n=== Workload 3: Mixed sizes ({} texts, {:.2} MB) ===\n", mixed.len(), mb_mixed);
    bench_workload(&tokie_tok, &hf_tok, &mixed);
}

fn bench_workload(tokie_tok: &tokie::Tokenizer, hf_tok: &tokenizers::Tokenizer, texts: &[&str]) {
    let total_bytes: usize = texts.iter().map(|t| t.len()).sum();
    let mb = total_bytes as f64 / 1_000_000.0;

    // Warmup
    let _ = tokie_tok.encode_batch(texts, false);
    let _ = hf_tok.encode_batch(texts.to_vec(), false);

    // Approach A: distributed (encode_batch)
    let start = Instant::now();
    let mut token_count_a = 0;
    for _ in 0..ITERATIONS {
        let results = tokie_tok.encode_batch(texts, false);
        token_count_a = results.iter().map(|r| r.len()).sum();
    }
    let elapsed_a = start.elapsed() / ITERATIONS as u32;
    let tp_a = mb / elapsed_a.as_secs_f64();
    println!("  Approach A (distributed):  {:.2} ms  {:.1} MB/s  ({} tokens)",
        elapsed_a.as_secs_f64() * 1000.0, tp_a, token_count_a);

    // Approach B: sequential loop (each encode may parallelize internally)
    let start = Instant::now();
    let mut token_count_b = 0;
    for _ in 0..ITERATIONS {
        let results: Vec<Vec<u32>> = texts.iter().map(|t| tokie_tok.encode(t, false)).collect();
        token_count_b = results.iter().map(|r| r.len()).sum();
    }
    let elapsed_b = start.elapsed() / ITERATIONS as u32;
    let tp_b = mb / elapsed_b.as_secs_f64();
    println!("  Approach B (sequential):   {:.2} ms  {:.1} MB/s  ({} tokens)",
        elapsed_b.as_secs_f64() * 1000.0, tp_b, token_count_b);

    // HF tokenizers encode_batch
    let start = Instant::now();
    let mut token_count_hf = 0;
    for _ in 0..ITERATIONS {
        let results = hf_tok.encode_batch(texts.to_vec(), false).unwrap();
        token_count_hf = results.iter().map(|r| r.get_ids().len()).sum();
    }
    let elapsed_hf = start.elapsed() / ITERATIONS as u32;
    let tp_hf = mb / elapsed_hf.as_secs_f64();
    println!("  HF tokenizers batch:       {:.2} ms  {:.1} MB/s  ({} tokens)",
        elapsed_hf.as_secs_f64() * 1000.0, tp_hf, token_count_hf);

    let best_tp = tp_a.max(tp_b);
    let winner = if tp_a >= tp_b { "A (distributed)" } else { "B (sequential)" };
    println!("\n  Winner: {} ({:.1}x vs HF)", winner, best_tp / tp_hf);
    assert_eq!(token_count_a, token_count_b, "Approach A and B token counts must match");
}
