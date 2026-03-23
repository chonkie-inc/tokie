//! Profile BPE encoding to find bottlenecks

use std::time::Instant;

fn main() {
    let json_path = "/Users/bhavnick/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.1/snapshots/27d67f1b5f57dc0953326b2601d68371d40ea8da/tokenizer.json";
    
    let tokie = tokie::Tokenizer::from_json(json_path).expect("load");
    
    // Load War and Peace
    let text = std::fs::read_to_string("benches/data/war_and_peace.txt").expect("read");
    let metaspace = "▁";
    let normalized_text: String = text
        .chars()
        .map(|c| if c == ' ' { metaspace.chars().next().unwrap() } else { c })
        .collect();
    let bytes = normalized_text.as_bytes();
    
    println!("Text size: {:.2} MB", bytes.len() as f64 / 1_000_000.0);
    
    // Get the encoder
    let encoder = tokie.encoder().as_sentencepiece().expect("sentencepiece");
    
    // Time just the BPE encoding (no pretokenization)
    let iterations = 3;
    let mut times = Vec::new();
    let mut token_count = 0;
    
    for i in 0..iterations {
        let start = Instant::now();
        let tokens = encoder.encode(bytes);
        let elapsed = start.elapsed();
        token_count = tokens.len();
        times.push(elapsed.as_secs_f64());
        println!("Run {}: {:.2}ms ({} tokens)", i+1, elapsed.as_secs_f64() * 1000.0, token_count);
    }
    
    let avg = times.iter().sum::<f64>() / iterations as f64;
    let throughput = bytes.len() as f64 / avg / 1_000_000.0;
    println!("\nAverage: {:.2}ms ({:.1} MB/s)", avg * 1000.0, throughput);
    println!("Tokens: {}", token_count);
    
    // Estimate operations
    let initial_symbols = normalized_text.chars().count();
    let merges_done = initial_symbols - token_count;
    println!("\nEstimated stats:");
    println!("  Initial symbols: {}", initial_symbols);
    println!("  Final tokens: {}", token_count);
    println!("  Merges performed: ~{}", merges_done);
    println!("  Time per merge: ~{:.0} ns", avg * 1_000_000_000.0 / merges_done as f64);
}
