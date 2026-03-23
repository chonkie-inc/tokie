//! Compare tokie BPE encoders vs huggingface tokenizers vs tiktoken vs kitoken
//!
//! Run with: cargo run --release --example compare_simple_bpe

use std::fs;
use std::time::Instant;

use kitoken::Kitoken;

const TEXT_PATH: &str = "benches/data/war_and_peace.txt";
const TOKENIZER_PATH: &str = "benches/data/gpt2_tokenizer.json";
const ITERATIONS: usize = 10;

fn main() {
    let text = fs::read_to_string(TEXT_PATH).expect("Failed to read text file");
    let bytes = text.len();

    println!("BPE Encoder Comparison");
    println!("======================");
    println!("Text: War and Peace ({:.2} MB, {} bytes)", bytes as f64 / 1_000_000.0, bytes);
    println!("Iterations: {}\n", ITERATIONS);

    // Load GPT-2 tokenizer data
    let json_str = fs::read_to_string(TOKENIZER_PATH).expect("Failed to read tokenizer");
    let data: serde_json::Value = serde_json::from_str(&json_str).unwrap();

    let model = &data["model"];
    let vocab_map = model["vocab"].as_object().unwrap();
    let merges_arr = model["merges"].as_array().unwrap();

    let mut vocab: Vec<(String, u32)> = vocab_map
        .iter()
        .map(|(k, v)| (k.clone(), v.as_u64().unwrap() as u32))
        .collect();
    vocab.sort_by_key(|(_, id)| *id);

    let full_vocab: Vec<(u32, Vec<u8>)> = vocab
        .iter()
        .map(|(s, id)| (*id, decode_bytelevel(s)))
        .collect();

    let merges: Vec<(u32, u32)> = merges_arr
        .iter()
        .filter_map(|m| {
            let s = m.as_str()?;
            let mut parts = s.split(' ');
            let left = vocab_map.get(parts.next()?)?.as_u64()? as u32;
            let right = vocab_map.get(parts.next()?)?.as_u64()? as u32;
            Some((left, right))
        })
        .collect();

    // Build tokie encoders
    let (simple_enc, _) =
        tokie::encoder::BytePairEncoder::from_vocab_and_merges(&full_vocab, &merges, 256);
    let (back_enc, _) =
        tokie::encoder::BacktrackingBytePairEncoder::from_vocab_and_merges(&full_vocab, &merges, 256);
    let pretok = tokie::pretok::PretokType::Gpt2.to_pretok().unwrap();

    // 1. tokie Simple BPE
    // Warmup
    let _: Vec<u32> = pretok
        .split(&text)
        .flat_map(|piece| simple_enc.encode(piece.as_bytes()))
        .collect();

    let start = Instant::now();
    let mut simple_tokens = 0;
    for _ in 0..ITERATIONS {
        let tokens: Vec<u32> = pretok
            .split(&text)
            .flat_map(|piece| simple_enc.encode(piece.as_bytes()))
            .collect();
        simple_tokens = tokens.len();
    }
    let simple_time = start.elapsed();
    let simple_ms = simple_time.as_secs_f64() * 1000.0 / ITERATIONS as f64;
    let simple_tp = (bytes * ITERATIONS) as f64 / simple_time.as_secs_f64() / 1_000_000.0;

    println!("tokie Simple BPE");
    println!("  Tokens:     {}", simple_tokens);
    println!("  Time:       {:.1} ms", simple_ms);
    println!("  Throughput: {:.1} MB/s\n", simple_tp);

    // 2. tokie Backtracking
    // Warmup
    let _: Vec<u32> = pretok
        .split(&text)
        .flat_map(|piece| back_enc.encode(piece.as_bytes()))
        .collect();

    let start = Instant::now();
    let mut back_tokens = 0;
    for _ in 0..ITERATIONS {
        let tokens: Vec<u32> = pretok
            .split(&text)
            .flat_map(|piece| back_enc.encode(piece.as_bytes()))
            .collect();
        back_tokens = tokens.len();
    }
    let back_time = start.elapsed();
    let back_ms = back_time.as_secs_f64() * 1000.0 / ITERATIONS as f64;
    let back_tp = (bytes * ITERATIONS) as f64 / back_time.as_secs_f64() / 1_000_000.0;

    println!("tokie Backtracking");
    println!("  Tokens:     {}", back_tokens);
    println!("  Time:       {:.1} ms", back_ms);
    println!("  Throughput: {:.1} MB/s\n", back_tp);

    // 3. HuggingFace tokenizers
    let hf = tokenizers::Tokenizer::from_file(TOKENIZER_PATH).expect("Failed to load HF tokenizer");

    // Warmup
    let _ = hf.encode(text.as_str(), false);

    let start = Instant::now();
    let mut hf_tokens = 0;
    for _ in 0..ITERATIONS {
        hf_tokens = hf.encode(text.as_str(), false).unwrap().get_ids().len();
    }
    let hf_time = start.elapsed();
    let hf_ms = hf_time.as_secs_f64() * 1000.0 / ITERATIONS as f64;
    let hf_tp = (bytes * ITERATIONS) as f64 / hf_time.as_secs_f64() / 1_000_000.0;

    println!("HuggingFace tokenizers (Rust)");
    println!("  Tokens:     {}", hf_tokens);
    println!("  Time:       {:.1} ms", hf_ms);
    println!("  Throughput: {:.1} MB/s\n", hf_tp);

    // 4. tiktoken-rs
    let tiktoken = tiktoken_rs::get_bpe_from_model("gpt2").expect("Failed to load tiktoken");

    // Warmup
    let _ = tiktoken.encode_ordinary(&text);

    let start = Instant::now();
    let mut tiktoken_tokens = 0;
    for _ in 0..ITERATIONS {
        tiktoken_tokens = tiktoken.encode_ordinary(&text).len();
    }
    let tiktoken_time = start.elapsed();
    let tiktoken_ms = tiktoken_time.as_secs_f64() * 1000.0 / ITERATIONS as f64;
    let tiktoken_tp = (bytes * ITERATIONS) as f64 / tiktoken_time.as_secs_f64() / 1_000_000.0;

    println!("tiktoken-rs");
    println!("  Tokens:     {}", tiktoken_tokens);
    println!("  Time:       {:.1} ms", tiktoken_ms);
    println!("  Throughput: {:.1} MB/s\n", tiktoken_tp);

    // 5. kitoken
    let kitoken = Kitoken::from_file(TOKENIZER_PATH).expect("Failed to load kitoken");

    // Warmup
    let _ = kitoken.encode(&text, true);

    let start = Instant::now();
    let mut kitoken_tokens = 0;
    for _ in 0..ITERATIONS {
        kitoken_tokens = kitoken.encode(&text, true).unwrap().len();
    }
    let kitoken_time = start.elapsed();
    let kitoken_ms = kitoken_time.as_secs_f64() * 1000.0 / ITERATIONS as f64;
    let kitoken_tp = (bytes * ITERATIONS) as f64 / kitoken_time.as_secs_f64() / 1_000_000.0;

    println!("kitoken");
    println!("  Tokens:     {}", kitoken_tokens);
    println!("  Time:       {:.1} ms", kitoken_ms);
    println!("  Throughput: {:.1} MB/s\n", kitoken_tp);

    // Summary
    println!("======================");
    println!("Summary (Sequential Throughput)");
    println!("======================\n");

    let results = [
        ("tokie Simple", simple_tp, simple_tokens),
        ("tokie Backtracking", back_tp, back_tokens),
        ("HuggingFace tokenizers", hf_tp, hf_tokens),
        ("tiktoken-rs", tiktoken_tp, tiktoken_tokens),
        ("kitoken", kitoken_tp, kitoken_tokens),
    ];

    // Sort by throughput descending
    let mut sorted = results.to_vec();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let fastest = sorted[0].1;
    println!("| Library | Throughput | vs Fastest | Tokens |");
    println!("|---------|------------|------------|--------|");
    for (name, tp, tokens) in &sorted {
        let ratio = fastest / tp;
        if (*tp - fastest).abs() < 0.1 {
            println!("| {} | {:.1} MB/s | (fastest) | {} |", name, tp, tokens);
        } else {
            println!("| {} | {:.1} MB/s | {:.1}x slower | {} |", name, tp, ratio, tokens);
        }
    }

    // Token count verification
    println!("\nToken count verification:");
    if simple_tokens == hf_tokens && simple_tokens == tiktoken_tokens
       && simple_tokens == back_tokens && simple_tokens == kitoken_tokens {
        println!("  All tokenizers produce {} tokens", simple_tokens);
    } else {
        println!("  simple: {}, backtrack: {}, HF: {}, tiktoken: {}, kitoken: {}",
                 simple_tokens, back_tokens, hf_tokens, tiktoken_tokens, kitoken_tokens);
    }
}

fn decode_bytelevel(s: &str) -> Vec<u8> {
    static NON_PRINTABLE: [u8; 68] = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138,
        139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156,
        157, 158, 159, 160, 173,
    ];
    let mut bytes = Vec::with_capacity(s.len());
    for c in s.chars() {
        let code = c as u32;
        let b = if code >= 256 && code < 256 + NON_PRINTABLE.len() as u32 {
            NON_PRINTABLE[(code - 256) as usize]
        } else if code <= 255 {
            code as u8
        } else {
            bytes.extend(c.to_string().as_bytes());
            continue;
        };
        bytes.push(b);
    }
    bytes
}
