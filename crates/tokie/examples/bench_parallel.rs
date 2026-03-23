//! Benchmark parallel encoding with pretokenization
//!
//! Tests throughput of different encoders using the same parallel chunking
//! strategy as the Tokenizer: split at whitespace, each thread does
//! pretokenization + encoding on its chunk.

use std::thread;
use std::time::Instant;

use memchunk::chunk;

fn main() {
    let text_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("benches/data/war_and_peace.txt");
    let text = std::fs::read_to_string(&text_path).unwrap();

    // Load GPT-2 tokenizer
    let json_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("benches/data/gpt2_tokenizer.json");

    let json_str = std::fs::read_to_string(&json_path).unwrap();
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

    // Build encoders
    let (simple_enc, _) =
        tokie::encoder::BytePairEncoder::from_vocab_and_merges(&full_vocab, &merges, 256);
    let (back_enc, _) =
        tokie::encoder::BacktrackingBytePairEncoder::from_vocab_and_merges(&full_vocab, &merges, 256);

    let pretok = tokie::pretok::PretokType::Gpt2.to_pretok().unwrap();

    let num_cpus = thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);

    println!("=== Parallel BPE Encoding Benchmark ===");
    println!("Text: {:.2} MB ({} bytes)", text.len() as f64 / 1_000_000.0, text.len());
    println!("CPUs: {}", num_cpus);
    println!();

    // Warmup
    let _ = encode_parallel_simple(&text, &pretok, &simple_enc, num_cpus);
    let _ = encode_parallel_backtrack(&text, &pretok, &back_enc, num_cpus);

    // === Sequential benchmarks ===
    println!("=== Sequential (single-threaded) ===\n");

    let start = Instant::now();
    let simple_seq: Vec<u32> = pretok
        .split(&text)
        .flat_map(|piece| simple_enc.encode(piece.as_bytes()))
        .collect();
    let simple_seq_time = start.elapsed();

    let start = Instant::now();
    let back_seq: Vec<u32> = pretok
        .split(&text)
        .flat_map(|piece| back_enc.encode(piece.as_bytes()))
        .collect();
    let back_seq_time = start.elapsed();

    println!(
        "Simple (O(n²)):     {:>7.1} ms, {:>6.1} MB/s, {} tokens",
        simple_seq_time.as_secs_f64() * 1000.0,
        text.len() as f64 / simple_seq_time.as_secs_f64() / 1_000_000.0,
        simple_seq.len()
    );
    println!(
        "Backtracking:       {:>7.1} ms, {:>6.1} MB/s, {} tokens",
        back_seq_time.as_secs_f64() * 1000.0,
        text.len() as f64 / back_seq_time.as_secs_f64() / 1_000_000.0,
        back_seq.len()
    );

    // === Parallel benchmarks ===
    println!("\n=== Parallel ({} threads) ===\n", num_cpus);

    let start = Instant::now();
    let simple_par = encode_parallel_simple(&text, &pretok, &simple_enc, num_cpus);
    let simple_par_time = start.elapsed();

    let start = Instant::now();
    let back_par = encode_parallel_backtrack(&text, &pretok, &back_enc, num_cpus);
    let back_par_time = start.elapsed();

    println!(
        "Simple (O(n²)):     {:>7.1} ms, {:>6.1} MB/s, {} tokens",
        simple_par_time.as_secs_f64() * 1000.0,
        text.len() as f64 / simple_par_time.as_secs_f64() / 1_000_000.0,
        simple_par.len()
    );
    println!(
        "Backtracking:       {:>7.1} ms, {:>6.1} MB/s, {} tokens",
        back_par_time.as_secs_f64() * 1000.0,
        text.len() as f64 / back_par_time.as_secs_f64() / 1_000_000.0,
        back_par.len()
    );

    // === Speedups ===
    println!("\n=== Parallel Speedup ===\n");
    println!(
        "Simple:       {:.1}x ({:.1} MB/s → {:.1} MB/s)",
        simple_seq_time.as_secs_f64() / simple_par_time.as_secs_f64(),
        text.len() as f64 / simple_seq_time.as_secs_f64() / 1_000_000.0,
        text.len() as f64 / simple_par_time.as_secs_f64() / 1_000_000.0
    );
    println!(
        "Backtracking: {:.1}x ({:.1} MB/s → {:.1} MB/s)",
        back_seq_time.as_secs_f64() / back_par_time.as_secs_f64(),
        text.len() as f64 / back_seq_time.as_secs_f64() / 1_000_000.0,
        text.len() as f64 / back_par_time.as_secs_f64() / 1_000_000.0
    );

    // === Correctness ===
    println!("\n=== Correctness ===\n");
    if simple_seq == simple_par {
        println!("Simple: sequential == parallel");
    } else {
        println!("Simple: MISMATCH ({} vs {})", simple_seq.len(), simple_par.len());
    }
    if back_seq == back_par {
        println!("Backtracking: sequential == parallel");
    } else {
        println!("Backtracking: MISMATCH ({} vs {})", back_seq.len(), back_par.len());
    }
    if simple_seq == back_seq {
        println!("Simple == Backtracking (both correct)");
    } else {
        println!("Simple vs Backtracking: {} vs {} tokens", simple_seq.len(), back_seq.len());
    }
}

fn encode_parallel_simple(
    text: &str,
    pretok: &tokie::pretok::Pretok,
    encoder: &tokie::encoder::BytePairEncoder,
    num_cpus: usize,
) -> Vec<u32> {
    let bytes = text.as_bytes();
    let chunks: Vec<&[u8]> = chunk(bytes)
        .size(bytes.len() / num_cpus)
        .delimiters(b" ")
        .prefix()
        .collect();

    if chunks.len() <= 1 {
        return pretok
            .split(text)
            .flat_map(|piece| encoder.encode(piece.as_bytes()))
            .collect();
    }

    let results: Vec<Vec<u32>> = thread::scope(|s| {
        chunks
            .iter()
            .map(|chunk_bytes| {
                s.spawn(move || {
                    let chunk_str = unsafe { std::str::from_utf8_unchecked(chunk_bytes) };
                    pretok
                        .split(chunk_str)
                        .flat_map(|piece| encoder.encode(piece.as_bytes()))
                        .collect()
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect()
    });

    let total: usize = results.iter().map(|v| v.len()).sum();
    let mut output = Vec::with_capacity(total);
    for chunk_tokens in results {
        output.extend(chunk_tokens);
    }
    output
}

fn encode_parallel_backtrack(
    text: &str,
    pretok: &tokie::pretok::Pretok,
    encoder: &tokie::encoder::BacktrackingBytePairEncoder,
    num_cpus: usize,
) -> Vec<u32> {
    let bytes = text.as_bytes();
    let chunks: Vec<&[u8]> = chunk(bytes)
        .size(bytes.len() / num_cpus)
        .delimiters(b" ")
        .prefix()
        .collect();

    if chunks.len() <= 1 {
        return pretok
            .split(text)
            .flat_map(|piece| encoder.encode(piece.as_bytes()))
            .collect();
    }

    let results: Vec<Vec<u32>> = thread::scope(|s| {
        chunks
            .iter()
            .map(|chunk_bytes| {
                s.spawn(move || {
                    let chunk_str = unsafe { std::str::from_utf8_unchecked(chunk_bytes) };
                    pretok
                        .split(chunk_str)
                        .flat_map(|piece| encoder.encode(piece.as_bytes()))
                        .collect()
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect()
    });

    let total: usize = results.iter().map(|v| v.len()).sum();
    let mut output = Vec::with_capacity(total);
    for chunk_tokens in results {
        output.extend(chunk_tokens);
    }
    output
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
