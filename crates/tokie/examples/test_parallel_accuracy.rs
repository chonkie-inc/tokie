//! Test accuracy of parallel pretokenization approaches.
//!
//! Compares token counts from:
//! 1. Sequential (correct baseline)
//! 2. Parallel with newline-only splits
//! 3. Parallel with space+newline splits

use memchunk::chunk;
use std::fs;
use std::thread;
use tokie::{Pretok, Tokenizer};

const TEXT_PATH: &str = "benches/data/war_and_peace.txt";
const TOKENIZER_PATH: &str = "benches/data/gpt2_tokenizer.json";

fn main() {
    println!("Parallel Pretokenization Accuracy Test");
    println!("======================================\n");

    let tokenizer = Tokenizer::from_json(TOKENIZER_PATH).expect("Failed to load tokenizer");
    let num_cpus = thread::available_parallelism().map(|p| p.get()).unwrap_or(1);

    println!("Testing with {} CPU cores\n", num_cpus);

    // Test cases: various edge cases
    let test_cases = vec![
        ("Simple", "Hello world"),
        ("Multiple spaces", "Hello  world"),
        ("Triple spaces", "Hello   world"),
        ("Trailing space", "Hello world "),
        ("Leading space", " Hello world"),
        ("Multiple trailing", "Hello world   "),
        ("Newline", "Hello\nworld"),
        ("Double newline", "Hello\n\nworld"),
        ("Mixed whitespace", "Hello \n world"),
        ("Tabs", "Hello\tworld"),
        ("Tab and space", "Hello\t world"),
        ("Contractions", "don't won't can't"),
        ("Numbers", "12345 67890"),
        ("CL100K numbers", "123456789"),
        ("Punctuation", "Hello, world! How's it going?"),
        ("Code-like", "func_name(arg1, arg2)"),
        ("Lots of spaces", "a b c d e f g h i j"),
        ("Long words", "internationalization globalization"),
    ];

    println!("=== GPT-2 Pretokenizer (small tests) ===\n");
    let pretok = Pretok::GPT2;

    for (name, text) in &test_cases {
        test_pretok_accuracy(name, text, &pretok, num_cpus);
    }

    println!("\n=== CL100K Pretokenizer (small tests) ===\n");
    let cl100k = Pretok::CL100K;
    for (name, text) in &test_cases {
        test_pretok_accuracy(name, text, &cl100k, num_cpus);
    }

    println!("\n=== O200K Pretokenizer (small tests) ===\n");
    let o200k = Pretok::O200K;
    for (name, text) in &test_cases {
        test_pretok_accuracy(name, text, &o200k, num_cpus);
    }

    // Full tokenization accuracy (small tests)
    println!("\n=== Full Encode Accuracy (small tests) ===\n");

    for (name, text) in &test_cases {
        test_encode_accuracy(name, text, &tokenizer, num_cpus);
    }

    // Test on real text - just counts, no collecting
    println!("\n=== War and Peace (counts only) ===\n");
    let war_and_peace = fs::read_to_string(TEXT_PATH).expect("Failed to read file");
    println!("Loaded {} bytes\n", war_and_peace.len());

    test_pretok_counts("GPT-2", &war_and_peace, &Pretok::GPT2, num_cpus);
    test_pretok_counts("CL100K", &war_and_peace, &Pretok::CL100K, num_cpus);
    test_pretok_counts("O200K", &war_and_peace, &Pretok::O200K, num_cpus);

    println!("\n=== War and Peace Full Encode ===\n");
    test_encode_counts("War and Peace", &war_and_peace, &tokenizer, num_cpus);
}

fn test_pretok_accuracy(name: &str, text: &str, pretok: &Pretok, num_cpus: usize) {
    let sequential: Vec<String> = pretok.split(text).map(|s| s.to_string()).collect();
    let parallel_newline = parallel_pretok(text, pretok, num_cpus, b"\n");
    let parallel_space = parallel_pretok(text, pretok, num_cpus, b" \n");

    let seq_count = sequential.len();
    let nl_count = parallel_newline.len();
    let sp_count = parallel_space.len();

    let nl_match = sequential == parallel_newline;
    let sp_match = sequential == parallel_space;

    let nl_status = if nl_match { "✓" } else { "✗" };
    let sp_status = if sp_match { "✓" } else { "✗" };

    println!(
        "{:25} seq={:4}  newline={:4} {}  space={:4} {}",
        name, seq_count, nl_count, nl_status, sp_count, sp_status
    );

    // Show differences if any
    if !nl_match {
        show_diff_owned("newline", &sequential, &parallel_newline);
    }
    if !sp_match {
        show_diff_owned("space", &sequential, &parallel_space);
    }
}

fn test_pretok_counts(name: &str, text: &str, pretok: &Pretok, num_cpus: usize) {
    let sequential = pretok.split(text).count();
    let parallel_newline = parallel_pretok_count(text, pretok, num_cpus, b"\n");
    let parallel_space = parallel_pretok_count(text, pretok, num_cpus, b" \n");

    let nl_match = sequential == parallel_newline;
    let sp_match = sequential == parallel_space;

    let nl_status = if nl_match { "✓" } else { "✗" };
    let sp_status = if sp_match { "✓" } else { "✗" };

    println!(
        "{:10} seq={:>8}  newline={:>8} {} (diff: {:+})  space={:>8} {} (diff: {:+})",
        name,
        sequential,
        parallel_newline,
        nl_status,
        parallel_newline as i64 - sequential as i64,
        parallel_space,
        sp_status,
        parallel_space as i64 - sequential as i64
    );
}

fn test_encode_accuracy(name: &str, text: &str, tokenizer: &Tokenizer, num_cpus: usize) {
    let sequential = tokenizer.encode(text, false);
    let parallel_newline = chunked_encode(text, tokenizer, num_cpus, b"\n");
    let parallel_space = chunked_encode(text, tokenizer, num_cpus, b" \n");

    let seq_count = sequential.len();
    let nl_count = parallel_newline.len();
    let sp_count = parallel_space.len();

    let nl_match = sequential == parallel_newline;
    let sp_match = sequential == parallel_space;

    let nl_status = if nl_match { "✓" } else { "✗" };
    let sp_status = if sp_match { "✓" } else { "✗" };

    println!(
        "{:25} seq={:4}  newline={:4} {}  space={:4} {}",
        name, seq_count, nl_count, nl_status, sp_count, sp_status
    );
}

fn test_encode_counts(name: &str, text: &str, tokenizer: &Tokenizer, num_cpus: usize) {
    let sequential = tokenizer.encode(text, false).len();
    let parallel_newline = chunked_encode_count(text, tokenizer, num_cpus, b"\n");
    let parallel_space = chunked_encode_count(text, tokenizer, num_cpus, b" \n");

    let nl_match = sequential == parallel_newline;
    let sp_match = sequential == parallel_space;

    let nl_status = if nl_match { "✓" } else { "✗" };
    let sp_status = if sp_match { "✓" } else { "✗" };

    println!(
        "{:15} seq={:>8}  newline={:>8} {} (diff: {:+})  space={:>8} {} (diff: {:+})",
        name,
        sequential,
        parallel_newline,
        nl_status,
        parallel_newline as i64 - sequential as i64,
        parallel_space,
        sp_status,
        parallel_space as i64 - sequential as i64
    );
}

/// Parallel pretokenization returning pieces as owned Strings (for small texts only).
fn parallel_pretok(
    text: &str,
    pretok: &Pretok,
    _num_cpus: usize,
    delimiters: &[u8],
) -> Vec<String> {
    let bytes = text.as_bytes();

    // For small texts, use target of 100 bytes to test splitting behavior
    let target_size = 100.min(bytes.len() / 2);

    if target_size == 0 || bytes.len() < 10 {
        return pretok.split(text).map(|s| s.to_string()).collect();
    }

    let chunks: Vec<&[u8]> = chunk(bytes)
        .size(target_size)
        .delimiters(delimiters)
        .prefix()
        .collect();

    if chunks.len() <= 1 {
        return pretok.split(text).map(|s| s.to_string()).collect();
    }

    // For small texts, we can safely collect
    let mut result = Vec::new();
    for chunk_bytes in chunks {
        let chunk_str = unsafe { std::str::from_utf8_unchecked(chunk_bytes) };
        result.extend(pretok.split(chunk_str).map(|s| s.to_string()));
    }
    result
}

/// Parallel pretokenization returning count only (memory efficient).
fn parallel_pretok_count(text: &str, pretok: &Pretok, num_cpus: usize, delimiters: &[u8]) -> usize {
    let bytes = text.as_bytes();
    let target_size = bytes.len() / num_cpus;

    if target_size == 0 {
        return pretok.split(text).count();
    }

    let chunks: Vec<&[u8]> = chunk(bytes)
        .size(target_size)
        .delimiters(delimiters)
        .prefix()
        .collect();

    if chunks.len() == 1 {
        return pretok.split(text).count();
    }

    thread::scope(|s| {
        let handles: Vec<_> = chunks
            .iter()
            .map(|chunk_bytes| {
                s.spawn(|| {
                    let chunk_str = unsafe { std::str::from_utf8_unchecked(chunk_bytes) };
                    pretok.split(chunk_str).count()
                })
            })
            .collect();

        handles.into_iter().map(|h| h.join().unwrap()).sum()
    })
}

/// Chunked parallel encode (for small texts).
fn chunked_encode(text: &str, tokenizer: &Tokenizer, _num_cpus: usize, delimiters: &[u8]) -> Vec<u32> {
    let bytes = text.as_bytes();

    // For small texts, use target of 100 bytes
    let target_size = 100.min(bytes.len() / 2);

    if target_size == 0 || bytes.len() < 10 {
        return tokenizer.encode(text, false);
    }

    let chunks: Vec<&[u8]> = chunk(bytes)
        .size(target_size)
        .delimiters(delimiters)
        .prefix()
        .collect();

    if chunks.len() <= 1 {
        return tokenizer.encode(text, false);
    }

    let pretok = tokenizer.pretokenizer().unwrap();
    let encoder = tokenizer.encoder();

    let mut result = Vec::new();
    for chunk_bytes in chunks {
        let chunk_str = unsafe { std::str::from_utf8_unchecked(chunk_bytes) };
        for piece in pretok.split(chunk_str) {
            result.extend(encoder.encode(piece.as_bytes()));
        }
    }
    result
}

/// Chunked parallel encode returning count only.
fn chunked_encode_count(
    text: &str,
    tokenizer: &Tokenizer,
    num_cpus: usize,
    delimiters: &[u8],
) -> usize {
    let bytes = text.as_bytes();
    let target_size = bytes.len() / num_cpus;

    if target_size == 0 {
        return tokenizer.encode(text, false).len();
    }

    let chunks: Vec<&[u8]> = chunk(bytes)
        .size(target_size)
        .delimiters(delimiters)
        .prefix()
        .collect();

    if chunks.len() == 1 {
        return tokenizer.encode(text, false).len();
    }

    let pretok = tokenizer.pretokenizer().unwrap();
    let encoder = tokenizer.encoder();

    thread::scope(|s| {
        let handles: Vec<_> = chunks
            .iter()
            .map(|chunk_bytes| {
                s.spawn(|| {
                    let chunk_str = unsafe { std::str::from_utf8_unchecked(chunk_bytes) };
                    pretok
                        .split(chunk_str)
                        .map(|piece| encoder.encode(piece.as_bytes()).len())
                        .sum::<usize>()
                })
            })
            .collect();

        handles.into_iter().map(|h| h.join().unwrap()).sum()
    })
}

fn show_diff_owned(label: &str, expected: &[String], actual: &[String]) {
    let min_len = expected.len().min(actual.len());
    for i in 0..min_len {
        if expected[i] != actual[i] {
            println!("    {} first diff at [{}]: {:?} vs {:?}", label, i, expected[i], actual[i]);
            return;
        }
    }
    if expected.len() != actual.len() {
        println!("    {} length diff: {} vs {}", label, expected.len(), actual.len());
    }
}
