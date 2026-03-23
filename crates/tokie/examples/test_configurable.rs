use std::fs;
use tokie::pretok::{Pretok, RegexPretok};

fn main() {
    let text = fs::read_to_string("benches/data/war_and_peace.txt")
        .expect("Failed to read file");

    println!("Testing Pretok configs against regex implementations\n");

    // Test GPT-2
    let gpt2_regex = RegexPretok::gpt2();
    let gpt2_config = Pretok::GPT2;

    let regex_pieces: Vec<&str> = gpt2_regex.split(&text).collect();
    let config_pieces: Vec<&str> = gpt2_config.split(&text).collect();

    println!("GPT-2:");
    println!("  Regex: {} pieces", regex_pieces.len());
    println!("  Pretok: {} pieces", config_pieces.len());

    if regex_pieces.len() == config_pieces.len() {
        let diffs: Vec<_> = regex_pieces.iter().zip(config_pieces.iter())
            .enumerate()
            .filter(|(_, (h, c))| h != c)
            .take(5)
            .collect();

        if diffs.is_empty() {
            println!("  ✓ All pieces match!");
        } else {
            println!("  ✗ {} differences found:", diffs.len());
            for (i, (h, c)) in &diffs {
                println!("    [{i}] regex={h:?} config={c:?}");
            }
        }
    } else {
        println!("  ✗ Piece count mismatch!");
        find_first_diff(&regex_pieces, &config_pieces);
    }

    println!();

    // Test CL100K
    let cl100k_regex = RegexPretok::cl100k();
    let cl100k_config = Pretok::CL100K;

    let regex_pieces: Vec<&str> = cl100k_regex.split(&text).collect();
    let config_pieces: Vec<&str> = cl100k_config.split(&text).collect();

    println!("CL100K:");
    println!("  Regex: {} pieces", regex_pieces.len());
    println!("  Pretok: {} pieces", config_pieces.len());

    if regex_pieces.len() == config_pieces.len() {
        let diffs: Vec<_> = regex_pieces.iter().zip(config_pieces.iter())
            .enumerate()
            .filter(|(_, (h, c))| h != c)
            .take(5)
            .collect();

        if diffs.is_empty() {
            println!("  ✓ All pieces match!");
        } else {
            println!("  ✗ {} differences found:", diffs.len());
            for (i, (h, c)) in &diffs {
                println!("    [{i}] regex={h:?} config={c:?}");
            }
        }
    } else {
        println!("  ✗ Piece count mismatch!");
        find_first_diff(&regex_pieces, &config_pieces);
    }

    println!();

    // Test O200K
    let o200k_regex = RegexPretok::o200k();
    let o200k_config = Pretok::O200K;

    let regex_pieces: Vec<&str> = o200k_regex.split(&text).collect();
    let config_pieces: Vec<&str> = o200k_config.split(&text).collect();

    println!("O200K:");
    println!("  Regex: {} pieces", regex_pieces.len());
    println!("  Pretok: {} pieces", config_pieces.len());

    if regex_pieces.len() == config_pieces.len() {
        let diffs: Vec<_> = regex_pieces.iter().zip(config_pieces.iter())
            .enumerate()
            .filter(|(_, (h, c))| h != c)
            .take(5)
            .collect();

        if diffs.is_empty() {
            println!("  ✓ All pieces match!");
        } else {
            println!("  ✗ {} differences found:", diffs.len());
            for (i, (h, c)) in &diffs {
                println!("    [{i}] regex={h:?} config={c:?}");
            }
        }
    } else {
        println!("  ✗ Piece count mismatch!");
        find_first_diff(&regex_pieces, &config_pieces);
    }
}

fn find_first_diff(hard: &[&str], config: &[&str]) {
    let min_len = hard.len().min(config.len());
    for i in 0..min_len {
        if hard[i] != config[i] {
            println!("  First diff at index {}:", i);
            println!("    Hard: {:?}", hard[i]);
            println!("    Config: {:?}", config[i]);
            let start = i.saturating_sub(2);
            let end = (i + 3).min(min_len);
            println!("  Context (hard):");
            for j in start..end {
                println!("    [{j}] {:?}", hard[j]);
            }
            println!("  Context (config):");
            for j in start..end {
                println!("    [{j}] {:?}", config[j]);
            }
            return;
        }
    }
    println!("  All compared pieces match, but lengths differ");
    println!("    Hard has {} more pieces", hard.len() as i64 - config.len() as i64);
}
