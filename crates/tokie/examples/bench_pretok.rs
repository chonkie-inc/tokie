//! Benchmark pretokenizer performance

use std::time::Instant;
use tokie::pretok::Pretok;

fn main() {
    let iterations = 20;

    let text = std::fs::read_to_string("benches/data/war_and_peace.txt")
        .expect("Need benches/data/war_and_peace.txt");

    println!("=== Pretokenizer Benchmark ({:.2} MB) ===\n", text.len() as f64 / 1_000_000.0);

    // GPT-2
    let start = Instant::now();
    let mut count = 0;
    for _ in 0..iterations {
        count = Pretok::GPT2.split(&text).count();
    }
    let time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
    let tp = text.len() as f64 / (time / 1000.0) / 1_000_000.0;
    println!("GPT2:   {:.2} ms ({:.0} MB/s) [{} pieces]", time, tp, count);

    // CL100K
    let start = Instant::now();
    for _ in 0..iterations {
        count = Pretok::CL100K.split(&text).count();
    }
    let time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
    let tp = text.len() as f64 / (time / 1000.0) / 1_000_000.0;
    println!("CL100K: {:.2} ms ({:.0} MB/s) [{} pieces]", time, tp, count);

    // O200K
    let start = Instant::now();
    for _ in 0..iterations {
        count = Pretok::O200K.split(&text).count();
    }
    let time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
    let tp = text.len() as f64 / (time / 1000.0) / 1_000_000.0;
    println!("O200K:  {:.2} ms ({:.0} MB/s) [{} pieces]", time, tp, count);

    // BERT
    let start = Instant::now();
    for _ in 0..iterations {
        count = Pretok::BERT.split(&text).count();
    }
    let time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
    let tp = text.len() as f64 / (time / 1000.0) / 1_000_000.0;
    println!("BERT:   {:.2} ms ({:.0} MB/s) [{} pieces]", time, tp, count);

    // Collect to Vec (measures allocation overhead)
    println!("\n=== With Vec collection ===\n");

    let start = Instant::now();
    let mut pieces = Vec::new();
    for _ in 0..iterations {
        pieces = Pretok::GPT2.split(&text).collect();
    }
    let time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
    let tp = text.len() as f64 / (time / 1000.0) / 1_000_000.0;
    println!("GPT2 (Vec):   {:.2} ms ({:.0} MB/s) [{} pieces]", time, tp, pieces.len());

    let start = Instant::now();
    for _ in 0..iterations {
        pieces = Pretok::BERT.split(&text).collect();
    }
    let time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
    let tp = text.len() as f64 / (time / 1000.0) / 1_000_000.0;
    println!("BERT (Vec):   {:.2} ms ({:.0} MB/s) [{} pieces]", time, tp, pieces.len());
}
