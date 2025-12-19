//! tokie - Fast BPE tokenizer using Aho-Corasick automata
//!
//! This crate implements Byte Pair Encoding (BPE) tokenization using the
//! algorithm from GitHub's rust-gems, which uses Aho-Corasick automata for
//! efficient suffix matching combined with compatibility checking.

mod compatibility;
mod tokenizer;
mod types;

pub use tokenizer::{BpeTokenizer, EncodeIter};
pub use types::TokenId;
