//! Fast pre-tokenization for BPE tokenizers.
//!
//! This module provides a configurable pretokenizer that achieves ~400 MiB/s
//! throughput while maintaining full Unicode support.
//!
//! # Example
//!
//! ```
//! use tokie::pretok::Pretok;
//!
//! // Use predefined configs
//! let pieces: Vec<&str> = Pretok::GPT2.split("Hello world").collect();
//! assert_eq!(pieces, vec!["Hello", " world"]);
//!
//! // CL100K chunks numbers into 1-3 digits
//! let pieces: Vec<&str> = Pretok::CL100K.split("12345").collect();
//! assert_eq!(pieces, vec!["123", "45"]);
//!
//! // O200K splits CamelCase and attaches contractions
//! let pieces: Vec<&str> = Pretok::O200K.split("don't").collect();
//! assert_eq!(pieces, vec!["don't"]);
//! ```

mod lexer;
mod regex;

pub use lexer::{ContractionPosition, LetterPrefix, LetterScanning, Pretok, PretokIter};
pub use regex::RegexPretok;

/// Type of pretokenizer for serialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum PretokType {
    None = 0,
    Gpt2 = 1,
    Cl100k = 2,
    O200k = 3,
    Bert = 4,
    Voyage = 5,
}

impl PretokType {
    /// Get the corresponding `Pretok` config for this type.
    pub fn to_pretok(self) -> Option<Pretok> {
        match self {
            PretokType::None => None,
            PretokType::Gpt2 => Some(Pretok::GPT2),
            PretokType::Cl100k => Some(Pretok::CL100K),
            PretokType::O200k => Some(Pretok::O200K),
            PretokType::Bert => Some(Pretok::BERT),
            PretokType::Voyage => Some(Pretok::VOYAGE),
        }
    }
}
