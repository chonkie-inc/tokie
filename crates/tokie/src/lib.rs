//! tokie - Fast BPE tokenizer using Aho-Corasick automata
//!
//! This crate implements Byte Pair Encoding (BPE) tokenization using the
//! algorithm from GitHub's rust-gems, which uses Aho-Corasick automata for
//! efficient suffix matching combined with compatibility checking.
//!
//! # Quick Start
//!
//! ```ignore
//! use tokie::Tokenizer;
//!
//! // Load from HuggingFace tokenizer.json
//! let tokenizer = Tokenizer::from_json("tokenizer.json")?;
//!
//! // Encode text (without special tokens)
//! let tokens = tokenizer.encode("Hello, world!", false);
//!
//! // Encode with special tokens (for model input)
//! let tokens_with_special = tokenizer.encode("Hello, world!", true);
//!
//! // Decode back
//! let text = tokenizer.decode(&tokens).unwrap();
//!
//! // Save/load binary format (fast)
//! tokenizer.to_file("model.tkz")?;
//! let tokenizer = Tokenizer::from_file("model.tkz")?;
//! ```
//!
//! # Loading from HuggingFace Hub
//!
//! Enable the `hf` feature to load tokenizers directly from HuggingFace:
//!
//! ```toml
//! tokie = { version = "0.1", features = ["hf"] }
//! ```
//!
//! ```ignore
//! use tokie::Tokenizer;
//!
//! let tokenizer = Tokenizer::from_pretrained("gpt2")?;
//! let tokenizer = Tokenizer::from_pretrained("meta-llama/Llama-3.2-8B")?;
//! ```
//!
//! # Architecture
//!
//! - [`Tokenizer`] - High-level API combining pre-tokenization + BPE encoding + decoding
//! - [`encoder`] - BPE encoders (backtracking for tiktoken, heap for LLaMA)
//! - [`Decoder`] - Token ID to bytes decoder (can be shared across encoder types)
//! - [`pretok`] - Fast pretokenizers (GPT-2: 566 MiB/s, cl100k, o200k)

mod decoder;
pub mod diff;
pub mod encoder;
pub mod hf;
#[cfg(feature = "hf")]
mod hub;
pub mod normalizer;
mod postprocessor;
pub mod pretok;
mod serde;
mod tokenizer;
mod types;

pub use encoder::{BacktrackingBytePairEncoder, BytePairEncoder, EncodeIter, Encoder, EncoderIter, EncoderType};
pub use decoder::Decoder;
pub use hf::JsonLoadError;
#[cfg(feature = "hf")]
pub use hub::{FromPretrainedOptions, HubError};
pub use normalizer::{bert_uncased_normalize, clean_text, fnr, metaspace_normalize, strip_accents, FnrFinder, Normalizer};
pub use postprocessor::PostProcessor;
pub use pretok::{Pretok, PretokIter, PretokType, RegexPretok};
pub use serde::SerdeError;
pub use tokenizer::{EncodingPair, TokenCount, Tokenizer, TokenizeIter};
pub use types::TokenId;

// Backward compatibility aliases
#[doc(hidden)]
#[deprecated(since = "0.2.0", note = "Use PretokType instead")]
pub type PretokenizerType = PretokType;
