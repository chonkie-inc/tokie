//! Fast pretokenizers for BPE tokenizers.
//!
//! Each pretokenizer is a zero-allocation, single-pass iterator over text pieces.
//!
//! # Example
//!
//! ```
//! use pretokie::Gpt2;
//!
//! let pieces: Vec<&str> = Gpt2::new("Hello world").collect();
//! assert_eq!(pieces, vec!["Hello", " world"]);
//! ```

mod core;
mod configs;
mod impls;
pub mod util;

pub use core::iter::Core;
pub use configs::{Gpt2Config, Cl100kConfig, O200kConfig, VoyageConfig, SmolLMConfig, DeepSeekConfig, QwenConfig};

pub type Gpt2<'a> = Core<'a, Gpt2Config>;
pub type Cl100k<'a> = Core<'a, Cl100kConfig>;
pub type O200k<'a> = Core<'a, O200kConfig>;
pub type Voyage<'a> = Core<'a, VoyageConfig>;
pub type SmolLM<'a> = Core<'a, SmolLMConfig>;
pub type DeepSeek<'a> = Core<'a, DeepSeekConfig>;
pub type Qwen<'a> = Core<'a, QwenConfig>;

pub use impls::bert::Bert;
#[cfg(feature = "regex")]
pub mod regex {
    //! Regex-based pretokenizer (requires `regex` feature).
    pub use crate::impls::regex::{Regex, RegexIter};
}
#[cfg(feature = "regex")]
pub use impls::regex::Regex;
