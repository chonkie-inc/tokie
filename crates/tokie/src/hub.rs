//! HuggingFace Hub integration for loading tokenizers.
//!
//! This module is only available when the `hf` feature is enabled:
//! ```toml
//! tokie = { version = "0.1", features = ["hf"] }
//! ```
//!
//! # Example
//! ```ignore
//! use tokie::Tokenizer;
//!
//! // Load from HuggingFace Hub
//! let tokenizer = Tokenizer::from_pretrained("gpt2")?;
//! let tokenizer = Tokenizer::from_pretrained("meta-llama/Llama-3.2-8B")?;
//!
//! // With options
//! let tokenizer = Tokenizer::from_pretrained_with_options(
//!     "gpt2",
//!     FromPretrainedOptions::default().revision("main"),
//! )?;
//! ```

use std::path::PathBuf;

use hf_hub::Repo;

use crate::hf::JsonLoadError;
use crate::serde::SerdeError;
use crate::Tokenizer;

/// Error type for `from_pretrained` operations.
#[derive(Debug)]
pub enum HubError {
    /// Failed to initialize the HuggingFace Hub API.
    ApiInit(hf_hub::api::sync::ApiError),
    /// Failed to download the tokenizer file.
    Download(hf_hub::api::sync::ApiError),
    /// Failed to load the tokenizer from JSON.
    Load(JsonLoadError),
    /// Failed to load the tokenizer from .tkz binary format.
    LoadBinary(SerdeError),
    /// The tokenizer.json file was not found in the repository.
    NotFound(String),
}

impl std::fmt::Display for HubError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HubError::ApiInit(e) => write!(f, "failed to initialize HuggingFace Hub API: {}", e),
            HubError::Download(e) => write!(f, "failed to download tokenizer: {}", e),
            HubError::Load(e) => write!(f, "failed to load tokenizer: {}", e),
            HubError::LoadBinary(e) => write!(f, "failed to load .tkz tokenizer: {}", e),
            HubError::NotFound(repo) => {
                write!(f, "tokenizer not found in repository '{}'", repo)
            }
        }
    }
}

impl std::error::Error for HubError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            HubError::ApiInit(e) => Some(e),
            HubError::Download(e) => Some(e),
            HubError::Load(e) => Some(e),
            HubError::LoadBinary(e) => Some(e),
            HubError::NotFound(_) => None,
        }
    }
}

impl From<JsonLoadError> for HubError {
    fn from(e: JsonLoadError) -> Self {
        HubError::Load(e)
    }
}

/// Options for `from_pretrained`.
#[derive(Debug, Clone, Default)]
pub struct FromPretrainedOptions {
    /// Git revision (branch, tag, or commit hash). Defaults to "main".
    pub revision: Option<String>,
    /// Custom cache directory. Defaults to HuggingFace cache (~/.cache/huggingface/hub).
    pub cache_dir: Option<PathBuf>,
    /// HuggingFace API token for private repositories.
    pub token: Option<String>,
}

impl FromPretrainedOptions {
    /// Set the git revision (branch, tag, or commit hash).
    pub fn revision(mut self, revision: impl Into<String>) -> Self {
        self.revision = Some(revision.into());
        self
    }

    /// Set a custom cache directory.
    pub fn cache_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.cache_dir = Some(path.into());
        self
    }

    /// Set the HuggingFace API token for private repositories.
    pub fn token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }
}

impl Tokenizer {
    /// Load a tokenizer from HuggingFace Hub.
    ///
    /// This first tries to download a `tokenizer.tkz` file (tokie's compact binary
    /// format) for faster loading. If not found, falls back to `tokenizer.json`.
    /// Files are cached locally for subsequent loads.
    ///
    /// # Arguments
    /// * `repo_id` - Repository ID (e.g., "gpt2", "meta-llama/Llama-3.2-8B")
    ///
    /// # Example
    /// ```ignore
    /// use tokie::Tokenizer;
    ///
    /// let tokenizer = Tokenizer::from_pretrained("gpt2")?;
    /// let tokens = tokenizer.encode("Hello, world!", false);
    /// ```
    pub fn from_pretrained(repo_id: impl AsRef<str>) -> Result<Self, HubError> {
        Self::from_pretrained_with_options(repo_id, FromPretrainedOptions::default())
    }

    /// Load a tokenizer from HuggingFace Hub with custom options.
    ///
    /// # Arguments
    /// * `repo_id` - Repository ID (e.g., "gpt2", "meta-llama/Llama-3.2-8B")
    /// * `options` - Configuration options (revision, cache_dir, token)
    ///
    /// # Example
    /// ```ignore
    /// use tokie::{Tokenizer, FromPretrainedOptions};
    ///
    /// let tokenizer = Tokenizer::from_pretrained_with_options(
    ///     "gpt2",
    ///     FromPretrainedOptions::default()
    ///         .revision("main")
    ///         .token("hf_xxx"),
    /// )?;
    /// ```
    pub fn from_pretrained_with_options(
        repo_id: impl AsRef<str>,
        options: FromPretrainedOptions,
    ) -> Result<Self, HubError> {
        let repo_id = repo_id.as_ref();

        // Build the API client
        let mut api_builder = hf_hub::api::sync::ApiBuilder::new();

        if let Some(cache_dir) = options.cache_dir {
            api_builder = api_builder.with_cache_dir(cache_dir);
        }

        if let Some(token) = options.token {
            api_builder = api_builder.with_token(Some(token));
        }

        let api = api_builder.build().map_err(HubError::ApiInit)?;

        // Build the repo reference
        let repo = if let Some(revision) = options.revision {
            Repo::with_revision(repo_id.to_string(), hf_hub::RepoType::Model, revision)
        } else {
            Repo::model(repo_id.to_string())
        };

        let repo_api = api.repo(repo);

        // Try tokenizer.tkz first (faster to load, smaller to download)
        if let Ok(tkz_path) = repo_api.get("tokenizer.tkz") {
            return Self::from_file(tkz_path).map_err(HubError::LoadBinary);
        }

        // Fall back to tokenizer.json
        let tokenizer_path = repo_api.get("tokenizer.json").map_err(HubError::Download)?;
        Self::from_json(tokenizer_path).map_err(HubError::Load)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires network access
    fn test_from_pretrained_gpt2() {
        let tokenizer = Tokenizer::from_pretrained("gpt2").expect("Failed to load GPT-2");
        let tokens = tokenizer.encode("Hello, world!", false);
        assert!(!tokens.is_empty());

        // Verify it produces expected tokens for GPT-2
        let decoded = tokenizer.decode(&tokens).unwrap();
        assert_eq!(decoded, "Hello, world!");
    }

    #[test]
    #[ignore] // Requires network access
    fn test_from_pretrained_with_revision() {
        let tokenizer = Tokenizer::from_pretrained_with_options(
            "gpt2",
            FromPretrainedOptions::default().revision("main"),
        )
        .expect("Failed to load GPT-2");

        let tokens = tokenizer.encode("Test", false);
        assert!(!tokens.is_empty());
    }
}
