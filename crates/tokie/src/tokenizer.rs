//! High-level Tokenizer that combines pre-tokenization with BPE encoding.

use std::cell::RefCell;
use std::cmp::Ordering;
use std::path::Path;
use std::thread;

use memchunk::chunk;

use crate::encoder::{Encoder, EncoderIter, EncoderType};
use crate::decoder::Decoder;
use crate::hf::{self, JsonLoadError};
use crate::normalizer::Normalizer;
use crate::padding::{Encoding, PaddingParams, TruncationParams, pad_batch, pad_encoding, truncate_ids, truncate_pair};
use crate::postprocessor::PostProcessor;
use crate::pretok::{Pretok, PretokType};
use crate::types::TokenId;

/// Backward-compatible alias for [`Encoding`].
pub type EncodingPair = Encoding;

/// High-level tokenizer combining pre-tokenization, BPE encoding, and decoding.
///
/// This is the main interface for tokenizing text. It handles:
/// 1. Pre-tokenization (splitting text using regex patterns)
/// 2. BPE encoding (converting each piece to token IDs)
/// 3. Decoding (converting token IDs back to text)
///
/// # Example
/// ```ignore
/// use tokie::Tokenizer;
///
/// let tokenizer = Tokenizer::from_json("tokenizer.json")?;
///
/// let tokens = tokenizer.encode("Hello, world!", false);
/// let text = tokenizer.decode(&tokens);
/// ```
pub struct Tokenizer {
    /// The underlying BPE encoder.
    encoder: Encoder,
    /// The decoder for converting tokens back to bytes.
    decoder: Decoder,
    /// Optional pre-tokenizer for splitting text before BPE.
    pretokenizer: Option<Pretok>,
    /// The type of pretokenizer (for serialization).
    pretokenizer_type: PretokType,
    /// Text normalizer (lowercase, NFC, etc.).
    normalizer: Normalizer,
    /// Post-processor for adding special tokens.
    post_processor: PostProcessor,
    /// Pad token ID (from vocabulary, persisted in .tkz).
    pad_token_id: Option<TokenId>,
    /// Padding configuration (runtime, not serialized).
    padding: Option<PaddingParams>,
    /// Truncation configuration (runtime, not serialized).
    truncation: Option<TruncationParams>,
}

impl Tokenizer {
    /// Create a new tokenizer with all components.
    pub fn new(
        encoder: Encoder,
        decoder: Decoder,
        pretokenizer_type: PretokType,
        normalizer: Normalizer,
        post_processor: PostProcessor,
    ) -> Self {
        let pretokenizer = pretokenizer_type.to_pretok();
        Self {
            encoder,
            decoder,
            pretokenizer,
            pretokenizer_type,
            normalizer,
            post_processor,
            pad_token_id: None,
            padding: None,
            truncation: None,
        }
    }

    /// Get the pretokenizer type.
    pub fn pretokenizer_type(&self) -> PretokType {
        self.pretokenizer_type
    }

    /// Get the normalizer.
    pub fn normalizer(&self) -> Normalizer {
        self.normalizer
    }

    /// Get the post-processor.
    pub fn post_processor(&self) -> &PostProcessor {
        &self.post_processor
    }

    /// Get the encoder type.
    pub fn encoder_type(&self) -> EncoderType {
        self.encoder.encoder_type()
    }

    /// Load a tokenizer from a HuggingFace tokenizer.json file.
    ///
    /// The pretokenizer type is auto-detected from the JSON.
    /// Uses Backtracking encoder by default.
    ///
    /// # Example
    /// ```ignore
    /// use tokie::Tokenizer;
    ///
    /// let tokenizer = Tokenizer::from_json("tokenizer.json")?;
    /// let tokens = tokenizer.encode("Hello, world!", false);
    /// ```
    pub fn from_json(path: impl AsRef<Path>) -> Result<Self, JsonLoadError> {
        hf::from_json(path)
    }

    /// Load a tokenizer from a HuggingFace tokenizer.json file with specific encoder type.
    ///
    /// # Example
    /// ```ignore
    /// use tokie::{Tokenizer, EncoderType};
    ///
    /// let tokenizer = Tokenizer::from_json_with_encoder("tokenizer.json", EncoderType::Simple)?;
    /// ```
    pub fn from_json_with_encoder(
        path: impl AsRef<Path>,
        encoder_type: EncoderType,
    ) -> Result<Self, JsonLoadError> {
        hf::from_json_with_encoder(path, encoder_type)
    }

    /// Get a reference to the underlying BPE encoder.
    pub fn encoder(&self) -> &Encoder {
        &self.encoder
    }

    /// Get a reference to the decoder.
    pub fn decoder(&self) -> &Decoder {
        &self.decoder
    }

    /// Get a reference to the pre-tokenizer, if any.
    pub fn pretokenizer(&self) -> Option<&Pretok> {
        self.pretokenizer.as_ref()
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.decoder.vocab_size()
    }

    /// Minimum text size (in bytes) to trigger chunked parallel encoding.
    /// Below this threshold, sequential encoding is used to avoid overhead.
    const PARALLEL_CHUNK_THRESHOLD: usize = 10_000;

    // --- Padding / truncation configuration ---

    /// Enable padding. Applied during `encode_batch()` and `encode_pair_batch()`.
    ///
    /// For single `encode()` with `PaddingStrategy::Fixed`, padding is also applied.
    pub fn enable_padding(&mut self, params: PaddingParams) -> &mut Self {
        self.padding = Some(params);
        self
    }

    /// Enable truncation. Applied during `encode()`, `encode_batch()`, and `encode_pair()`.
    pub fn enable_truncation(&mut self, params: TruncationParams) -> &mut Self {
        self.truncation = Some(params);
        self
    }

    /// Disable padding.
    pub fn no_padding(&mut self) -> &mut Self {
        self.padding = None;
        self
    }

    /// Disable truncation.
    pub fn no_truncation(&mut self) -> &mut Self {
        self.truncation = None;
        self
    }

    /// Set the pad token ID.
    pub fn set_pad_token_id(&mut self, id: TokenId) -> &mut Self {
        self.pad_token_id = Some(id);
        self
    }

    /// Get the pad token ID, if set.
    pub fn pad_token_id(&self) -> Option<TokenId> {
        self.pad_token_id
    }

    /// Get current padding configuration.
    pub fn padding(&self) -> Option<&PaddingParams> {
        self.padding.as_ref()
    }

    /// Get current truncation configuration.
    pub fn truncation(&self) -> Option<&TruncationParams> {
        self.truncation.as_ref()
    }

    // --- Encoding ---

    /// Encode text into an [`Encoding`] with token IDs, attention mask, and type IDs.
    ///
    /// If truncation is configured, the output is truncated to `max_length`.
    /// If padding is configured with `Fixed` strategy, the output is padded.
    ///
    /// # Example
    /// ```ignore
    /// let enc = tokenizer.encode("Hello, world!", true);
    /// println!("ids: {:?}", enc.ids);
    /// println!("attention_mask: {:?}", enc.attention_mask);
    /// ```
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Encoding {
        let mut tokens = self.encode_raw(text, false);

        // Apply truncation to content tokens (before special tokens)
        if let Some(ref trunc) = self.truncation {
            let special = if add_special_tokens {
                self.post_processor.num_special_tokens_single()
            } else {
                0
            };
            let max_content = trunc.max_length.saturating_sub(special);
            truncate_ids(&mut tokens, max_content, trunc.direction);
        }

        let ids = if add_special_tokens {
            self.post_processor.process(&tokens)
        } else {
            tokens
        };

        let mut encoding = Encoding::from_ids(ids);

        // Apply fixed padding for single encode
        if let Some(ref pad) = self.padding {
            if let crate::padding::PaddingStrategy::Fixed(n) = pad.strategy {
                pad_encoding(&mut encoding, n, pad);
            }
        }

        encoding
    }

    /// Encode a pair of texts and return an [`Encoding`] with IDs, attention mask, and type IDs.
    ///
    /// If truncation is configured, sequences are truncated according to the truncation strategy.
    ///
    /// # Example
    /// ```ignore
    /// let enc = tokenizer.encode_pair("What is Berlin?", "Berlin is the capital of Germany.", true);
    /// // enc.ids:            [CLS] query tokens [SEP] doc tokens [SEP]
    /// // enc.type_ids:       0     0...         0     1...       1
    /// // enc.attention_mask: 1     1...         1     1...       1
    /// ```
    pub fn encode_pair(&self, text_a: &str, text_b: &str, add_special_tokens: bool) -> Encoding {
        let mut tokens_a = self.encode_raw(text_a, false);
        let mut tokens_b = self.encode_raw(text_b, false);

        // Apply truncation to content tokens (before special tokens)
        if let Some(ref trunc) = self.truncation {
            let special = if add_special_tokens {
                self.post_processor.num_special_tokens_pair()
            } else {
                0
            };
            let max_content = trunc.max_length.saturating_sub(special);
            truncate_pair(&mut tokens_a, &mut tokens_b, max_content, trunc.strategy, trunc.direction);
        }

        let (ids, type_ids) = if add_special_tokens {
            self.post_processor.process_pair(&tokens_a, &tokens_b)
        } else {
            let mut ids = Vec::with_capacity(tokens_a.len() + tokens_b.len());
            ids.extend_from_slice(&tokens_a);
            ids.extend_from_slice(&tokens_b);
            let mut type_ids = vec![0u8; tokens_a.len()];
            type_ids.extend(vec![1u8; tokens_b.len()]);
            (ids, type_ids)
        };

        Encoding::from_pair(ids, type_ids)
    }

    /// Encode text to raw token IDs (internal helper, no Encoding wrapper).
    ///
    /// This is the core encoding path used by all public methods.
    fn encode_raw(&self, text: &str, add_special_tokens: bool) -> Vec<TokenId> {
        let tokens = if text.len() >= Self::PARALLEL_CHUNK_THRESHOLD {
            self.encode_parallel(text, self.pretokenizer.as_ref())
        } else {
            let normalized = self.normalizer.normalize(text);
            match &self.pretokenizer {
                Some(pretok) => self.encode_sequential(normalized.as_ref(), pretok),
                None => self.encoder.encode(normalized.as_ref().as_bytes()),
            }
        };

        if add_special_tokens {
            self.post_processor.process(&tokens)
        } else {
            tokens
        }
    }

    /// Sequential encoding: pretokenize then BPE encode each piece.
    #[inline]
    fn encode_sequential(&self, text: &str, pretok: &Pretok) -> Vec<TokenId> {
        pretok
            .split(text)
            .flat_map(|piece| self.encoder.encode(piece.as_bytes()))
            .collect()
    }

    /// Parallel encoding: split text into chunks at whitespace boundaries,
    /// then each thread does normalization + pretokenization + encoding on its chunk.
    ///
    /// This is more efficient than normalizing the entire text first because:
    /// 1. Many chunks will be pure ASCII and can skip NFD normalization
    /// 2. Normalization work is parallelized across threads
    /// 3. Better cache locality (each thread works on its own data)
    ///
    /// For SentencePiece tokenizers (Metaspace normalizer), we split at metaspace
    /// boundaries instead of spaces, since the normalizer replaces spaces with ▁.
    ///
    /// When `pretok` is None (e.g., Unigram models), the encoder is called directly
    /// on normalized chunks without pretokenization.
    fn encode_parallel(&self, text: &str, pretok: Option<&Pretok>) -> Vec<TokenId> {
        let bytes = text.as_bytes();
        let num_cpus = thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);

        let target_size = bytes.len() / num_cpus;

        // Split at space boundaries in the raw text.
        // Each thread will normalize its chunk (which may add metaspaces for SentencePiece).
        // We use prefix() to keep the space with the following chunk (matches how
        // SentencePiece prepends ▁ to words).
        let chunks: Vec<&[u8]> = chunk(bytes)
            .size(target_size)
            .delimiters(b" ")
            .prefix()
            .collect();

        if chunks.len() <= 1 {
            // Single chunk: normalize and encode sequentially
            let normalized = self.normalizer.normalize(text);
            return match pretok {
                Some(p) => self.encode_sequential(normalized.as_ref(), p),
                None => self.encoder.encode(normalized.as_ref().as_bytes()),
            };
        }

        // Each thread: normalize + optionally pretokenize + encode its chunk
        let encoder = &self.encoder;
        let normalizer = &self.normalizer;
        let results: Vec<Vec<TokenId>> = thread::scope(|s| {
            chunks
                .iter()
                .map(|chunk_bytes| {
                    s.spawn(move || {
                        // SAFETY: Input was valid UTF-8, and we only split at ASCII
                        // whitespace (space/newline), preserving UTF-8 validity.
                        let chunk_str = unsafe { std::str::from_utf8_unchecked(chunk_bytes) };

                        // Normalize this chunk
                        let normalized = normalizer.normalize(chunk_str);

                        match pretok {
                            Some(p) => {
                                // With pretokenizer: split then encode each piece
                                p.split(normalized.as_ref())
                                    .flat_map(|piece| encoder.encode(piece.as_bytes()))
                                    .collect()
                            }
                            None => {
                                // Without pretokenizer: encode directly (Unigram, etc.)
                                encoder.encode(normalized.as_ref().as_bytes())
                            }
                        }
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|h| h.join().unwrap())
                .collect()
        });

        // Flatten with pre-allocated capacity
        let total: usize = results.iter().map(|v| v.len()).sum();
        let mut output = Vec::with_capacity(total);
        for chunk_tokens in results {
            output.extend(chunk_tokens);
        }
        output
    }

    /// Encode bytes directly without pre-tokenization.
    ///
    /// This bypasses the pre-tokenizer and encodes raw bytes.
    /// Useful when you've already done your own text processing.
    pub fn encode_bytes(&self, bytes: &[u8]) -> Vec<TokenId> {
        self.encoder.encode(bytes)
    }

    /// Returns a streaming iterator over encoded tokens.
    ///
    /// Note: When a pre-tokenizer is configured, this currently collects
    /// all tokens (not truly streaming across pre-token boundaries).
    /// For true streaming without pre-tokenization, use `encode_bytes_iter`.
    pub fn encode_iter<'a>(&'a self, text: &'a str) -> TokenizeIter<'a> {
        TokenizeIter::new(self, text)
    }

    /// Returns a streaming iterator over encoded tokens from bytes.
    ///
    /// Bypasses pre-tokenization for true streaming.
    pub fn encode_bytes_iter<'a>(&'a self, bytes: &'a [u8]) -> EncoderIter<'a> {
        self.encoder.encode_iter(bytes)
    }

    /// Decode token IDs back to a string.
    ///
    /// Returns `None` if the decoded bytes are not valid UTF-8.
    pub fn decode(&self, tokens: &[TokenId]) -> Option<String> {
        self.decoder.decode_to_string(tokens)
    }

    /// Decode token IDs back to bytes.
    pub fn decode_bytes(&self, tokens: &[TokenId]) -> Vec<u8> {
        self.decoder.decode(tokens)
    }

    /// Get the byte sequence for a token.
    pub fn token_to_bytes(&self, token: TokenId) -> &[u8] {
        self.decoder.token_to_bytes(token)
    }

    /// Encode multiple texts in parallel, returning [`Encoding`] for each.
    ///
    /// If truncation is configured, each text is truncated.
    /// If padding is configured, all encodings are padded to the same length.
    ///
    /// # Example
    /// ```ignore
    /// let texts = vec!["Hello, world!", "How are you?", "Goodbye!"];
    /// let encodings = tokenizer.encode_batch(&texts, true);
    /// assert_eq!(encodings.len(), 3);
    /// ```
    pub fn encode_batch(&self, texts: &[&str], add_special_tokens: bool) -> Vec<Encoding> {
        let num_cpus = thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);

        let mut encodings = if texts.len() > num_cpus {
            self.encode_batch_distributed(texts, add_special_tokens)
        } else {
            self.encode_batch_sequential(texts, add_special_tokens)
        };

        // Apply batch padding after all texts are encoded
        if let Some(ref pad) = self.padding {
            pad_batch(&mut encodings, pad);
        }

        encodings
    }

    /// Distribute texts evenly across threads for encoding.
    fn encode_batch_distributed(&self, texts: &[&str], add_special_tokens: bool) -> Vec<Encoding> {
        if texts.is_empty() {
            return Vec::new();
        }

        let num_cpus = thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);

        if num_cpus == 1 || texts.len() == 1 {
            return texts.iter().map(|t| self.encode(t, add_special_tokens)).collect();
        }

        let chunk_size = (texts.len() + num_cpus - 1) / num_cpus;

        thread::scope(|s| {
            let handles: Vec<_> = texts
                .chunks(chunk_size)
                .map(|text_chunk| {
                    s.spawn(|| {
                        text_chunk
                            .iter()
                            .map(|t| self.encode(t, add_special_tokens))
                            .collect::<Vec<_>>()
                    })
                })
                .collect();

            handles
                .into_iter()
                .flat_map(|h: std::thread::ScopedJoinHandle<'_, Vec<Encoding>>| h.join().unwrap())
                .collect()
        })
    }

    /// Sequential encoding loop, each encode() may parallelize internally.
    fn encode_batch_sequential(&self, texts: &[&str], add_special_tokens: bool) -> Vec<Encoding> {
        texts.iter().map(|t| self.encode(t, add_special_tokens)).collect()
    }

    /// Count tokens for multiple texts in parallel.
    ///
    /// # Example
    /// ```ignore
    /// let texts = vec!["Hello, world!", "How are you?"];
    /// let counts = tokenizer.count_tokens_batch(&texts);
    /// assert_eq!(counts.len(), 2);
    /// ```
    pub fn count_tokens_batch(&self, texts: &[&str]) -> Vec<usize> {
        if texts.is_empty() {
            return Vec::new();
        }

        let num_cpus = thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);

        if num_cpus == 1 || texts.len() == 1 {
            return texts.iter().map(|t| self.count_tokens(t)).collect();
        }

        let chunk_size = (texts.len() + num_cpus - 1) / num_cpus;

        thread::scope(|s| {
            let handles: Vec<_> = texts
                .chunks(chunk_size)
                .map(|text_chunk| {
                    s.spawn(|| {
                        text_chunk
                            .iter()
                            .map(|t| self.count_tokens(t))
                            .collect::<Vec<_>>()
                    })
                })
                .collect();

            handles
                .into_iter()
                .flat_map(|h| h.join().unwrap())
                .collect()
        })
    }

    /// Count tokens without storing them.
    ///
    /// Uses the same parallelized encoding path as `encode()`.
    /// Does not include special tokens in the count.
    pub fn count_tokens(&self, text: &str) -> usize {
        self.encode_raw(text, false).len()
    }

    /// Returns a lazy token count that supports comparison operators.
    ///
    /// Uses early termination - stops counting as soon as the comparison result is known.
    ///
    /// # Example
    /// ```ignore
    /// if tokenizer.token_count(text) > 8192 {
    ///     println!("text exceeds context window");
    /// }
    /// ```
    pub fn token_count<'a>(&'a self, text: &'a str) -> TokenCount<'a> {
        TokenCount {
            iter: RefCell::new(Some(self.encoder.encode_iter(text.as_bytes()))),
        }
    }
}

/// Lazy token count that supports comparison with `usize`.
///
/// Enables idiomatic comparisons like `tokenizer.token_count(text) > 8192`.
/// The count is computed lazily and stops early when the result is determined.
///
/// Note: Each `TokenCount` can only be compared once (the iterator is consumed).
pub struct TokenCount<'a> {
    iter: RefCell<Option<EncoderIter<'a>>>,
}

impl PartialEq<usize> for TokenCount<'_> {
    fn eq(&self, other: &usize) -> bool {
        self.partial_cmp(other) == Some(Ordering::Equal)
    }
}

impl PartialOrd<usize> for TokenCount<'_> {
    fn partial_cmp(&self, limit: &usize) -> Option<Ordering> {
        let iter = self.iter.borrow_mut().take()?;
        let count = iter.take(*limit + 1).count();
        Some(count.cmp(limit))
    }
}

/// Iterator over tokens from the high-level Tokenizer.
pub struct TokenizeIter<'a> {
    tokenizer: &'a Tokenizer,
    // Current state for pretokenized case
    pretokens: Option<Box<dyn Iterator<Item = &'a str> + 'a>>,
    current_encoder_iter: Option<EncoderIter<'a>>,
    // For non-pretokenized case
    bytes_iter: Option<EncoderIter<'a>>,
}

impl<'a> TokenizeIter<'a> {
    fn new(tokenizer: &'a Tokenizer, text: &'a str) -> Self {
        if tokenizer.pretokenizer.is_some() {
            let pretokens = tokenizer.pretokenizer.as_ref().unwrap().split(text);
            Self {
                tokenizer,
                pretokens: Some(Box::new(pretokens)),
                current_encoder_iter: None,
                bytes_iter: None,
            }
        } else {
            Self {
                tokenizer,
                pretokens: None,
                current_encoder_iter: None,
                bytes_iter: Some(tokenizer.encoder.encode_iter(text.as_bytes())),
            }
        }
    }
}

impl<'a> Iterator for TokenizeIter<'a> {
    type Item = TokenId;

    fn next(&mut self) -> Option<TokenId> {
        // Non-pretokenized case: just use the bytes iterator
        if let Some(ref mut iter) = self.bytes_iter {
            return iter.next();
        }

        // Pretokenized case: iterate through pretokens
        loop {
            // Try to get next token from current encoder iterator
            if let Some(ref mut encoder_iter) = self.current_encoder_iter {
                if let Some(token) = encoder_iter.next() {
                    return Some(token);
                }
            }

            // Current encoder exhausted, get next pretoken
            if let Some(ref mut pretokens) = self.pretokens {
                if let Some(piece) = pretokens.next() {
                    self.current_encoder_iter =
                        Some(self.tokenizer.encoder.encode_iter(piece.as_bytes()));
                    continue;
                }
            }

            // No more pretokens
            return None;
        }
    }
}

impl std::iter::FusedIterator for TokenizeIter<'_> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::BacktrackingBytePairEncoder;
    use crate::padding::{PaddingStrategy, PaddingDirection, TruncationStrategy, TruncationDirection};

    fn make_test_tokenizer() -> Tokenizer {
        // Simple encoder: a=0, b=1, c=2, space=3
        // Merges: a+b->ab(256)
        let base_tokens: Vec<Vec<u8>> = (0u8..=255).map(|b| vec![b]).collect();
        let merges = vec![(b'a' as u32, b'b' as u32)]; // a+b -> ab
        let (encoder, token_bytes) = BacktrackingBytePairEncoder::from_merges(&merges, &base_tokens);
        let decoder = Decoder::new(token_bytes);
        Tokenizer::new(Encoder::Backtracking(encoder), decoder, PretokType::None, Normalizer::None, PostProcessor::None)
    }

    fn make_test_tokenizer_with_pretokenizer() -> Tokenizer {
        let base_tokens: Vec<Vec<u8>> = (0u8..=255).map(|b| vec![b]).collect();
        let merges = vec![(b'a' as u32, b'b' as u32)]; // a+b -> ab
        let (encoder, token_bytes) = BacktrackingBytePairEncoder::from_merges(&merges, &base_tokens);
        let decoder = Decoder::new(token_bytes);
        Tokenizer::new(Encoder::Backtracking(encoder), decoder, PretokType::Gpt2, Normalizer::None, PostProcessor::None)
    }

    fn make_bert_tokenizer() -> Tokenizer {
        let base_tokens: Vec<Vec<u8>> = (0u8..=255).map(|b| vec![b]).collect();
        let merges = vec![(b'a' as u32, b'b' as u32)];
        let (encoder, token_bytes) = BacktrackingBytePairEncoder::from_merges(&merges, &base_tokens);
        let decoder = Decoder::new(token_bytes);
        Tokenizer::new(Encoder::Backtracking(encoder), decoder, PretokType::None, Normalizer::None, PostProcessor::bert(101, 102))
    }

    #[test]
    fn test_tokenizer_no_pretokenizer() {
        let tokenizer = make_test_tokenizer();
        let enc = tokenizer.encode("abc", false);
        // 'a'+'b' merges to token 256, then 'c' is token 99
        assert_eq!(enc.ids.len(), 2);
    }

    #[test]
    fn test_encode_returns_encoding() {
        let tokenizer = make_test_tokenizer();
        let enc = tokenizer.encode("abc", false);
        assert_eq!(enc.ids.len(), enc.attention_mask.len());
        assert_eq!(enc.ids.len(), enc.type_ids.len());
        assert!(enc.attention_mask.iter().all(|&m| m == 1));
        assert!(enc.type_ids.iter().all(|&t| t == 0));
    }

    #[test]
    fn test_tokenizer_with_pretokenizer() {
        let tokenizer = make_test_tokenizer_with_pretokenizer();
        let enc = tokenizer.encode("Hello world", false);
        assert!(!enc.ids.is_empty());
        let decoded = tokenizer.decode(&enc.ids).unwrap();
        assert_eq!(decoded, "Hello world");
    }

    #[test]
    fn test_count_tokens() {
        let tokenizer = make_test_tokenizer_with_pretokenizer();
        let text = "Hello world";
        let count = tokenizer.count_tokens(text);
        let enc = tokenizer.encode(text, false);
        assert_eq!(count, enc.ids.len());
    }

    #[test]
    fn test_token_count_comparisons() {
        let tokenizer = make_test_tokenizer_with_pretokenizer();
        let text = "Hello world test";
        let total = tokenizer.count_tokens(text);
        assert!(tokenizer.token_count(text) > total - 1);
        assert!(!(tokenizer.token_count(text) > total));
        assert!(tokenizer.token_count(text) < total + 1);
        assert!(!(tokenizer.token_count(text) < total));
        assert!(tokenizer.token_count(text) == total);
    }

    #[test]
    fn test_encode_iter() {
        let tokenizer = make_test_tokenizer_with_pretokenizer();
        let text = "Hello world";
        let tokens: Vec<_> = tokenizer.encode_iter(text).collect();
        let expected = tokenizer.encode_raw(text, false);
        assert_eq!(tokens, expected);
    }

    #[test]
    fn test_decode_bytes() {
        let tokenizer = make_test_tokenizer();
        let text = b"abc";
        let tokens = tokenizer.encode_bytes(text);
        let decoded = tokenizer.decode_bytes(&tokens);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_encode_batch_empty() {
        let tokenizer = make_test_tokenizer();
        let result = tokenizer.encode_batch(&[], false);
        assert!(result.is_empty());
    }

    #[test]
    fn test_encode_batch_single() {
        let tokenizer = make_test_tokenizer_with_pretokenizer();
        let single = tokenizer.encode("Hello world", false);
        let batch = tokenizer.encode_batch(&["Hello world"], false);
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0], single);
    }

    #[test]
    fn test_encode_batch_multiple() {
        let tokenizer = make_test_tokenizer_with_pretokenizer();
        let texts = vec!["Hello world", "abc def", "test"];
        let batch = tokenizer.encode_batch(&texts, false);
        assert_eq!(batch.len(), 3);
        for (i, text) in texts.iter().enumerate() {
            assert_eq!(batch[i], tokenizer.encode(text, false));
        }
    }

    #[test]
    fn test_encode_batch_preserves_order() {
        let tokenizer = make_test_tokenizer_with_pretokenizer();
        let texts: Vec<&str> = (0..20).map(|i| match i % 4 {
            0 => "alpha",
            1 => "beta gamma",
            2 => "delta epsilon zeta",
            _ => "x",
        }).collect();
        let batch = tokenizer.encode_batch(&texts, false);
        assert_eq!(batch.len(), texts.len());
        for (i, text) in texts.iter().enumerate() {
            assert_eq!(batch[i], tokenizer.encode(text, false));
        }
    }

    #[test]
    fn test_encode_batch_with_special_tokens() {
        let tokenizer = make_test_tokenizer_with_pretokenizer();
        let texts = vec!["Hello", "world"];
        let batch_with = tokenizer.encode_batch(&texts, true);
        let batch_without = tokenizer.encode_batch(&texts, false);
        assert_eq!(batch_with, batch_without);
    }

    #[test]
    fn test_encode_batch_approaches_match() {
        let tokenizer = make_test_tokenizer_with_pretokenizer();
        let texts: Vec<&str> = vec!["Hello world", "abc def ghi", "test one two three", "x"];
        let distributed = tokenizer.encode_batch_distributed(&texts, false);
        let sequential = tokenizer.encode_batch_sequential(&texts, false);
        assert_eq!(distributed, sequential);
    }

    #[test]
    fn test_count_tokens_batch() {
        let tokenizer = make_test_tokenizer_with_pretokenizer();
        let texts = vec!["Hello world", "abc", "test one two"];
        let counts = tokenizer.count_tokens_batch(&texts);
        assert_eq!(counts.len(), 3);
        for (i, text) in texts.iter().enumerate() {
            assert_eq!(counts[i], tokenizer.count_tokens(text));
        }
    }

    #[test]
    fn test_count_tokens_batch_empty() {
        let tokenizer = make_test_tokenizer();
        let result = tokenizer.count_tokens_batch(&[]);
        assert!(result.is_empty());
    }

    // --- Truncation tests ---

    #[test]
    fn test_encode_with_truncation() {
        let mut tokenizer = make_test_tokenizer();
        // "abcde" without merges would be 5 tokens, with a+b merge = 4 tokens
        // Set max_length to 3
        tokenizer.enable_truncation(TruncationParams {
            max_length: 3,
            strategy: TruncationStrategy::LongestFirst,
            direction: TruncationDirection::Right,
            stride: 0,
        });
        let enc = tokenizer.encode("abcde", false);
        assert!(enc.ids.len() <= 3);
    }

    #[test]
    fn test_encode_truncation_preserves_special_tokens() {
        let mut tokenizer = make_bert_tokenizer();
        // BERT: [CLS] tokens [SEP] = 2 special tokens
        tokenizer.enable_truncation(TruncationParams {
            max_length: 4,
            strategy: TruncationStrategy::LongestFirst,
            direction: TruncationDirection::Right,
            stride: 0,
        });
        let enc = tokenizer.encode("abcde", true);
        assert!(enc.ids.len() <= 4);
        // First token should be CLS (101), last should be SEP (102)
        assert_eq!(enc.ids[0], 101);
        assert_eq!(*enc.ids.last().unwrap(), 102);
    }

    #[test]
    fn test_encode_pair_with_truncation() {
        let mut tokenizer = make_bert_tokenizer();
        // BERT pair: [CLS] A [SEP] B [SEP] = 3 special tokens
        tokenizer.enable_truncation(TruncationParams {
            max_length: 7,
            strategy: TruncationStrategy::LongestFirst,
            direction: TruncationDirection::Right,
            stride: 0,
        });
        let enc = tokenizer.encode_pair("abcde", "fghij", true);
        assert!(enc.ids.len() <= 7);
        assert_eq!(enc.ids[0], 101); // CLS
    }

    // --- Padding tests ---

    #[test]
    fn test_encode_batch_with_padding() {
        let mut tokenizer = make_test_tokenizer();
        tokenizer.enable_padding(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            pad_id: 0,
            ..Default::default()
        });
        let batch = tokenizer.encode_batch(&["ab", "abcde"], false);
        // All encodings should be same length (longest)
        assert_eq!(batch[0].ids.len(), batch[1].ids.len());
        // Shorter one should have 0s in attention_mask
        let short_mask = &batch[0].attention_mask;
        assert!(short_mask.iter().any(|&m| m == 0));
        // Longer one should have all 1s
        assert!(batch[1].attention_mask.iter().all(|&m| m == 1));
    }

    #[test]
    fn test_encode_with_fixed_padding() {
        let mut tokenizer = make_test_tokenizer();
        tokenizer.enable_padding(PaddingParams {
            strategy: PaddingStrategy::Fixed(10),
            pad_id: 0,
            ..Default::default()
        });
        let enc = tokenizer.encode("ab", false);
        assert_eq!(enc.ids.len(), 10);
        assert_eq!(enc.attention_mask.iter().filter(|&&m| m == 0).count(), 10 - 1); // "ab" merges to 1 token
    }

    #[test]
    fn test_encode_batch_with_fixed_padding() {
        let mut tokenizer = make_test_tokenizer();
        tokenizer.enable_padding(PaddingParams {
            strategy: PaddingStrategy::Fixed(8),
            pad_id: 0,
            ..Default::default()
        });
        let batch = tokenizer.encode_batch(&["ab", "cd", "e"], false);
        assert!(batch.iter().all(|e| e.ids.len() == 8));
    }

    #[test]
    fn test_encode_batch_left_padding() {
        let mut tokenizer = make_test_tokenizer();
        tokenizer.enable_padding(PaddingParams {
            strategy: PaddingStrategy::Fixed(5),
            direction: PaddingDirection::Left,
            pad_id: 0,
            ..Default::default()
        });
        let enc = tokenizer.encode("ab", false);
        // "ab" merges to 1 token, left padded to 5
        assert_eq!(enc.ids.len(), 5);
        assert_eq!(enc.attention_mask[0], 0); // left side is padding
        assert_eq!(*enc.attention_mask.last().unwrap(), 1); // right side is real
    }

    #[test]
    fn test_no_padding_no_truncation_defaults() {
        let tokenizer = make_test_tokenizer();
        assert!(tokenizer.padding().is_none());
        assert!(tokenizer.truncation().is_none());
        assert!(tokenizer.pad_token_id().is_none());
    }

    #[test]
    fn test_config_methods() {
        let mut tokenizer = make_test_tokenizer();
        tokenizer.enable_padding(PaddingParams::default());
        assert!(tokenizer.padding().is_some());
        tokenizer.no_padding();
        assert!(tokenizer.padding().is_none());

        tokenizer.enable_truncation(TruncationParams {
            max_length: 512,
            strategy: TruncationStrategy::LongestFirst,
            direction: TruncationDirection::Right,
            stride: 0,
        });
        assert!(tokenizer.truncation().is_some());
        tokenizer.no_truncation();
        assert!(tokenizer.truncation().is_none());

        tokenizer.set_pad_token_id(0);
        assert_eq!(tokenizer.pad_token_id(), Some(0));
    }
}
