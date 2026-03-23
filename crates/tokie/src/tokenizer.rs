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
use crate::postprocessor::PostProcessor;
use crate::pretok::{Pretok, PretokType};
use crate::types::TokenId;

/// Result of encoding a pair of texts (e.g. for cross-encoder models).
#[derive(Debug, Clone)]
pub struct EncodingPair {
    /// Token IDs for the combined pair.
    pub ids: Vec<TokenId>,
    /// Attention mask (1 for real tokens, used for padding in batches).
    pub attention_mask: Vec<u8>,
    /// Token type IDs (0 for first sequence, 1 for second sequence).
    pub type_ids: Vec<u8>,
}

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

    /// Encode text into token IDs.
    ///
    /// Text is first normalized (if a normalizer is configured), then split
    /// by the pre-tokenizer (if configured), then each piece is BPE-encoded.
    /// For texts >= 10KB, uses chunked parallel encoding with per-thread normalization.
    ///
    /// If `add_special_tokens` is true, post-processing is applied (e.g., adding
    /// `[CLS]` and `[SEP]` for BERT, or `<|begin_of_text|>` for LLaMA 3).
    ///
    /// # Example
    /// ```ignore
    /// // Without special tokens (default for most use cases)
    /// let tokens = tokenizer.encode("Hello, world!", false);
    ///
    /// // With special tokens (for model input)
    /// let tokens = tokenizer.encode("Hello, world!", true);
    /// ```
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Vec<TokenId> {
        let tokens = if text.len() >= Self::PARALLEL_CHUNK_THRESHOLD {
            // Parallel path for large texts
            self.encode_parallel(text, self.pretokenizer.as_ref())
        } else {
            // Sequential path
            let normalized = self.normalizer.normalize(text);
            match &self.pretokenizer {
                Some(pretok) => self.encode_sequential(normalized.as_ref(), pretok),
                None => self.encoder.encode(normalized.as_ref().as_bytes()),
            }
        };

        // Apply post-processing if requested
        if add_special_tokens {
            self.post_processor.process(&tokens)
        } else {
            tokens
        }
    }

    /// Encode a pair of texts with special tokens and return IDs, attention mask, and type IDs.
    ///
    /// This is the equivalent of HuggingFace's `tokenizer.encode((text_a, text_b), add_special_tokens)`.
    /// Used for cross-encoder models that take sentence pairs as input.
    ///
    /// # Example
    /// ```ignore
    /// let pair = tokenizer.encode_pair("What is Berlin?", "Berlin is the capital of Germany.", true);
    /// // pair.ids:            [CLS] query tokens [SEP] doc tokens [SEP]
    /// // pair.type_ids:       0     0...         0     1...       1
    /// // pair.attention_mask: 1     1...         1     1...       1
    /// ```
    pub fn encode_pair(&self, text_a: &str, text_b: &str, add_special_tokens: bool) -> EncodingPair {
        let tokens_a = self.encode(text_a, false);
        let tokens_b = self.encode(text_b, false);

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

        let attention_mask = vec![1u8; ids.len()];
        EncodingPair { ids, attention_mask, type_ids }
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

    /// Count tokens without storing them.
    ///
    /// Uses the same parallelized encoding path as `encode()`.
    /// Does not include special tokens in the count.
    pub fn count_tokens(&self, text: &str) -> usize {
        self.encode(text, false).len()
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

    #[test]
    fn test_tokenizer_no_pretokenizer() {
        let tokenizer = make_test_tokenizer();

        let tokens = tokenizer.encode("abc", false);
        // 'a'+'b' merges to token 256, then 'c' is token 99
        assert_eq!(tokens.len(), 2);
    }

    #[test]
    fn test_tokenizer_with_pretokenizer() {
        let tokenizer = make_test_tokenizer_with_pretokenizer();

        // "Hello world" gets pre-tokenized to ["Hello", " world"]
        let tokens = tokenizer.encode("Hello world", false);
        assert!(!tokens.is_empty());

        // Verify decode roundtrip
        let decoded = tokenizer.decode(&tokens).unwrap();
        assert_eq!(decoded, "Hello world");
    }

    #[test]
    fn test_count_tokens() {
        let tokenizer = make_test_tokenizer_with_pretokenizer();

        let text = "Hello world";
        let count = tokenizer.count_tokens(text);
        let tokens = tokenizer.encode(text, false);
        assert_eq!(count, tokens.len());
    }

    #[test]
    fn test_token_count_comparisons() {
        let tokenizer = make_test_tokenizer_with_pretokenizer();

        let text = "Hello world test";
        let total = tokenizer.count_tokens(text);

        // Greater than
        assert!(tokenizer.token_count(text) > total - 1);
        assert!(!(tokenizer.token_count(text) > total));

        // Less than
        assert!(tokenizer.token_count(text) < total + 1);
        assert!(!(tokenizer.token_count(text) < total));

        // Equal
        assert!(tokenizer.token_count(text) == total);
        assert!(!(tokenizer.token_count(text) == total + 1));

        // Greater than or equal
        assert!(tokenizer.token_count(text) >= total);
        assert!(tokenizer.token_count(text) >= total - 1);
        assert!(!(tokenizer.token_count(text) >= total + 1));

        // Less than or equal
        assert!(tokenizer.token_count(text) <= total);
        assert!(tokenizer.token_count(text) <= total + 1);
        assert!(!(tokenizer.token_count(text) <= total - 1));
    }

    #[test]
    fn test_encode_iter() {
        let tokenizer = make_test_tokenizer_with_pretokenizer();

        let text = "Hello world";
        let tokens: Vec<_> = tokenizer.encode_iter(text).collect();
        let expected = tokenizer.encode(text, false);
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
}
