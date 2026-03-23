//! Unigram encoder using Viterbi dynamic programming.
//!
//! Implements the Unigram language model tokenization algorithm as used by
//! SentencePiece Unigram models (T5, XLM-RoBERTa, ALBERT, mT5, etc.).
//!
//! Unlike BPE (deterministic merges), Unigram uses probabilistic tokenization:
//! - Each token has a **log probability score**
//! - Find segmentation that **maximizes total score** using Viterbi DP
//! - Time: O(n × L) where L = max token length
//!
//! Algorithm:
//! 1. Forward pass: for each position, find all tokens that end here and track best score
//! 2. Backward pass: reconstruct the optimal segmentation from backpointers

use daggrs::{DoubleArrayAhoCorasick, MatchKind, Trie};
use foldhash::HashMap as FoldHashMap;
use smallvec::SmallVec;

use crate::types::TokenId;

/// Default chunk size for chunked encoding.
/// 64KB chunks keep memory usage low (~5MB for DP arrays).
const DEFAULT_CHUNK_SIZE: usize = 64 * 1024; // 64KB chunks

/// Overlap size for boundary reconciliation (must be > max token length).
const OVERLAP_SIZE: usize = 256;

/// Unigram encoder using Viterbi dynamic programming.
#[derive(Clone)]
pub struct UnigramEncoder {
    /// Aho-Corasick automaton for efficient token matching.
    matcher: DoubleArrayAhoCorasick,

    /// Token scores (log probabilities), indexed by token ID.
    scores: Vec<f32>,

    /// Unknown token ID.
    unk_token: TokenId,

    /// Byte fallback: byte value -> token ID for <0xXX> tokens.
    /// u32::MAX means no byte fallback token exists.
    byte_tokens: [TokenId; 256],

    /// Token lengths in bytes.
    token_lengths: Vec<u16>,

    /// Vocabulary size.
    vocab_size: usize,

    /// Maps byte sequence -> token ID for early exit.
    token_cache: FoldHashMap<Vec<u8>, TokenId>,
}

/// Maximum token length to cache for early exit lookup.
const MAX_CACHED_TOKEN_LEN: usize = 16;

impl std::fmt::Debug for UnigramEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnigramEncoder")
            .field("vocab_size", &self.vocab_size)
            .field("unk_token", &self.unk_token)
            .finish()
    }
}

impl UnigramEncoder {
    /// Create encoder from vocabulary with scores.
    ///
    /// # Arguments
    /// * `vocab` - List of (token_id, token_bytes, score) tuples, sorted by ID
    /// * `unk_token` - Token ID to use for unknown sequences
    pub fn from_vocab_with_scores(
        vocab: &[(u32, Vec<u8>, f32)],
        unk_token: TokenId,
    ) -> (Self, Vec<Vec<u8>>) {
        let token_bytes: Vec<Vec<u8>> = vocab.iter().map(|(_, bytes, _)| bytes.clone()).collect();
        let scores: Vec<f32> = vocab.iter().map(|(_, _, score)| *score).collect();

        // Build byte fallback table (<0xXX> tokens)
        let mut byte_tokens = [u32::MAX; 256];
        for (id, bytes, _) in vocab {
            // <0xXX> tokens are 6 bytes: '<', '0', 'x', hex, hex, '>'
            if bytes.len() == 6 && bytes.starts_with(b"<0x") && bytes.ends_with(b">") {
                if let Ok(byte_val) = u8::from_str_radix(
                    std::str::from_utf8(&bytes[3..5]).unwrap_or(""),
                    16,
                ) {
                    if byte_tokens[byte_val as usize] == u32::MAX {
                        byte_tokens[byte_val as usize] = *id;
                    }
                }
            }
        }

        // Build Aho-Corasick matcher for all tokens
        // Use Overlapping to get ALL matches at each position (required for Viterbi)
        let mut trie = Trie::new();
        for (id, bytes, _) in vocab {
            if !bytes.is_empty() {
                trie.add(bytes, *id);
            }
        }
        trie.build(MatchKind::Overlapping);
        let matcher = trie.compile();

        let token_lengths: Vec<u16> = token_bytes.iter().map(|b| b.len() as u16).collect();

        // Build token_cache for early exit
        let mut token_cache = FoldHashMap::default();
        for (id, bytes, _) in vocab {
            if bytes.len() <= MAX_CACHED_TOKEN_LEN {
                token_cache.insert(bytes.clone(), *id);
            }
        }

        let encoder = Self {
            matcher,
            scores,
            unk_token,
            byte_tokens,
            token_lengths,
            vocab_size: vocab.len(),
            token_cache,
        };

        (encoder, token_bytes)
    }

    /// Create encoder from pre-built components (for deserialization).
    pub fn from_parts(
        matcher: DoubleArrayAhoCorasick,
        scores: Vec<f32>,
        unk_token: TokenId,
        byte_tokens: [TokenId; 256],
        token_lengths: Vec<u16>,
        token_bytes: &[Vec<u8>],
    ) -> Self {
        let vocab_size = scores.len();

        // Build token_cache for early exit
        let mut token_cache = FoldHashMap::default();
        for (id, bytes) in token_bytes.iter().enumerate() {
            if bytes.len() <= MAX_CACHED_TOKEN_LEN {
                token_cache.insert(bytes.clone(), id as TokenId);
            }
        }

        Self {
            matcher,
            scores,
            unk_token,
            byte_tokens,
            token_lengths,
            vocab_size,
            token_cache,
        }
    }

    /// Get vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get number of base tokens (for Unigram, this is vocab_size since there are no merges).
    pub fn num_base_tokens(&self) -> usize {
        self.vocab_size
    }

    /// Get unknown token ID.
    pub fn unk_token(&self) -> TokenId {
        self.unk_token
    }

    /// Get token scores.
    pub fn scores(&self) -> &[f32] {
        &self.scores
    }

    /// Get byte fallback tokens.
    pub fn byte_tokens(&self) -> &[TokenId; 256] {
        &self.byte_tokens
    }

    /// Get token lengths.
    pub fn token_lengths(&self) -> &[u16] {
        &self.token_lengths
    }

    /// Get reference to the DAAC matcher.
    pub fn matcher(&self) -> &DoubleArrayAhoCorasick {
        &self.matcher
    }

    /// Get token length in bytes.
    #[inline]
    pub fn token_len(&self, token: TokenId) -> usize {
        self.token_lengths[token as usize] as usize
    }

    /// Check if two tokens can appear adjacent in valid Unigram encoding.
    /// For Unigram, any pair is valid since there are no merge constraints.
    #[inline]
    pub fn is_valid_pair(&self, _token1: TokenId, _token2: TokenId) -> bool {
        true
    }

    /// Encode text to token IDs using Viterbi dynamic programming.
    ///
    /// For inputs larger than 64KB, automatically uses chunked encoding to avoid
    /// memory scaling issues (Viterbi DP allocates O(n) arrays which can be gigabytes
    /// for large inputs).
    ///
    /// Implementation note: We use f64 for accumulated scores during Viterbi to avoid
    /// precision loss when comparing paths with tiny score differences (~0.0003) after
    /// accumulating many tokens.
    pub fn encode(&self, text: &[u8]) -> Vec<TokenId> {
        if text.is_empty() {
            return Vec::new();
        }

        // Automatically use chunked encoding for large inputs to avoid memory issues
        if text.len() > DEFAULT_CHUNK_SIZE {
            return self.encode_chunked(text, DEFAULT_CHUNK_SIZE);
        }

        self.encode_single(text)
    }

    /// Encode a single chunk using Viterbi dynamic programming (no chunking).
    ///
    /// This is the core Viterbi algorithm. For large inputs, use `encode()` which
    /// automatically chunks, or `encode_chunked()` with a custom chunk size.
    ///
    /// **Warning**: For large inputs (>64KB), this allocates O(n) memory for DP arrays.
    /// Use `encode()` instead which automatically chunks large inputs.
    ///
    /// Algorithm using DAAC with Overlapping mode:
    /// 1. Run DAAC once over entire text to collect ALL matches - O(n + M)
    /// 2. Group matches by starting position
    /// 3. Process positions in order, using pre-grouped matches
    /// 4. Backward pass: reconstruct path from backpointers
    pub fn encode_single(&self, text: &[u8]) -> Vec<TokenId> {
        // NOTE: Unlike BPE, Unigram cannot use early exit for single-token matches.
        // Even if the input matches a single token, Viterbi might find a better
        // segmentation (e.g., "ab" as [a, b] scores -0.2 vs "ab" scores -10.0).
        // We must always run the full DP algorithm.

        let n = text.len();

        // best_score[i] = best log probability to reach position i
        // backptr[i] = (token_id, start_pos) that achieves best_score[i]
        let mut best_score = vec![f64::NEG_INFINITY; n + 1];
        let mut backptr: Vec<(TokenId, usize)> = vec![(0, 0); n + 1];
        best_score[0] = 0.0;

        // Penalty for unknown tokens (very negative to prefer any real token)
        const UNK_PENALTY: f64 = -100.0;

        // OPTIMIZATION: Group matches by start position using SmallVec
        // Most positions have few matches, so SmallVec avoids heap allocation
        type MatchList = SmallVec<[(usize, TokenId); 8]>;
        let mut matches_at: Vec<MatchList> = vec![SmallVec::new(); n];

        for m in self.matcher.find_iter(text) {
            matches_at[m.start].push((m.end, m.pattern_id));
        }

        // Forward pass: process positions in order
        for pos in 0..n {
            if best_score[pos] == f64::NEG_INFINITY {
                continue;
            }

            let current_score = best_score[pos];
            let has_match = !matches_at[pos].is_empty();

            // Process all matches starting at this position
            for &(end, token_id) in &matches_at[pos] {
                let token_score = self.scores[token_id as usize] as f64;
                let new_score = current_score + token_score;
                if new_score > best_score[end] {
                    best_score[end] = new_score;
                    backptr[end] = (token_id, pos);
                }
            }

            // Byte fallback for this position (try <0xXX> tokens)
            let byte_val = text[pos];
            let byte_token = self.byte_tokens[byte_val as usize];
            if byte_token != u32::MAX {
                let token_score = self.scores[byte_token as usize] as f64;
                let new_score = current_score + token_score;

                if new_score > best_score[pos + 1] {
                    best_score[pos + 1] = new_score;
                    backptr[pos + 1] = (byte_token, pos);
                }
            } else if !has_match {
                // No DAAC match and no byte fallback: use <unk>
                let new_score = current_score + UNK_PENALTY;
                if new_score > best_score[pos + 1] {
                    best_score[pos + 1] = new_score;
                    backptr[pos + 1] = (self.unk_token, pos);
                }
            }
        }

        // Handle case where Viterbi couldn't reach the end
        if best_score[n] == f64::NEG_INFINITY {
            return self.encode_with_unk_bridging(text, &best_score, &backptr);
        }

        // Backward pass: reconstruct tokens from backpointers
        self.collect_tokens_from_backptr(&backptr, n)
    }

    /// Collect tokens from backpointer array (backward pass of Viterbi).
    #[inline]
    fn collect_tokens_from_backptr(&self, backptr: &[(TokenId, usize)], end: usize) -> Vec<TokenId> {
        let mut tokens = Vec::new();
        let mut pos = end;
        while pos > 0 {
            let (token_id, start_pos) = backptr[pos];
            tokens.push(token_id);
            pos = start_pos;
        }
        tokens.reverse();
        tokens
    }

    /// Collect tokens with their byte spans from backpointer array.
    /// Returns Vec<(token_id, start_pos, end_pos)>.
    #[inline]
    fn collect_tokens_with_spans(&self, backptr: &[(TokenId, usize)], end: usize) -> Vec<(TokenId, usize, usize)> {
        let mut tokens = Vec::new();
        let mut pos = end;
        while pos > 0 {
            let (token_id, start_pos) = backptr[pos];
            tokens.push((token_id, start_pos, pos));
            pos = start_pos;
        }
        tokens.reverse();
        tokens
    }

    /// Encode with <unk> bridging for positions that couldn't be reached.
    fn encode_with_unk_bridging(
        &self,
        text: &[u8],
        _best_score: &[f64],
        _backptr: &[(TokenId, usize)],
    ) -> Vec<TokenId> {
        // This path is hit when even with <unk> fallback, Viterbi couldn't complete.
        // This shouldn't happen with our modified algorithm, but as a safety net,
        // we use a greedy approach: try longest match, else use <unk> for one byte.
        let n = text.len();
        let mut tokens = Vec::new();
        let mut pos = 0;

        while pos < n {
            // Try to find longest matching token
            let max_len = (n - pos).min(MAX_CACHED_TOKEN_LEN);
            let mut best_match: Option<(usize, TokenId)> = None;

            for len in (1..=max_len).rev() {
                let substr = &text[pos..pos + len];
                if let Some(&token_id) = self.token_cache.get(substr) {
                    best_match = Some((len, token_id));
                    break;
                }
            }

            // Also check DAAC for longer tokens (only at position 0)
            let remaining = &text[pos..];
            if let Some(m) = self.matcher.find_iter(remaining).next() {
                if m.start == 0 && (best_match.is_none() || m.end > best_match.unwrap().0) {
                    best_match = Some((m.end, m.pattern_id));
                }
            }

            if let Some((len, token_id)) = best_match {
                tokens.push(token_id);
                pos += len;
            } else {
                // No token found - try byte fallback
                let byte_val = text[pos];
                let byte_token = self.byte_tokens[byte_val as usize];
                if byte_token != u32::MAX {
                    tokens.push(byte_token);
                } else {
                    // No byte fallback - use <unk>
                    tokens.push(self.unk_token);
                }
                pos += 1;
            }
        }

        tokens
    }

    /// Encode text using chunked processing for better memory efficiency.
    ///
    /// For large texts, this splits at space/newline boundaries and encodes
    /// each chunk separately. This keeps Viterbi DP arrays small (~10MB per chunk)
    /// instead of allocating gigabytes for large inputs.
    ///
    /// Unlike BPE, Unigram can safely chunk at any position because:
    /// 1. No merge dependencies between chunks
    /// 2. Each chunk's optimal segmentation is independent
    /// 3. Total score is just the sum of chunk scores
    ///
    /// # Arguments
    ///
    /// * `text` - UTF-8 encoded input bytes
    /// * `chunk_size` - Target chunk size in bytes (default: 64KB)
    ///
    /// # Returns
    ///
    /// A vector of token IDs.
    pub fn encode_chunked(&self, text: &[u8], chunk_size: usize) -> Vec<TokenId> {
        if text.len() <= chunk_size {
            return self.encode_single(text);
        }

        let mut result = Vec::with_capacity(text.len() / 3);
        let mut pos = 0;

        while pos < text.len() {
            let remaining = text.len() - pos;

            if remaining <= chunk_size {
                // Last chunk: encode to the end
                let chunk_tokens = self.encode_single(&text[pos..]);
                result.extend_from_slice(&chunk_tokens);
                break;
            }

            // Encode chunk with overlap for boundary reconciliation
            let chunk_end = pos + chunk_size;
            let extended_end = (chunk_end + OVERLAP_SIZE).min(text.len());
            let chunk = &text[pos..extended_end];

            // Get tokens with their byte spans (relative to chunk start)
            let tokens_with_spans = self.encode_single_with_spans(chunk);

            if tokens_with_spans.is_empty() {
                pos = extended_end;
                continue;
            }

            // Find the last token that ends at or before the target boundary
            let target_boundary = chunk_size; // Relative to chunk start
            let mut cut_index = tokens_with_spans.len();
            let mut cut_pos = chunk.len();

            for (i, &(_token_id, _start, end)) in tokens_with_spans.iter().enumerate() {
                if end <= target_boundary {
                    cut_index = i + 1;
                    cut_pos = end;
                } else if end > target_boundary {
                    // This token crosses the boundary
                    // Use the previous token as the cut point
                    break;
                }
            }

            // Add tokens up to the cut point
            for &(token_id, _, _) in &tokens_with_spans[..cut_index] {
                result.push(token_id);
            }

            // Move position forward
            pos += cut_pos;
        }

        result
    }

    /// Encode a single chunk and return tokens with their byte spans.
    /// Returns Vec<(token_id, start_pos, end_pos)> where positions are relative to input start.
    fn encode_single_with_spans(&self, text: &[u8]) -> Vec<(TokenId, usize, usize)> {
        if text.is_empty() {
            return Vec::new();
        }

        let n = text.len();
        let mut best_score = vec![f64::NEG_INFINITY; n + 1];
        let mut backptr: Vec<(TokenId, usize)> = vec![(0, 0); n + 1];
        best_score[0] = 0.0;

        const UNK_PENALTY: f64 = -100.0;

        // Group matches by start position
        type MatchList = SmallVec<[(usize, TokenId); 8]>;
        let mut matches_at: Vec<MatchList> = vec![SmallVec::new(); n];

        for m in self.matcher.find_iter(text) {
            matches_at[m.start].push((m.end, m.pattern_id));
        }

        // Forward pass
        for pos in 0..n {
            if best_score[pos] == f64::NEG_INFINITY {
                continue;
            }

            let current_score = best_score[pos];
            let has_match = !matches_at[pos].is_empty();

            for &(end, token_id) in &matches_at[pos] {
                let token_score = self.scores[token_id as usize] as f64;
                let new_score = current_score + token_score;
                if new_score > best_score[end] {
                    best_score[end] = new_score;
                    backptr[end] = (token_id, pos);
                }
            }

            let byte_val = text[pos];
            let byte_token = self.byte_tokens[byte_val as usize];
            if byte_token != u32::MAX {
                let token_score = self.scores[byte_token as usize] as f64;
                let new_score = current_score + token_score;
                if new_score > best_score[pos + 1] {
                    best_score[pos + 1] = new_score;
                    backptr[pos + 1] = (byte_token, pos);
                }
            } else if !has_match {
                let new_score = current_score + UNK_PENALTY;
                if new_score > best_score[pos + 1] {
                    best_score[pos + 1] = new_score;
                    backptr[pos + 1] = (self.unk_token, pos);
                }
            }
        }

        if best_score[n] == f64::NEG_INFINITY {
            // Fallback: return empty (encode_single handles this case)
            return Vec::new();
        }

        self.collect_tokens_with_spans(&backptr, n)
    }

    /// Encode using default chunk size (64KB).
    #[inline]
    pub fn encode_chunked_default(&self, text: &[u8]) -> Vec<TokenId> {
        self.encode_chunked(text, DEFAULT_CHUNK_SIZE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_unigram() {
        // Simple vocab: "hello" (high score), "hell", "o", "h", "e", "l"
        let vocab = vec![
            (0, b"h".to_vec(), -1.0),
            (1, b"e".to_vec(), -1.0),
            (2, b"l".to_vec(), -1.0),
            (3, b"o".to_vec(), -1.0),
            (4, b"hell".to_vec(), -2.0),
            (5, b"hello".to_vec(), -3.0), // Best: one token for "hello"
        ];

        let (encoder, _) = UnigramEncoder::from_vocab_with_scores(&vocab, 0);

        // "hello" should be tokenized as single token [5] due to Viterbi
        // Score -3.0 vs h+e+l+l+o = -5.0
        assert_eq!(encoder.encode(b"hello"), vec![5]);

        // "h" should be single token [0]
        assert_eq!(encoder.encode(b"h"), vec![0]);
    }

    #[test]
    fn test_viterbi_chooses_best_path() {
        // Vocab where individual chars score better than combined
        let vocab = vec![
            (0, b"a".to_vec(), -0.1),    // Very good score
            (1, b"b".to_vec(), -0.1),    // Very good score
            (2, b"ab".to_vec(), -10.0),  // Bad score - should not be chosen
        ];

        let (encoder, _) = UnigramEncoder::from_vocab_with_scores(&vocab, 0);

        // "ab" should be tokenized as [0, 1] (score -0.2) not [2] (score -10.0)
        assert_eq!(encoder.encode(b"ab"), vec![0, 1]);
    }

    #[test]
    fn test_byte_fallback() {
        // Vocab with byte fallback tokens
        let vocab = vec![
            (0, b"<0x00>".to_vec(), -5.0),
            (1, b"<0x01>".to_vec(), -5.0),
            (2, b"<0xFF>".to_vec(), -5.0),
            (3, b"hello".to_vec(), -1.0),
        ];

        let (encoder, _) = UnigramEncoder::from_vocab_with_scores(&vocab, 0);

        // Verify byte tokens were parsed correctly
        assert_eq!(encoder.byte_tokens[0x00], 0);
        assert_eq!(encoder.byte_tokens[0x01], 1);
        assert_eq!(encoder.byte_tokens[0xFF], 2);
    }

    #[test]
    fn test_empty_input() {
        let vocab = vec![(0, b"a".to_vec(), -1.0)];
        let (encoder, _) = UnigramEncoder::from_vocab_with_scores(&vocab, 0);
        let empty: Vec<TokenId> = vec![];
        assert_eq!(encoder.encode(b""), empty);
    }

    #[test]
    fn test_vocab_size() {
        let vocab = vec![
            (0, b"a".to_vec(), -1.0),
            (1, b"b".to_vec(), -1.0),
            (2, b"c".to_vec(), -1.0),
        ];
        let (encoder, _) = UnigramEncoder::from_vocab_with_scores(&vocab, 0);
        assert_eq!(encoder.vocab_size(), 3);
        assert_eq!(encoder.num_base_tokens(), 3);
    }
}
