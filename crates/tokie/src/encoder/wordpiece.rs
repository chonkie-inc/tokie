//! WordPiece encoder using Aho-Corasick automata with custom failure links.
//!
//! WordPiece tokenization (used by BERT, DistilBERT, etc.) uses greedy
//! longest-match-first (Maximum Matching) instead of BPE merge rules.
//!
//! Key differences from BPE:
//! - No merge rules - vocabulary is pre-defined
//! - Continuation tokens use "##" prefix (configurable)
//! - Unknown tokens return [UNK] token ID

use daggrs::{DoubleArrayAhoCorasick, Trie};
use foldhash::HashMap as FoldHashMap;

use crate::types::TokenId;

/// Default continuation prefix for WordPiece (BERT-style).
pub const DEFAULT_CONTINUATION_PREFIX: &[u8] = b"##";

/// WordPiece encoder using DAAC with custom failure links.
///
/// This encoder builds a trie/DAAC where:
/// - All vocabulary tokens are added as patterns
/// - Failure links from completed tokens point to the continuation anchor ("##")
/// - Tokenization uses greedy longest-match with anchor jumping
#[derive(Clone)]
pub struct WordPieceEncoder {
    /// The DAAC automaton with WordPiece failure links.
    matcher: DoubleArrayAhoCorasick,

    /// Token ID for unknown/untokenizable words.
    unk_token: TokenId,

    /// Continuation prefix (e.g., b"##").
    continuation_prefix: Vec<u8>,

    /// Total vocabulary size.
    vocab_size: usize,

    /// Single-byte fast path: direct array lookup (no hashing).
    byte_lut: [TokenId; 256],

    /// Multi-byte token cache for early exit.
    token_cache: FoldHashMap<Vec<u8>, TokenId>,
}

impl WordPieceEncoder {
    /// Create a WordPiece encoder from vocabulary.
    ///
    /// # Arguments
    /// * `vocab` - Vocabulary mapping token bytes to token IDs
    /// * `unk_token` - Token ID to return for unknown words
    /// * `continuation_prefix` - Prefix for continuation tokens (default: "##")
    pub fn from_vocab(
        vocab: &[(Vec<u8>, TokenId)],
        unk_token: TokenId,
        continuation_prefix: &[u8],
    ) -> Self {
        let mut trie = Trie::new();

        // Add all vocabulary tokens to the trie
        for (bytes, token_id) in vocab {
            trie.add(bytes, *token_id);
        }

        // Build with WordPiece mode (failure links -> anchor)
        trie.build_wordpiece(continuation_prefix);

        // Compile to DAAC for fast matching
        let matcher = trie.compile();

        // Build single-byte fast path array
        let mut byte_lut = [unk_token; 256];
        for (bytes, token_id) in vocab {
            if bytes.len() == 1 {
                byte_lut[bytes[0] as usize] = *token_id;
            }
        }

        // Build multi-byte token cache
        let mut token_cache = FoldHashMap::default();
        for (bytes, token_id) in vocab {
            if bytes.len() <= 16 {
                token_cache.insert(bytes.clone(), *token_id);
            }
        }

        Self {
            matcher,
            unk_token,
            continuation_prefix: continuation_prefix.to_vec(),
            vocab_size: vocab.len(),
            byte_lut,
            token_cache,
        }
    }

    /// Create a WordPiece encoder from vocabulary with default "##" prefix.
    pub fn from_vocab_default(vocab: &[(Vec<u8>, TokenId)], unk_token: TokenId) -> Self {
        Self::from_vocab(vocab, unk_token, DEFAULT_CONTINUATION_PREFIX)
    }

    /// Create a WordPiece encoder from pre-built components (for fast deserialization).
    ///
    /// This avoids rebuilding the DAAC from scratch.
    pub fn from_parts(
        matcher: DoubleArrayAhoCorasick,
        unk_token: TokenId,
        continuation_prefix: Vec<u8>,
        vocab_size: usize,
        token_bytes: &[Vec<u8>],
    ) -> Self {
        // Build single-byte fast path array
        let mut byte_lut = [unk_token; 256];
        for (token_id, bytes) in token_bytes.iter().enumerate() {
            if bytes.len() == 1 {
                byte_lut[bytes[0] as usize] = token_id as TokenId;
            }
        }

        // Build early exit cache
        let mut token_cache = FoldHashMap::default();
        for (token_id, bytes) in token_bytes.iter().enumerate() {
            if bytes.len() <= 16 {
                token_cache.insert(bytes.clone(), token_id as TokenId);
            }
        }

        Self {
            matcher,
            unk_token,
            continuation_prefix,
            vocab_size,
            byte_lut,
            token_cache,
        }
    }

    /// Encode a word (pre-tokenized piece) into token IDs.
    ///
    /// Uses longest-match-first (MaxMatch) algorithm with rewind for WordPiece.
    /// Returns `[unk_token]` if the word cannot be fully tokenized.
    pub fn encode(&self, word: &[u8]) -> Vec<TokenId> {
        if word.is_empty() {
            return Vec::new();
        }

        // Single-byte fast path: direct array lookup (no hashing)
        if word.len() == 1 {
            return vec![self.byte_lut[word[0] as usize]];
        }

        // Early exit for cached tokens
        if let Some(&token_id) = self.token_cache.get(word) {
            return vec![token_id];
        }

        // Get anchor state for continuation matching
        let anchor = match self.matcher.anchor {
            Some(a) => a,
            None => return vec![self.unk_token],
        };

        let mut result = Vec::new();
        let mut pos = 0usize;
        let mut state = self.matcher.start_state();
        let mut last_match: Option<(usize, TokenId)> = None;

        loop {
            // Inner loop: process characters until end of word or transition failure
            while pos < word.len() {
                if let Some(next_state) = self.try_transition(state, word[pos]) {
                    state = next_state;
                    pos += 1;

                    // Record match if this state has an output
                    if let Some(output) = self.matcher.outputs(state).next() {
                        last_match = Some((pos, output.pattern_id));
                    }
                } else {
                    // Transition failed - emit last match or return UNK
                    if let Some((end_pos, token_id)) = last_match.take() {
                        result.push(token_id);
                        pos = end_pos;
                        state = anchor;
                    } else {
                        return vec![self.unk_token];
                    }
                }
            }

            // End of word reached - emit pending match if any
            if let Some((end_pos, token_id)) = last_match.take() {
                result.push(token_id);

                // If match doesn't cover all remaining chars, rewind and continue
                // This handles cases like "grippe" → ["grip", "##pe"] where
                // transitions for "gripp"/"grippe" succeed without outputs
                if end_pos < word.len() {
                    pos = end_pos;
                    state = anchor;
                    continue;
                }
            }

            break;
        }

        if result.is_empty() {
            vec![self.unk_token]
        } else {
            result
        }
    }

    /// Try to transition to next state, returns None if no valid transition.
    #[inline]
    fn try_transition(&self, state: u32, byte: u8) -> Option<u32> {
        let states = &self.matcher.states;
        let current = &states[state as usize];
        let child = current.base ^ (byte as u32);

        if (child as usize) < states.len() && states[child as usize].check == state {
            Some(child)
        } else {
            None
        }
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get the unknown token ID.
    pub fn unk_token(&self) -> TokenId {
        self.unk_token
    }

    /// Get the continuation prefix.
    pub fn continuation_prefix(&self) -> &[u8] {
        &self.continuation_prefix
    }

    /// Get a reference to the underlying DAAC matcher.
    pub fn matcher(&self) -> &DoubleArrayAhoCorasick {
        &self.matcher
    }

    /// Check if two tokens can appear adjacent.
    /// For WordPiece, this is always true (no merge constraints).
    pub fn is_valid_pair(&self, _token1: TokenId, _token2: TokenId) -> bool {
        true
    }

    /// Number of base tokens (for compatibility - WordPiece doesn't have this concept).
    pub fn num_base_tokens(&self) -> usize {
        self.vocab_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_vocab() -> Vec<(Vec<u8>, TokenId)> {
        vec![
            (b"[UNK]".to_vec(), 0),
            (b"un".to_vec(), 1),
            (b"break".to_vec(), 2),
            (b"##break".to_vec(), 3),
            (b"##able".to_vec(), 4),
            (b"##ing".to_vec(), 5),
        ]
    }

    #[test]
    fn test_wordpiece_single_token() {
        let vocab = make_test_vocab();
        let encoder = WordPieceEncoder::from_vocab_default(&vocab, 0);

        // "un" should encode to [1]
        let tokens = encoder.encode(b"un");
        assert_eq!(tokens, vec![1]);
    }

    #[test]
    fn test_wordpiece_continuation() {
        let vocab = make_test_vocab();
        let encoder = WordPieceEncoder::from_vocab_default(&vocab, 0);

        // "unbreakable" should encode to ["un", "##break", "##able"] = [1, 3, 4]
        let tokens = encoder.encode(b"unbreakable");
        assert_eq!(tokens, vec![1, 3, 4]);
    }

    #[test]
    fn test_wordpiece_unknown() {
        let vocab = make_test_vocab();
        let encoder = WordPieceEncoder::from_vocab_default(&vocab, 0);

        // "xyz" is not in vocab, should return [UNK]
        let tokens = encoder.encode(b"xyz");
        assert_eq!(tokens, vec![0]);
    }

    #[test]
    fn test_wordpiece_empty() {
        let vocab = make_test_vocab();
        let encoder = WordPieceEncoder::from_vocab_default(&vocab, 0);

        let tokens = encoder.encode(b"");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_wordpiece_partial_unknown() {
        let vocab = make_test_vocab();
        let encoder = WordPieceEncoder::from_vocab_default(&vocab, 0);

        // "unxyz" - "un" matches but "xyz" doesn't have continuation
        let tokens = encoder.encode(b"unxyz");
        assert_eq!(tokens, vec![0]); // Should return [UNK]
    }
}
