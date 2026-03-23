//! SentencePiece BPE v2 - Fast O(n²) encoder with character-level initialization.
//!
//! This encoder combines:
//! - Character-level initialization from SentencePieceBPE (handles multi-byte UTF-8 tokens)
//! - O(n²) scan-and-merge algorithm from BytePairEncoder (faster for small inputs)
//!
//! The key insight is that SentencePiece tokenizers have multi-byte tokens in their
//! vocabulary (like metaspace `▁`), so we can't initialize with raw byte tokens.
//! Instead, we scan for the longest matching token at each position.

use foldhash::HashMap as FoldHashMap;
use memchunk::chunk;
use smallvec::SmallVec;

use crate::types::TokenId;

/// Metaspace character (▁) in UTF-8: E2 96 81.
const METASPACE: [u8; 3] = [0xE2, 0x96, 0x81];

/// Get UTF-8 character length from the first byte.
#[inline]
fn utf8_char_len(b: u8) -> usize {
    if b < 0x80 {
        1
    } else if b < 0xE0 {
        2
    } else if b < 0xF0 {
        3
    } else {
        4
    }
}

/// Pack two u32 token IDs into a single u64 key for faster hashing.
#[inline(always)]
fn pack_pair(left: TokenId, right: TokenId) -> u64 {
    ((left as u64) << 32) | (right as u64)
}

/// Fast SentencePiece BPE encoder with character-level initialization.
///
/// Uses O(n²) scan-and-merge like BytePairEncoder, but initializes with
/// character-level tokens instead of byte tokens to handle multi-byte
/// vocabulary entries (like metaspace `▁`).
#[derive(Clone)]
pub struct SentencePieceBPEv2 {
    /// Maps (left_token, right_token) -> (merged_token, merge_rank).
    pair_lookup: FoldHashMap<u64, (TokenId, u32)>,

    /// Maps byte value -> token ID (for single-byte fallback).
    byte_lut: [TokenId; 256],

    /// Maps byte sequence -> token ID for token lookup.
    token_cache: FoldHashMap<Vec<u8>, TokenId>,

    /// Maximum token length in bytes (for early exit and initialization).
    max_token_len: usize,

    /// Total vocabulary size.
    vocab_size: usize,

    /// Number of base tokens.
    num_base_tokens: usize,
}

impl SentencePieceBPEv2 {
    /// Create encoder from a complete vocabulary and merge rules.
    ///
    /// For SentencePiece-style tokenizers (Mistral, Llama 1/2, Gemma).
    pub fn from_vocab_and_merges(
        vocab: &[(u32, Vec<u8>)],
        merges: &[(TokenId, TokenId)],
        num_base_tokens: usize,
    ) -> (Self, Vec<Vec<u8>>) {
        let token_bytes: Vec<Vec<u8>> = vocab.iter().map(|(_, bytes)| bytes.clone()).collect();

        // Build byte -> token mapping for single-byte fallback
        let mut byte_lut = [u32::MAX; 256];
        for (id, bytes) in vocab {
            if bytes.len() == 1 {
                let byte_val = bytes[0] as usize;
                if byte_lut[byte_val] == u32::MAX {
                    byte_lut[byte_val] = *id;
                }
            }
        }

        // Build bytes -> token ID lookup for ALL tokens
        let all_bytes_to_id: FoldHashMap<Vec<u8>, TokenId> = vocab
            .iter()
            .map(|(id, bytes)| (bytes.clone(), *id))
            .collect();

        // Build pair_lookup with merge ranks
        let mut pair_lookup = FoldHashMap::default();
        for (merge_index, &(left, right)) in merges.iter().enumerate() {
            let mut merged_bytes = token_bytes[left as usize].clone();
            merged_bytes.extend_from_slice(&token_bytes[right as usize]);

            if let Some(&merged_id) = all_bytes_to_id.get(&merged_bytes) {
                pair_lookup
                    .entry(pack_pair(left, right))
                    .or_insert((merged_id, merge_index as u32));
            }
        }

        // Find max token length for early exit and initialization
        let max_token_len = token_bytes.iter().map(|t| t.len()).max().unwrap_or(16);

        // Build token_cache for early exit and initialization (all tokens)
        let mut token_cache = FoldHashMap::default();
        for (token_id, bytes) in token_bytes.iter().enumerate() {
            token_cache.insert(bytes.clone(), token_id as TokenId);
        }

        let encoder = Self {
            pair_lookup,
            byte_lut,
            token_cache,
            max_token_len,
            vocab_size: vocab.len(),
            num_base_tokens,
        };

        (encoder, token_bytes)
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get the number of base tokens.
    pub fn num_base_tokens(&self) -> usize {
        self.num_base_tokens
    }

    /// Check if two tokens can appear adjacent in a valid BPE encoding.
    #[inline]
    pub fn is_valid_pair(&self, token1: TokenId, token2: TokenId) -> bool {
        !self.pair_lookup.contains_key(&pack_pair(token1, token2))
    }

    /// Reconstruct encoder from serialized parts.
    pub fn from_parts(
        merges: &[(TokenId, TokenId, TokenId)], // (left, right, merged_id)
        byte_lut: [TokenId; 256],
        token_cache: FoldHashMap<Vec<u8>, TokenId>,
        vocab_size: usize,
        num_base_tokens: usize,
    ) -> Self {
        let mut pair_lookup = FoldHashMap::default();
        for (merge_index, &(left, right, merged_id)) in merges.iter().enumerate() {
            pair_lookup.insert(pack_pair(left, right), (merged_id, merge_index as u32));
        }

        // Compute max token length from cache
        let max_token_len = token_cache.keys().map(|k| k.len()).max().unwrap_or(16);

        Self {
            pair_lookup,
            byte_lut,
            token_cache,
            max_token_len,
            vocab_size,
            num_base_tokens,
        }
    }

    /// Find the token for ONE UTF-8 character at the given position.
    ///
    /// This matches v1's character-level initialization: we look up exactly one
    /// UTF-8 character (not greedy longest match). Multi-byte UTF-8 characters
    /// like metaspace (▁) are looked up as a whole, then merging handles the rest.
    #[inline]
    fn find_token_at(&self, text: &[u8], pos: usize) -> (TokenId, usize) {
        // Get UTF-8 character length
        let char_len = utf8_char_len(text[pos]);
        let end = (pos + char_len).min(text.len());
        let char_bytes = &text[pos..end];

        // Try to find this UTF-8 character in the vocab
        if let Some(&token_id) = self.token_cache.get(char_bytes) {
            return (token_id, char_bytes.len());
        }

        // Fallback to single byte
        let byte_token = self.byte_lut[text[pos] as usize];
        if byte_token != u32::MAX {
            (byte_token, 1)
        } else {
            // Unknown byte - this shouldn't happen with proper vocab
            (0, 1)
        }
    }

    /// Encode bytes into BPE tokens.
    ///
    /// Uses character-level initialization (handles multi-byte tokens like `▁`)
    /// followed by optimized merge loop that only re-evaluates affected positions.
    #[inline]
    pub fn encode(&self, text: &[u8]) -> Vec<TokenId> {
        if text.is_empty() {
            return Vec::new();
        }

        // OPTIMIZATION 1: Early exit if entire input is a single token
        if text.len() <= self.max_token_len {
            if let Some(&token_id) = self.token_cache.get(text) {
                return vec![token_id];
            }
        }

        // OPTIMIZATION 2: Single UTF-8 character fast path
        let first_char_len = utf8_char_len(text[0]);
        if text.len() == first_char_len {
            // Single character - try cache, then byte_lut
            if let Some(&token_id) = self.token_cache.get(text) {
                return vec![token_id];
            }
            if first_char_len == 1 {
                let token = self.byte_lut[text[0] as usize];
                if token != u32::MAX {
                    return vec![token];
                }
            }
        }

        // Initialize with character-level tokens (not byte-level!)
        // This is the key difference from BytePairEncoder
        let mut tokens: SmallVec<[TokenId; 64]> = SmallVec::new();
        let mut pos = 0;
        while pos < text.len() {
            let (token, len) = self.find_token_at(text, pos);
            tokens.push(token);
            pos += len;
        }

        let mut len = tokens.len();

        // Early exit if only one token after initialization
        if len == 1 {
            return tokens.into_vec();
        }

        // OPTIMIZATION 3: Pre-compute merge info for all adjacent pairs
        // merge_info[i] = Some((merged_token, rank)) if tokens[i] and tokens[i+1] can merge
        let mut merge_info: SmallVec<[Option<(TokenId, u32)>; 64]> = SmallVec::new();
        for i in 0..len - 1 {
            merge_info.push(self.pair_lookup.get(&pack_pair(tokens[i], tokens[i + 1])).copied());
        }
        merge_info.push(None); // Sentinel for last position

        // Merge loop with incremental updates
        loop {
            // Find best merge from cached info
            let mut best_rank = u32::MAX;
            let mut best_pos = usize::MAX;
            let mut best_merged = 0;

            for i in 0..len - 1 {
                if let Some((merged, rank)) = merge_info[i] {
                    // Match v1's tie-breaking: for equal ranks, prefer later positions
                    if rank < best_rank || (rank == best_rank && i > best_pos) {
                        best_rank = rank;
                        best_pos = i;
                        best_merged = merged;
                    }
                }
            }

            if best_pos == usize::MAX {
                break; // No more merges
            }

            // Apply the merge
            tokens[best_pos] = best_merged;
            tokens.copy_within(best_pos + 2..len, best_pos + 1);

            // Update merge_info for affected positions only
            // Position best_pos-1: now pairs with merged token
            if best_pos > 0 {
                merge_info[best_pos - 1] = self.pair_lookup.get(&pack_pair(tokens[best_pos - 1], best_merged)).copied();
            }
            // Position best_pos: now pairs with what was at best_pos+2
            if best_pos + 1 < len - 1 {
                merge_info[best_pos] = self.pair_lookup.get(&pack_pair(best_merged, tokens[best_pos + 1])).copied();
            } else {
                merge_info[best_pos] = None;
            }
            // Shift merge_info left (remove the merged-away pair's entry)
            merge_info.copy_within(best_pos + 2..len, best_pos + 1);

            len -= 1;
        }

        tokens.truncate(len);
        tokens.into_vec()
    }

    /// Encode text using chunked processing for better performance.
    ///
    /// For large texts, this splits at metaspace (▁) boundaries and encodes
    /// each chunk separately. This keeps each chunk small enough for O(n²)
    /// to be efficient.
    ///
    /// # Arguments
    ///
    /// * `text` - UTF-8 encoded input bytes
    /// * `chunk_size` - Target chunk size in bytes (will align to metaspace boundaries)
    ///
    /// # Returns
    ///
    /// A vector of token IDs.
    pub fn encode_chunked(&self, text: &[u8], chunk_size: usize) -> Vec<TokenId> {
        if text.len() <= chunk_size {
            return self.encode(text);
        }

        let mut result = Vec::with_capacity(text.len() / 4);

        for chunk_bytes in chunk(text)
            .size(chunk_size)
            .pattern(&METASPACE)
            .prefix()
            .consecutive()
            .forward_fallback()
        {
            let chunk_tokens = self.encode(chunk_bytes);
            result.extend_from_slice(&chunk_tokens);
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_merge() {
        let vocab = vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"ab".to_vec()),
        ];
        let merges = vec![(0, 1)]; // a + b -> ab

        let (encoder, _) = SentencePieceBPEv2::from_vocab_and_merges(&vocab, &merges, 2);

        assert_eq!(encoder.encode(b"ab"), vec![2]);
        assert_eq!(encoder.encode(b"a"), vec![0]);
        assert_eq!(encoder.encode(b"ba"), vec![1, 0]);
    }

    #[test]
    fn test_metaspace_token() {
        // Metaspace is a 3-byte UTF-8 character
        let metaspace = "▁".as_bytes().to_vec(); // [0xE2, 0x96, 0x81]

        let vocab = vec![
            (0, metaspace.clone()),
            (1, b"H".to_vec()),
            (2, [metaspace.clone(), b"H".to_vec()].concat()),
        ];
        let merges = vec![(0, 1)]; // ▁ + H -> ▁H

        let (encoder, _) = SentencePieceBPEv2::from_vocab_and_merges(&vocab, &merges, 2);

        assert_eq!(encoder.encode("▁H".as_bytes()), vec![2]);
        assert_eq!(encoder.encode("▁".as_bytes()), vec![0]);
    }

    #[test]
    fn test_merge_priority() {
        let vocab = vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"c".to_vec()),
            (3, b"ab".to_vec()),
            (4, b"bc".to_vec()),
            (5, b"abc".to_vec()),
        ];
        // ab has rank 0 (highest priority), bc has rank 1, abc has rank 2
        let merges = vec![
            (0, 1), // a + b -> ab (rank 0)
            (1, 2), // b + c -> bc (rank 1) - but ab forms first!
            (3, 2), // ab + c -> abc (rank 2)
        ];

        let (encoder, _) = SentencePieceBPEv2::from_vocab_and_merges(&vocab, &merges, 3);

        // "abc" should become: a+b->ab (rank 0), then ab+c->abc (rank 2)
        // NOT: b+c->bc (rank 1) because ab forms first
        assert_eq!(encoder.encode(b"abc"), vec![5]);
    }
}
