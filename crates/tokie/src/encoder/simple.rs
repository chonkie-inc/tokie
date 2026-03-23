//! Optimized Simple BPE encoder with single-byte fast path.
//!
//! Key optimizations:
//! 1. Single-byte fast path: 21.5% of pieces are 1 byte - use direct array lookup
//! 2. Early exit for single tokens (88.9% of pieces)
//! 3. foldhash + packed u64 keys for fast hash lookups

use foldhash::HashMap as FoldHashMap;
use smallvec::SmallVec;

use crate::types::TokenId;

/// Maximum token length to cache for early exit lookup.
/// Covers 99%+ of pretokenized pieces.
const MAX_CACHED_TOKEN_LEN: usize = 16;

/// Pack two u32 token IDs into a single u64 key for faster hashing.
#[inline(always)]
fn pack_pair(left: TokenId, right: TokenId) -> u64 {
    ((left as u64) << 32) | (right as u64)
}

/// Optimized BPE encoder with single-byte fast path.
///
/// Adds direct array lookup for single-byte inputs (21.5% of pieces)
/// before falling back to hash-based early exit lookup.
#[derive(Clone)]
pub struct BytePairEncoder {
    /// Maps (left_token, right_token) -> (merged_token, merge_rank).
    /// Uses foldhash for fast lookups.
    pair_lookup: FoldHashMap<u64, (TokenId, u32)>,

    /// Maps byte value -> token ID (for base tokens).
    byte_lut: [TokenId; 256],

    /// Maps byte sequence -> token ID for early exit.
    /// Only includes tokens up to MAX_CACHED_TOKEN_LEN bytes.
    /// Uses foldhash for fast lookups.
    token_cache: FoldHashMap<Vec<u8>, TokenId>,

    /// Total vocabulary size.
    vocab_size: usize,

    /// Number of base (single-byte) tokens.
    num_base_tokens: usize,
}

impl BytePairEncoder {
    /// Create a new encoder from merge rules.
    pub fn from_merges(
        merges: &[(TokenId, TokenId)],
        base_tokens: &[Vec<u8>],
    ) -> (Self, Vec<Vec<u8>>) {
        Self::from_merges_with_added(merges, base_tokens, &[])
    }

    /// Create encoder from merge rules with added tokens.
    pub fn from_merges_with_added(
        merges: &[(TokenId, TokenId)],
        base_tokens: &[Vec<u8>],
        added_tokens: &[(u32, Vec<u8>)],
    ) -> (Self, Vec<Vec<u8>>) {
        let mut token_bytes: Vec<Vec<u8>> = base_tokens.to_vec();
        let mut pair_lookup = FoldHashMap::default();

        // Build byte -> token mapping
        let mut byte_lut = [0u32; 256];
        for (token_id, bytes) in base_tokens.iter().enumerate() {
            if bytes.len() == 1 {
                byte_lut[bytes[0] as usize] = token_id as TokenId;
            }
        }
        // Fallback for unmapped bytes
        for (i, token) in byte_lut.iter_mut().enumerate() {
            if *token == 0 && i < base_tokens.len() {
                if base_tokens.get(i).is_some_and(|b| b.len() == 1 && b[0] == i as u8) {
                    *token = i as TokenId;
                }
            }
        }

        // Handle added tokens interleaved with merges
        let mut added_sorted: Vec<_> = added_tokens.to_vec();
        added_sorted.sort_by_key(|(id, _)| *id);
        let mut added_iter = added_sorted.into_iter().peekable();

        for (merge_index, &(left, right)) in merges.iter().enumerate() {
            let next_id = token_bytes.len() as TokenId;

            // Insert any added tokens that come before this merge
            while let Some(&(added_id, _)) = added_iter.peek() {
                if added_id <= next_id {
                    let (_, bytes) = added_iter.next().unwrap();
                    token_bytes.push(bytes);
                } else {
                    break;
                }
            }

            let new_id = token_bytes.len() as TokenId;
            pair_lookup.insert(pack_pair(left, right), (new_id, merge_index as u32));

            let mut bytes = token_bytes[left as usize].clone();
            bytes.extend_from_slice(&token_bytes[right as usize]);
            token_bytes.push(bytes);
        }

        // Append remaining added tokens
        for (_, bytes) in added_iter {
            token_bytes.push(bytes);
        }

        // Build token_cache for early exit optimization (foldhash for speed)
        let mut token_cache = FoldHashMap::default();
        for (token_id, bytes) in token_bytes.iter().enumerate() {
            if bytes.len() <= MAX_CACHED_TOKEN_LEN {
                token_cache.insert(bytes.clone(), token_id as TokenId);
            }
        }

        let vocab_size = token_bytes.len();
        let num_base_tokens = base_tokens.len();

        let encoder = Self {
            pair_lookup,
            byte_lut,
            token_cache,
            vocab_size,
            num_base_tokens,
        };

        (encoder, token_bytes)
    }

    /// Create encoder from a complete vocabulary and merge rules.
    ///
    /// For tokenizers (like LLaMA 3) where vocab has pre-assigned IDs.
    pub fn from_vocab_and_merges(
        vocab: &[(u32, Vec<u8>)],
        merges: &[(TokenId, TokenId)],
        num_base_tokens: usize,
    ) -> (Self, Vec<Vec<u8>>) {
        let token_bytes: Vec<Vec<u8>> = vocab.iter().map(|(_, bytes)| bytes.clone()).collect();

        // Build byte -> token mapping by scanning ALL tokens.
        // For SentencePiece models, byte tokens (<0x00>, etc.) can be scattered
        // throughout the vocab, not just at the beginning.
        let mut byte_lut = [u32::MAX; 256];
        for (token_id, bytes) in token_bytes.iter().enumerate() {
            if bytes.len() == 1 {
                let byte_val = bytes[0] as usize;
                // Only set if not already mapped (prefer earlier token IDs)
                if byte_lut[byte_val] == u32::MAX {
                    byte_lut[byte_val] = token_id as TokenId;
                }
            }
        }

        // Build bytes -> token ID lookup for ALL tokens (used in construction)
        let all_bytes_to_id: FoldHashMap<Vec<u8>, TokenId> = vocab
            .iter()
            .map(|(id, bytes)| (bytes.clone(), *id))
            .collect();

        // Build pair_lookup with merge ranks (pack (u32, u32) into u64 for faster hashing)
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

        // Build token_cache for early exit (foldhash for speed)
        let mut token_cache = FoldHashMap::default();
        for (token_id, bytes) in token_bytes.iter().enumerate() {
            if bytes.len() <= MAX_CACHED_TOKEN_LEN {
                token_cache.insert(bytes.clone(), token_id as TokenId);
            }
        }

        let encoder = Self {
            pair_lookup,
            byte_lut,
            token_cache,
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

    /// Get a reference to the pair lookup table.
    pub fn pair_lookup(&self) -> &FoldHashMap<u64, (TokenId, u32)> {
        &self.pair_lookup
    }

    /// Check if two tokens can appear adjacent in a valid BPE encoding.
    ///
    /// Returns false if there exists a merge that would combine these tokens.
    /// Note: This is a simplified check compared to BacktrackingBytePairEncoder.
    #[inline]
    pub fn is_valid_pair(&self, token1: TokenId, token2: TokenId) -> bool {
        // If there's a merge for this pair, they shouldn't appear adjacent
        !self.pair_lookup.contains_key(&pack_pair(token1, token2))
    }

    /// Reconstruct encoder from serialized parts with pre-computed merged IDs and lookups.
    ///
    /// Used during deserialization to rebuild the encoder from saved data.
    /// - `merged_id` is pre-computed during serialization
    /// - `byte_lut` and `token_cache` are pre-built from decoder data (single copy)
    pub fn from_parts(
        merges: &[(TokenId, TokenId, TokenId)], // (left, right, merged_id)
        byte_lut: [TokenId; 256],
        token_cache: FoldHashMap<Vec<u8>, TokenId>,
        vocab_size: usize,
        num_base_tokens: usize,
    ) -> Self {
        // Build pair_lookup directly from pre-computed merged IDs - O(num_merges)
        let mut pair_lookup = FoldHashMap::default();
        for (merge_index, &(left, right, merged_id)) in merges.iter().enumerate() {
            pair_lookup.insert(pack_pair(left, right), (merged_id, merge_index as u32));
        }

        Self {
            pair_lookup,
            byte_lut,
            token_cache,
            vocab_size,
            num_base_tokens,
        }
    }

    /// Encode bytes into BPE tokens.
    ///
    /// Optimizations:
    /// 1. Single-byte fast path: direct array lookup (no hashing)
    /// 2. Early exit: hash lookup for known tokens
    /// 3. SmallVec: avoid heap allocation for small pieces
    #[inline]
    pub fn encode(&self, text: &[u8]) -> Vec<TokenId> {
        if text.is_empty() {
            return Vec::new();
        }

        // OPTIMIZATION 1: Single-byte fast path (21.5% of pieces)
        // Direct array lookup - no hashing needed!
        if text.len() == 1 {
            return vec![self.byte_lut[text[0] as usize]];
        }

        // OPTIMIZATION 2: Early exit if input is already a single token
        // This handles most of the remaining 88.9% of pretokenized pieces
        if text.len() <= MAX_CACHED_TOKEN_LEN {
            if let Some(&token_id) = self.token_cache.get(text) {
                return vec![token_id];
            }
        }

        // Initialize with byte tokens using SmallVec (stack allocation for ≤16 tokens)
        let mut tokens: SmallVec<[TokenId; 16]> = text
            .iter()
            .map(|&b| self.byte_lut[b as usize])
            .collect();

        let mut len = tokens.len();

        // Merge until no more merges possible
        while len > 1 {
            // Find the lowest-rank merge
            let mut best_rank = u32::MAX;
            let mut best_pos = usize::MAX;
            let mut best_merged = 0;

            for i in 0..len - 1 {
                if let Some(&(merged, rank)) = self.pair_lookup.get(&pack_pair(tokens[i], tokens[i + 1])) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_pos = i;
                        best_merged = merged;
                    }
                }
            }

            if best_pos == usize::MAX {
                break; // No more merges
            }

            // Apply the merge: replace pair with merged token
            tokens[best_pos] = best_merged;
            // Shift remaining tokens left by one
            tokens.copy_within(best_pos + 2..len, best_pos + 1);
            len -= 1;
        }

        tokens.truncate(len);
        tokens.into_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::Decoder;

    #[test]
    fn test_encode_basic() {
        let base_tokens = vec![vec![b'a'], vec![b'b'], vec![b'c']];
        let merges = vec![(0, 1), (3, 2)]; // a+b->ab(3), ab+c->abc(4)

        let (encoder, token_bytes) = BytePairEncoder::from_merges(&merges, &base_tokens);
        let decoder = Decoder::new(token_bytes);

        let encoded = encoder.encode(b"abc");
        assert_eq!(encoded, vec![4]); // abc

        assert_eq!(decoder.decode(&encoded), b"abc");
    }

    #[test]
    fn test_single_byte_fast_path() {
        let base_tokens = vec![vec![b'a'], vec![b'b'], vec![b'c']];
        let merges = vec![(0, 1)];

        let (encoder, _) = BytePairEncoder::from_merges(&merges, &base_tokens);

        // Single byte should use fast path (direct array lookup)
        assert_eq!(encoder.encode(b"a"), vec![0]);
        assert_eq!(encoder.encode(b"b"), vec![1]);
        assert_eq!(encoder.encode(b"c"), vec![2]);
    }

    #[test]
    fn test_early_exit_multi_byte_token() {
        let base_tokens = vec![vec![b'a'], vec![b'b'], vec![b'c']];
        let merges = vec![(0, 1), (3, 2)]; // a+b->ab(3), ab+c->abc(4)

        let (encoder, _) = BytePairEncoder::from_merges(&merges, &base_tokens);

        // "ab" is token 3, should use early exit
        assert_eq!(encoder.encode(b"ab"), vec![3]);

        // "abc" is token 4, should use early exit
        assert_eq!(encoder.encode(b"abc"), vec![4]);
    }

    #[test]
    fn test_encode_roundtrip() {
        let base_tokens = vec![vec![b'a'], vec![b'b'], vec![b'c'], vec![b'd']];
        let merges = vec![(0, 1), (2, 3), (4, 5)];

        let (encoder, token_bytes) = BytePairEncoder::from_merges(&merges, &base_tokens);
        let decoder = Decoder::new(token_bytes);

        for text in [b"abcd".as_slice(), b"ab", b"cd", b"abcdabcd", b"a", b""] {
            let encoded = encoder.encode(text);
            let decoded = decoder.decode(&encoded);
            assert_eq!(decoded, text);
        }
    }
}
