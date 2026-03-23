//! SentencePiece BPE encoder.
//!
//! This module implements Byte Pair Encoding (BPE) for SentencePiece-style tokenizers
//! such as Mistral, LLaMA, and Gemma.
//!
//! # Algorithm
//!
//! The encoder uses a Radix Heap data structure for O(1) amortized operations,
//! exploiting the monotonic property of BPE: once we process merge rank k, we
//! never need ranks < k again.
//!
//! ## Steps
//!
//! 1. Split input into character-level symbols (UTF-8 aware)
//! 2. Add all adjacent pairs to radix heap ordered by merge rank
//! 3. Pop lowest-rank entry, merge symbols, add new pairs
//! 4. Repeat until no mergeable pairs remain
//!
//! # Features
//!
//! - **Zero-allocation encoding** via [`EncodeState`] for repeated calls
//! - **Parallel chunking** via [`SentencePieceBPE::encode_chunked`] using memchunk
//! - **Cache-friendly** processing for large documents
//!
//! # Example
//!
//! ```ignore
//! use tokie::encoder::SentencePieceBPE;
//!
//! let vocab = vec![
//!     (0, b"a".to_vec()),
//!     (1, b"b".to_vec()),
//!     (2, b"ab".to_vec()),
//! ];
//! let merges = vec![(0, 1)]; // a + b -> ab
//!
//! let (encoder, _) = SentencePieceBPE::from_vocab_and_merges(&vocab, &merges, 2);
//! assert_eq!(encoder.encode(b"ab"), vec![2]);
//! ```

use foldhash::HashMap as FoldHashMap;
use memchunk::chunk;

use crate::types::TokenId;

// =============================================================================
// Internal Constants
// =============================================================================

/// Sentinel value for "no link" in the symbol linked list.
const NONE: u32 = u32::MAX;

/// Metaspace character (▁) in UTF-8: E2 96 81.
/// Used for chunking at word boundaries in SentencePiece tokenizers.
const METASPACE: [u8; 3] = [0xE2, 0x96, 0x81];

// =============================================================================
// Helper Functions
// =============================================================================

/// Pack two u32 token IDs into a single u64 for efficient hash lookups.
#[inline(always)]
fn pack_pair(left: TokenId, right: TokenId) -> u64 {
    ((left as u64) << 32) | (right as u64)
}

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

// =============================================================================
// Symbol (Internal)
// =============================================================================

/// A symbol in the linked list during BPE encoding.
///
/// Symbols form a doubly-linked list that gets progressively merged.
/// When two symbols merge, the right one is marked with `len = 0`.
#[derive(Clone, Copy)]
struct Symbol {
    /// Token ID for this symbol.
    token: TokenId,
    /// Index of previous symbol (NONE if first).
    prev: u32,
    /// Index of next symbol (NONE if last).
    next: u32,
    /// Byte length of this symbol's piece (0 = merged away).
    len: u16,
}

// =============================================================================
// HeapEntry (Internal)
// =============================================================================

/// An entry in the radix heap representing a potential merge.
#[derive(Clone, Copy)]
struct HeapEntry {
    /// Composite key: `(rank << 32) | left_index`.
    /// This provides automatic ordering by rank, then by position (leftmost wins ties).
    key: u64,
    /// Index of the right symbol in the pair.
    right: u32,
    /// Combined byte length at insertion time (for staleness detection).
    size: u32,
}

impl HeapEntry {
    /// Create a new heap entry.
    #[inline(always)]
    fn new(rank: u32, left: u32, right: u32, size: u32) -> Self {
        Self {
            key: ((rank as u64) << 32) | (left as u64),
            right,
            size,
        }
    }

    /// Extract the left symbol index from the composite key.
    #[inline(always)]
    fn left(&self) -> u32 {
        self.key as u32
    }

    /// Extract the merge rank from the composite key.
    #[cfg(test)]
    #[inline(always)]
    fn rank(&self) -> u32 {
        (self.key >> 32) as u32
    }
}

// =============================================================================
// RadixHeap (Internal)
// =============================================================================

/// A radix heap optimized for monotonic u64 keys.
///
/// The radix heap exploits the property that BPE merge ranks are processed
/// in increasing order. Once we pop rank k, we never see rank < k again.
/// This gives O(1) amortized push and pop operations.
///
/// Keys are composite: `(rank << 32) | position`, ensuring that ties
/// (same rank) are broken by position (leftmost first).
struct RadixHeap {
    /// Buckets indexed by MSB difference from `last_min`.
    /// - `buckets[0]`: entries with key == last_min
    /// - `buckets[i]`: entries where MSB(key XOR last_min) == i-1
    buckets: [Vec<HeapEntry>; 65],
    /// Last minimum key that was popped.
    last_min: u64,
    /// Total number of entries.
    len: usize,
}

impl RadixHeap {
    /// Create an empty radix heap.
    fn new() -> Self {
        Self {
            buckets: std::array::from_fn(|_| Vec::new()),
            last_min: 0,
            len: 0,
        }
    }

    /// Compute bucket index for a given key.
    #[inline]
    fn bucket_index(&self, key: u64) -> usize {
        if key == self.last_min {
            0
        } else {
            let diff = key ^ self.last_min;
            (64 - diff.leading_zeros()) as usize
        }
    }

    /// Push an entry onto the heap.
    #[inline]
    fn push(&mut self, entry: HeapEntry) {
        let idx = self.bucket_index(entry.key);
        self.buckets[idx].push(entry);
        self.len += 1;
    }

    /// Pop the minimum entry from the heap.
    fn pop(&mut self) -> Option<HeapEntry> {
        if self.len == 0 {
            return None;
        }

        // Find first non-empty bucket
        let mut bucket_idx = 0;
        while bucket_idx < 65 && self.buckets[bucket_idx].is_empty() {
            bucket_idx += 1;
        }

        if bucket_idx >= 65 {
            return None;
        }

        // Bucket 0 contains exact matches - just pop any
        if bucket_idx == 0 {
            self.len -= 1;
            return self.buckets[0].pop();
        }

        // Find minimum in this bucket
        let bucket = &mut self.buckets[bucket_idx];
        let mut min_idx = 0;
        let mut min_key = bucket[0].key;
        for (i, entry) in bucket.iter().enumerate().skip(1) {
            if entry.key < min_key {
                min_key = entry.key;
                min_idx = i;
            }
        }

        // Update last_min and extract minimum entry
        self.last_min = min_key;
        let min_entry = bucket.swap_remove(min_idx);

        // Redistribute remaining entries
        let entries: Vec<HeapEntry> = std::mem::take(bucket);
        for entry in entries {
            let new_idx = self.bucket_index(entry.key);
            self.buckets[new_idx].push(entry);
        }

        self.len -= 1;
        Some(min_entry)
    }

    /// Clear the heap for reuse, preserving allocated capacity.
    fn clear(&mut self) {
        for bucket in &mut self.buckets {
            bucket.clear();
        }
        self.last_min = 0;
        self.len = 0;
    }
}

// =============================================================================
// EncodeState (Public)
// =============================================================================

/// Pre-allocated encoding state for zero-allocation encoding.
///
/// Reuse this across multiple [`SentencePieceBPE::encode_with_state`] calls
/// to avoid repeated memory allocations. After the first call, the internal
/// buffers are warmed up and subsequent calls incur no allocations.
///
/// # Example
///
/// ```ignore
/// use tokie::encoder::{SentencePieceBPE, EncodeState};
///
/// let encoder = /* ... */;
/// let mut state = EncodeState::new();
///
/// // First call warms up buffers
/// let tokens1 = encoder.encode_with_state(b"hello", &mut state);
///
/// // Subsequent calls reuse buffers - zero allocation
/// let tokens2 = encoder.encode_with_state(b"world", &mut state);
/// ```
pub struct EncodeState {
    symbols: Vec<Symbol>,
    heap: RadixHeap,
    result: Vec<TokenId>,
}

impl EncodeState {
    /// Create a new encoding state with default capacity.
    pub fn new() -> Self {
        Self {
            symbols: Vec::new(),
            heap: RadixHeap::new(),
            result: Vec::new(),
        }
    }

    /// Create a new encoding state with pre-allocated capacity.
    ///
    /// Use this if you know the approximate input size to avoid
    /// reallocation during the first encoding call.
    pub fn with_capacity(text_len: usize) -> Self {
        Self {
            symbols: Vec::with_capacity(text_len),
            heap: RadixHeap::new(),
            result: Vec::with_capacity(text_len / 4),
        }
    }

    /// Clear the state for reuse.
    fn clear(&mut self) {
        self.symbols.clear();
        self.heap.clear();
        self.result.clear();
    }
}

impl Default for EncodeState {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// SentencePieceBPE (Public)
// =============================================================================

/// SentencePiece BPE encoder using Radix Heap.
///
/// This encoder implements the BPE algorithm used by SentencePiece tokenizers
/// such as Mistral, LLaMA, and Gemma. It produces identical output to
/// HuggingFace's tokenizers library.
///
/// # Performance
///
/// - **Sequential**: ~1.6x faster than HuggingFace tokenizers
/// - **Parallel**: ~7.7x faster than HuggingFace tokenizers (8 threads)
///
/// # Creating an Encoder
///
/// Use [`SentencePieceBPE::from_vocab_and_merges`] to create an encoder from
/// vocabulary and merge rules extracted from a tokenizer.json file.
///
/// # Encoding Methods
///
/// | Method | Allocations | Use Case |
/// |--------|-------------|----------|
/// | [`encode`](Self::encode) | Per call | Simple, one-off encoding |
/// | [`encode_with_state`](Self::encode_with_state) | First call only | Repeated encoding |
/// | [`encode_chunked`](Self::encode_chunked) | Per call | Large documents with parallelization |
#[derive(Clone)]
pub struct SentencePieceBPE {
    /// Merge pair (packed) → (merged_token_id, merge_rank).
    pair_lookup: FoldHashMap<u64, (TokenId, u32)>,
    /// Maximum merge rank (for bounds checking).
    max_rank: u32,
    /// Number of base tokens (typically 256 for byte-level).
    num_base_tokens: usize,
    /// Total vocabulary size.
    vocab_size: usize,
    /// Bytes → token ID mapping for character/token lookup.
    token_cache: FoldHashMap<Vec<u8>, TokenId>,
    /// Single byte → token ID mapping for fallback.
    byte_lut: [TokenId; 256],
    /// Token ID → byte length mapping.
    token_lengths: Vec<u16>,
}

impl std::fmt::Debug for SentencePieceBPE {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SentencePieceBPE")
            .field("vocab_size", &self.vocab_size)
            .field("num_base_tokens", &self.num_base_tokens)
            .field("merges", &self.pair_lookup.len())
            .field("max_rank", &self.max_rank)
            .finish()
    }
}

impl SentencePieceBPE {
    /// Create an encoder from vocabulary and merge rules.
    ///
    /// # Arguments
    ///
    /// * `vocab` - Token ID and byte sequence pairs, sorted by ID
    /// * `merges` - Merge rules as (left_token, right_token) pairs, in priority order
    /// * `num_base_tokens` - Number of base tokens (typically 256)
    ///
    /// # Returns
    ///
    /// A tuple of (encoder, token_bytes) where token_bytes maps token IDs to byte sequences.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let vocab = vec![
    ///     (0, b"a".to_vec()),
    ///     (1, b"b".to_vec()),
    ///     (2, b"ab".to_vec()),
    /// ];
    /// let merges = vec![(0, 1)]; // a + b -> ab
    ///
    /// let (encoder, token_bytes) = SentencePieceBPE::from_vocab_and_merges(&vocab, &merges, 2);
    /// ```
    pub fn from_vocab_and_merges(
        vocab: &[(u32, Vec<u8>)],
        merges: &[(TokenId, TokenId)],
        num_base_tokens: usize,
    ) -> (Self, Vec<Vec<u8>>) {
        let token_bytes: Vec<Vec<u8>> = vocab.iter().map(|(_, bytes)| bytes.clone()).collect();

        let bytes_to_id: FoldHashMap<Vec<u8>, TokenId> = vocab
            .iter()
            .map(|(id, bytes)| (bytes.clone(), *id))
            .collect();

        // Build pair_lookup with merge ranks
        let mut pair_lookup: FoldHashMap<u64, (TokenId, u32)> = FoldHashMap::default();
        let mut max_rank = 0u32;

        for (merge_rank, &(left, right)) in merges.iter().enumerate() {
            let mut merged_bytes = token_bytes[left as usize].clone();
            merged_bytes.extend_from_slice(&token_bytes[right as usize]);

            if let Some(&merged_id) = bytes_to_id.get(&merged_bytes) {
                pair_lookup
                    .entry(pack_pair(left, right))
                    .or_insert((merged_id, merge_rank as u32));
                max_rank = max_rank.max(merge_rank as u32);
            }
        }

        // Build single-byte fallback table
        let mut byte_lut = [u32::MAX; 256];
        for (id, bytes) in vocab {
            if bytes.len() == 1 {
                let byte_val = bytes[0] as usize;
                if byte_lut[byte_val] == u32::MAX {
                    byte_lut[byte_val] = *id;
                }
            }
        }

        let token_lengths: Vec<u16> = token_bytes.iter().map(|b| b.len() as u16).collect();

        let token_cache: FoldHashMap<Vec<u8>, TokenId> = vocab
            .iter()
            .map(|(id, bytes)| (bytes.clone(), *id))
            .collect();

        let encoder = Self {
            pair_lookup,
            max_rank,
            num_base_tokens,
            vocab_size: vocab.len(),
            token_cache,
            byte_lut,
            token_lengths,
        };

        (encoder, token_bytes)
    }

    /// Get the vocabulary size.
    #[inline]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get the number of base tokens.
    #[inline]
    pub fn num_base_tokens(&self) -> usize {
        self.num_base_tokens
    }

    /// Get access to the pair lookup table (for serialization).
    #[inline]
    pub fn pair_lookup(&self) -> &FoldHashMap<u64, (TokenId, u32)> {
        &self.pair_lookup
    }

    /// Reconstruct encoder from serialized parts with pre-built lookups.
    ///
    /// Used during deserialization to rebuild the encoder from saved data.
    /// All lookups (byte_lut, token_cache, token_lengths) are pre-built from decoder
    /// data, avoiding intermediate Vec<Vec<u8>> allocation.
    pub fn from_parts(
        merges: &[(TokenId, TokenId, TokenId)], // (left, right, merged_id)
        byte_lut: [TokenId; 256],
        token_cache: FoldHashMap<Vec<u8>, TokenId>,
        token_lengths: Vec<u16>,
        vocab_size: usize,
        num_base_tokens: usize,
    ) -> Self {
        // Build pair_lookup directly from pre-computed merged IDs - O(num_merges)
        let mut pair_lookup: FoldHashMap<u64, (TokenId, u32)> = FoldHashMap::default();
        let mut max_rank = 0u32;

        for (merge_rank, &(left, right, merged_id)) in merges.iter().enumerate() {
            pair_lookup
                .entry(pack_pair(left, right))
                .or_insert((merged_id, merge_rank as u32));
            max_rank = max_rank.max(merge_rank as u32);
        }

        Self {
            pair_lookup,
            max_rank,
            num_base_tokens,
            vocab_size,
            token_cache,
            byte_lut,
            token_lengths,
        }
    }

    /// Check if two tokens can appear adjacent in a valid BPE encoding.
    ///
    /// For SentencePiece, this always returns true since any adjacency is valid.
    #[inline]
    pub fn is_valid_pair(&self, _token1: TokenId, _token2: TokenId) -> bool {
        true
    }

    /// Look up merge information for a token pair.
    #[inline]
    fn get_merge(&self, left: TokenId, right: TokenId) -> Option<(TokenId, u32)> {
        self.pair_lookup.get(&pack_pair(left, right)).copied()
    }

    /// Encode text to token IDs.
    ///
    /// This is the main encoding method. For repeated encoding of many strings,
    /// consider using [`encode_with_state`](Self::encode_with_state) to avoid
    /// repeated allocations.
    ///
    /// # Arguments
    ///
    /// * `text` - UTF-8 encoded input bytes
    ///
    /// # Returns
    ///
    /// A vector of token IDs.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let tokens = encoder.encode(b"hello world");
    /// ```
    pub fn encode(&self, text: &[u8]) -> Vec<TokenId> {
        if text.is_empty() {
            return Vec::new();
        }

        // Fast path: check if entire text is a single known token
        if let Some(&token_id) = self.token_cache.get(text) {
            return vec![token_id];
        }

        // Initialize symbols from input
        let mut symbols = self.init_symbols(text);
        if symbols.is_empty() {
            return Vec::new();
        }

        // Build initial heap
        let mut heap = RadixHeap::new();
        self.init_heap(&symbols, &mut heap);

        // Run merge loop
        self.merge_loop(&mut symbols, &mut heap);

        // Collect results
        self.collect_results(&symbols)
    }

    /// Encode text to token IDs using pre-allocated state.
    ///
    /// This method reuses the provided state buffers, avoiding allocations
    /// after the first call. Returns a slice into the state's result buffer.
    ///
    /// # Arguments
    ///
    /// * `text` - UTF-8 encoded input bytes
    /// * `state` - Pre-allocated encoding state for buffer reuse
    ///
    /// # Returns
    ///
    /// A slice of token IDs (borrowed from the state).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut state = EncodeState::new();
    ///
    /// // Each call reuses the same buffers
    /// for text in texts {
    ///     let tokens = encoder.encode_with_state(text.as_bytes(), &mut state);
    ///     process(tokens);
    /// }
    /// ```
    pub fn encode_with_state<'a>(&self, text: &[u8], state: &'a mut EncodeState) -> &'a [TokenId] {
        state.clear();

        if text.is_empty() {
            return &state.result;
        }

        // Fast path: check if entire text is a single known token
        if let Some(&token_id) = self.token_cache.get(text) {
            state.result.push(token_id);
            return &state.result;
        }

        // Initialize symbols
        self.init_symbols_into(text, &mut state.symbols);
        if state.symbols.is_empty() {
            return &state.result;
        }

        // Build initial heap
        self.init_heap(&state.symbols, &mut state.heap);

        // Run merge loop
        self.merge_loop(&mut state.symbols, &mut state.heap);

        // Collect results
        self.collect_results_into(&state.symbols, &mut state.result);

        &state.result
    }

    /// Encode text using chunked processing for better cache locality.
    ///
    /// For large texts, this splits at metaspace (▁) boundaries and encodes
    /// each chunk separately. This improves L1/L2 cache utilization.
    ///
    /// Uses memchunk with `consecutive()` to split at the START of metaspace
    /// runs (preserving `▁▁▁` sequences) and `forward_fallback()` to avoid
    /// mid-word splits.
    ///
    /// # Arguments
    ///
    /// * `text` - UTF-8 encoded input bytes
    /// * `state` - Pre-allocated state for buffer reuse
    /// * `chunk_size` - Target chunk size in bytes (will align to metaspace boundaries)
    ///
    /// # Returns
    ///
    /// A vector of token IDs.
    pub fn encode_chunked(
        &self,
        text: &[u8],
        state: &mut EncodeState,
        chunk_size: usize,
    ) -> Vec<TokenId> {
        if text.len() <= chunk_size {
            return self.encode_with_state(text, state).to_vec();
        }

        let mut result = Vec::with_capacity(text.len() / 4);

        for chunk_bytes in chunk(text)
            .size(chunk_size)
            .pattern(&METASPACE)
            .prefix()
            .consecutive()
            .forward_fallback()
        {
            let chunk_tokens = self.encode_with_state(chunk_bytes, state);
            result.extend_from_slice(chunk_tokens);
        }

        result
    }

    // -------------------------------------------------------------------------
    // Private Helper Methods
    // -------------------------------------------------------------------------

    /// Initialize symbols from input text.
    fn init_symbols(&self, text: &[u8]) -> Vec<Symbol> {
        let mut symbols = Vec::with_capacity(text.len());
        self.init_symbols_into(text, &mut symbols);
        symbols
    }

    /// Initialize symbols into an existing vector.
    fn init_symbols_into(&self, text: &[u8], symbols: &mut Vec<Symbol>) {
        let mut pos = 0;

        while pos < text.len() {
            let char_len = utf8_char_len(text[pos]);
            let end = (pos + char_len).min(text.len());
            let char_bytes = &text[pos..end];

            let (token, len) = if let Some(&token_id) = self.token_cache.get(char_bytes) {
                (token_id, char_bytes.len())
            } else {
                let byte_token = self.byte_lut[text[pos] as usize];
                (byte_token, 1)
            };

            if token != u32::MAX {
                let idx = symbols.len() as u32;
                symbols.push(Symbol {
                    token,
                    prev: if idx == 0 { NONE } else { idx - 1 },
                    next: NONE,
                    len: self.token_lengths.get(token as usize).copied().unwrap_or(len as u16),
                });
                if idx > 0 {
                    symbols[(idx - 1) as usize].next = idx;
                }
            }

            pos += len;
        }
    }

    /// Initialize the heap with all mergeable adjacent pairs.
    fn init_heap(&self, symbols: &[Symbol], heap: &mut RadixHeap) {
        for i in 0..symbols.len().saturating_sub(1) {
            let left_sym = &symbols[i];
            let right_sym = &symbols[i + 1];

            if let Some((_, rank)) = self.get_merge(left_sym.token, right_sym.token) {
                heap.push(HeapEntry::new(
                    rank,
                    i as u32,
                    (i + 1) as u32,
                    left_sym.len as u32 + right_sym.len as u32,
                ));
            }
        }
    }

    /// Run the main merge loop.
    fn merge_loop(&self, symbols: &mut [Symbol], heap: &mut RadixHeap) {
        while let Some(entry) = heap.pop() {
            let left_idx = entry.left() as usize;
            let right_idx = entry.right as usize;

            // Validate entry is still current
            let left = &symbols[left_idx];
            let right = &symbols[right_idx];

            if left.len == 0 || right.len == 0 {
                continue; // Symbol was merged away
            }
            if left.next != entry.right {
                continue; // Symbols no longer adjacent
            }
            if (left.len as u32 + right.len as u32) != entry.size {
                continue; // Stale entry
            }

            // Perform merge
            let (merged_token, _) = self.get_merge(left.token, right.token).unwrap();
            let new_len = left.len + right.len;
            let right_next = right.next;

            symbols[left_idx].token = merged_token;
            symbols[left_idx].len = new_len;
            symbols[left_idx].next = right_next;
            symbols[right_idx].len = 0;

            if right_next != NONE {
                symbols[right_next as usize].prev = entry.left();
            }

            // Add new pairs
            let left_prev = symbols[left_idx].prev;
            if left_prev != NONE {
                let prev = &symbols[left_prev as usize];
                if prev.len > 0 {
                    if let Some((_, rank)) = self.get_merge(prev.token, merged_token) {
                        heap.push(HeapEntry::new(
                            rank,
                            left_prev,
                            entry.left(),
                            prev.len as u32 + new_len as u32,
                        ));
                    }
                }
            }

            if right_next != NONE {
                let next = &symbols[right_next as usize];
                if next.len > 0 {
                    if let Some((_, rank)) = self.get_merge(merged_token, next.token) {
                        heap.push(HeapEntry::new(
                            rank,
                            entry.left(),
                            right_next,
                            new_len as u32 + next.len as u32,
                        ));
                    }
                }
            }
        }
    }

    /// Collect results by traversing the linked list.
    fn collect_results(&self, symbols: &[Symbol]) -> Vec<TokenId> {
        let mut result = Vec::new();
        self.collect_results_into(symbols, &mut result);
        result
    }

    /// Collect results into an existing vector.
    fn collect_results_into(&self, symbols: &[Symbol], result: &mut Vec<TokenId>) {
        let mut idx = 0u32;
        while idx != NONE && (idx as usize) < symbols.len() {
            let sym = &symbols[idx as usize];
            if sym.len > 0 {
                result.push(sym.token);
            }
            idx = sym.next;
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

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
        let merges = vec![(0, 1)];

        let (encoder, _) = SentencePieceBPE::from_vocab_and_merges(&vocab, &merges, 2);

        assert_eq!(encoder.encode(b"ab"), vec![2]);
        assert_eq!(encoder.encode(b"a"), vec![0]);
        assert_eq!(encoder.encode(b"ba"), vec![1, 0]);
    }

    #[test]
    fn test_merge_rank_priority() {
        let vocab = vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"c".to_vec()),
            (3, b"ab".to_vec()),
            (4, b"bc".to_vec()),
            (5, b"abc".to_vec()),
        ];
        let merges = vec![
            (0, 1), // a + b -> ab (rank 0)
            (3, 2), // ab + c -> abc (rank 1)
            (1, 2), // b + c -> bc (rank 2)
        ];

        let (encoder, _) = SentencePieceBPE::from_vocab_and_merges(&vocab, &merges, 3);

        assert_eq!(encoder.encode(b"abc"), vec![5]);
    }

    #[test]
    fn test_unicode_char() {
        let vocab = vec![
            (0, "▁".as_bytes().to_vec()),
            (1, "H".as_bytes().to_vec()),
            (2, "▁H".as_bytes().to_vec()),
        ];
        let merges = vec![(0, 1)];

        let (encoder, _) = SentencePieceBPE::from_vocab_and_merges(&vocab, &merges, 2);

        assert_eq!(encoder.encode("▁H".as_bytes()), vec![2]);
        assert_eq!(encoder.encode("▁".as_bytes()), vec![0]);
    }

    #[test]
    fn test_radix_heap() {
        let mut heap = RadixHeap::new();

        heap.push(HeapEntry::new(5, 0, 1, 2));
        heap.push(HeapEntry::new(3, 1, 2, 2));
        heap.push(HeapEntry::new(7, 2, 3, 2));
        heap.push(HeapEntry::new(1, 3, 4, 2));

        assert_eq!(heap.pop().unwrap().rank(), 1);
        assert_eq!(heap.pop().unwrap().rank(), 3);
        assert_eq!(heap.pop().unwrap().rank(), 5);
        assert_eq!(heap.pop().unwrap().rank(), 7);
        assert!(heap.pop().is_none());
    }

    #[test]
    fn test_radix_heap_tie_breaking() {
        let mut heap = RadixHeap::new();

        // Same rank, different positions - should break ties by position
        heap.push(HeapEntry::new(5, 3, 4, 2));
        heap.push(HeapEntry::new(5, 1, 2, 2));
        heap.push(HeapEntry::new(5, 2, 3, 2));

        assert_eq!(heap.pop().unwrap().left(), 1);
        assert_eq!(heap.pop().unwrap().left(), 2);
        assert_eq!(heap.pop().unwrap().left(), 3);
        assert!(heap.pop().is_none());
    }

    #[test]
    fn test_encode_with_state() {
        let vocab = vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"ab".to_vec()),
        ];
        let merges = vec![(0, 1)];

        let (encoder, _) = SentencePieceBPE::from_vocab_and_merges(&vocab, &merges, 2);
        let mut state = EncodeState::new();

        assert_eq!(encoder.encode_with_state(b"ab", &mut state), &[2]);
        assert_eq!(encoder.encode_with_state(b"a", &mut state), &[0]);
        assert_eq!(encoder.encode_with_state(b"ba", &mut state), &[1, 0]);

        // Test reuse
        for _ in 0..5 {
            assert_eq!(encoder.encode_with_state(b"ab", &mut state), &[2]);
        }
    }

    #[test]
    fn test_encode_chunked() {
        let vocab = vec![
            (0, "▁".as_bytes().to_vec()),
            (1, "a".as_bytes().to_vec()),
            (2, "b".as_bytes().to_vec()),
            (3, "c".as_bytes().to_vec()),
            (4, "▁a".as_bytes().to_vec()),
            (5, "▁ab".as_bytes().to_vec()),
        ];
        let merges = vec![
            (0, 1), // ▁ + a -> ▁a
            (4, 2), // ▁a + b -> ▁ab
        ];

        let (encoder, _) = SentencePieceBPE::from_vocab_and_merges(&vocab, &merges, 4);
        let mut state = EncodeState::new();

        let text = "▁ab▁ab▁ab".as_bytes();
        let regular = encoder.encode(text);
        let chunked = encoder.encode_chunked(text, &mut state, 6);

        assert_eq!(regular, chunked);
    }
}
