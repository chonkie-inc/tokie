//! Binary serialization for fast tokenizer loading.
//!
//! This module provides efficient save/load functionality using a custom binary format
//! that stores pre-built DAAC state, eliminating the need to rebuild the automaton.
//!
//! # File Format
//!
//! ```text
//! Header (88 bytes):
//!   - magic: "TOKI" (4 bytes)
//!   - version: u32 (4 bytes) - currently v9
//!   - encoder_type: u32 (4 bytes) - 0=Backtracking, 1=Simple, 2=WordPiece
//!   - pretokenizer_type: u32 (4 bytes) - 0=None, 1=GPT2, 2=CL100K, 3=O200K, 4=BERT, 5=Voyage
//!   - normalizer_type: u32 (4 bytes) - 0=None, 1=BertUncased, 2=BertCased, 3=Nfc
//!   - post_processor_type: u32 (4 bytes) - 0=None, 1=Bert, 2=Prefix, 3=Template
//!   - vocab_size: u32 (4 bytes)
//!   - num_merges: u32 (4 bytes)
//!   - num_base_tokens: u32 (4 bytes)
//!   - reserved: u32 (4 bytes)
//!   - token_data_offset: u32, token_data_checksum: u32
//!   - merge_data_offset: u32, merge_data_checksum: u32
//!   - daac_data_offset: u32, daac_data_checksum: u32
//!   - prefix_data_offset: u32, prefix_data_checksum: u32
//!   - pp_data_offset: u32, pp_data_checksum: u32
//!
//! Sections:
//!   - TOKEN_DATA: Decoder's flat buffer (offsets + data)
//!   - MERGE_DATA: split_table as raw bytes
//!   - DAAC_DATA: Pre-built DoubleArrayAhoCorasick state (empty for Simple encoder)
//!   - PREFIX_DATA: next_prefix_match table (empty for Simple encoder)
//!   - PP_DATA: Post-processor parameters (empty for None)
//! ```

use core::mem::size_of;
use std::io::{Read, Write};

use crate::encoder::{BacktrackingBytePairEncoder, BytePairEncoder, Encoder, EncoderType, SentencePieceBPE, UnigramEncoder, WordPieceEncoder};
use crate::decoder::Decoder;
use crate::normalizer::Normalizer;
use crate::postprocessor::PostProcessor;
use crate::pretok::PretokType;
use crate::tokenizer::Tokenizer;
use crate::types::{Split, TokenId};
use daggrs::DoubleArrayAhoCorasick;
use foldhash::HashMap as FoldHashMap;

const MAGIC: &[u8; 4] = b"TOKI";
const VERSION: u32 = 10; // v10 adds merged_id to merge data for Simple/SentencePiece
const HEADER_SIZE: usize = 88;

impl PretokType {
    fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::None),
            1 => Some(Self::Gpt2),
            2 => Some(Self::Cl100k),
            3 => Some(Self::O200k),
            4 => Some(Self::Bert),
            5 => Some(Self::Voyage),
            _ => None,
        }
    }
}

impl Normalizer {
    fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::None),
            1 => Some(Self::BertUncased),
            2 => Some(Self::BertCased),
            3 => Some(Self::Nfc),
            4 => Some(Self::Metaspace),
            5 => Some(Self::SentencePiece),
            6 => Some(Self::SentencePieceLowercase),
            7 => Some(Self::MetaspaceReplace),
            _ => None,
        }
    }

    fn to_u32(&self) -> u32 {
        match self {
            Self::None => 0,
            Self::BertUncased => 1,
            Self::BertCased => 2,
            Self::Nfc => 3,
            Self::Metaspace => 4,
            Self::SentencePiece => 5,
            Self::SentencePieceLowercase => 6,
            Self::MetaspaceReplace => 7,
        }
    }
}

impl PostProcessor {
    fn type_id(&self) -> u32 {
        match self {
            Self::None => 0,
            Self::Bert { .. } => 1,
            Self::Prefix { .. } => 2,
            Self::Template { .. } => 3,
        }
    }

    fn serialize(&self) -> Vec<u8> {
        match self {
            Self::None => Vec::new(),
            Self::Bert { cls_token, sep_token } => {
                let mut buf = Vec::with_capacity(8);
                buf.extend_from_slice(&cls_token.to_le_bytes());
                buf.extend_from_slice(&sep_token.to_le_bytes());
                buf
            }
            Self::Prefix { bos_token } => {
                bos_token.to_le_bytes().to_vec()
            }
            Self::Template {
                single_prefix,
                single_suffix,
                pair_a_prefix,
                pair_a_suffix,
                pair_b_prefix,
                pair_b_suffix,
            } => {
                // Format: 6 length-prefixed arrays of u32 tokens
                let mut buf = Vec::new();
                for tokens in [
                    single_prefix,
                    single_suffix,
                    pair_a_prefix,
                    pair_a_suffix,
                    pair_b_prefix,
                    pair_b_suffix,
                ] {
                    buf.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
                    for &token in tokens {
                        buf.extend_from_slice(&token.to_le_bytes());
                    }
                }
                buf
            }
        }
    }

    fn deserialize(type_id: u32, data: &[u8]) -> Option<Self> {
        match type_id {
            0 => Some(Self::None),
            1 => {
                if data.len() < 8 {
                    return None;
                }
                let cls_token = u32::from_le_bytes(data[0..4].try_into().ok()?);
                let sep_token = u32::from_le_bytes(data[4..8].try_into().ok()?);
                Some(Self::Bert { cls_token, sep_token })
            }
            2 => {
                if data.len() < 4 {
                    return None;
                }
                let bos_token = u32::from_le_bytes(data[0..4].try_into().ok()?);
                Some(Self::Prefix { bos_token })
            }
            3 => {
                // Parse 6 length-prefixed arrays
                let mut offset = 0;
                let mut arrays = Vec::new();
                for _ in 0..6 {
                    if offset + 4 > data.len() {
                        return None;
                    }
                    let len = u32::from_le_bytes(data[offset..offset + 4].try_into().ok()?) as usize;
                    offset += 4;
                    let mut tokens = Vec::with_capacity(len);
                    for _ in 0..len {
                        if offset + 4 > data.len() {
                            return None;
                        }
                        tokens.push(u32::from_le_bytes(data[offset..offset + 4].try_into().ok()?));
                        offset += 4;
                    }
                    arrays.push(tokens);
                }
                Some(Self::Template {
                    single_prefix: arrays.remove(0),
                    single_suffix: arrays.remove(0),
                    pair_a_prefix: arrays.remove(0),
                    pair_a_suffix: arrays.remove(0),
                    pair_b_prefix: arrays.remove(0),
                    pair_b_suffix: arrays.remove(0),
                })
            }
            _ => None,
        }
    }
}

/// Fast CRC32 checksum using hardware acceleration when available.
fn crc32(data: &[u8]) -> u32 {
    crc32fast::hash(data)
}

/// Error type for serialization/deserialization.
#[derive(Debug)]
pub enum SerdeError {
    Io(std::io::Error),
    InvalidMagic,
    UnsupportedVersion(u32),
    InvalidEncoderType(u32),
    InvalidPretokenizer(u32),
    InvalidNormalizer(u32),
    InvalidPostProcessor(u32),
    ChecksumMismatch { section: &'static str },
    InvalidData(&'static str),
}

impl std::fmt::Display for SerdeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {}", e),
            Self::InvalidMagic => write!(f, "Invalid magic bytes (not a TOKI file)"),
            Self::UnsupportedVersion(v) => write!(f, "Unsupported version: {}", v),
            Self::InvalidEncoderType(v) => write!(f, "Invalid encoder type: {}", v),
            Self::InvalidPretokenizer(v) => write!(f, "Invalid pretokenizer type: {}", v),
            Self::InvalidNormalizer(v) => write!(f, "Invalid normalizer type: {}", v),
            Self::InvalidPostProcessor(v) => write!(f, "Invalid post-processor type: {}", v),
            Self::ChecksumMismatch { section } => write!(f, "Checksum mismatch in {}", section),
            Self::InvalidData(msg) => write!(f, "Invalid data: {}", msg),
        }
    }
}

impl std::error::Error for SerdeError {}

impl From<std::io::Error> for SerdeError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl Tokenizer {
    /// Save the tokenizer to a file.
    ///
    /// This saves the pre-built DAAC state, enabling fast loading without
    /// rebuilding the automaton.
    pub fn to_file(&self, path: impl AsRef<std::path::Path>) -> Result<(), SerdeError> {
        let file = std::fs::File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);
        self.save(&mut writer)
    }

    /// Save the tokenizer to a writer.
    pub fn save<W: Write>(&self, writer: &mut W) -> Result<(), SerdeError> {
        let encoder_type = self.encoder_type();
        let pretokenizer_type = self.pretokenizer_type();
        let normalizer = self.normalizer();
        let post_processor = self.post_processor();
        let encoder = self.encoder();
        let decoder = self.decoder();

        // Serialize sections based on encoder type
        let token_data = serialize_decoder(decoder);

        // Serialize sections based on encoder type
        let (merge_data, daac_data, prefix_data) = match encoder {
            Encoder::Backtracking(enc) => {
                let merge = serialize_splits(enc.split_table());
                let daac = enc.matcher().serialize();
                let prefix = serialize_prefix_match(enc.next_prefix_match_table());
                (merge, daac, prefix)
            }
            Encoder::Simple(enc) => {
                // Simple encoder: serialize pair_lookup as merges, empty DAAC/prefix
                let merge = serialize_pair_lookup(enc);
                let daac = Vec::new();
                let prefix = Vec::new();
                (merge, daac, prefix)
            }
            Encoder::WordPiece(enc) => {
                // WordPiece: serialize DAAC with anchor, empty merge/prefix
                let merge = serialize_wordpiece_config(enc);
                let daac = enc.matcher().serialize();
                let prefix = Vec::new();
                (merge, daac, prefix)
            }
            Encoder::SentencePiece(enc) => {
                // SentencePiece: serialize pair_lookup as merges, empty DAAC/prefix
                let merge = serialize_sentencepiece_config(enc);
                let daac = Vec::new();
                let prefix = Vec::new();
                (merge, daac, prefix)
            }
            Encoder::Unigram(enc) => {
                // Unigram: serialize scores, unk_token, byte_tokens in merge_data
                // DAAC in daac_data, prefix_data empty
                let merge = serialize_unigram_config(enc);
                let daac = enc.matcher().serialize();
                let prefix = Vec::new();
                (merge, daac, prefix)
            }
        };

        // Serialize post-processor
        let pp_data = post_processor.serialize();

        // Compute checksums
        let token_checksum = crc32(&token_data);
        let merge_checksum = crc32(&merge_data);
        let daac_checksum = crc32(&daac_data);
        let prefix_checksum = crc32(&prefix_data);
        let pp_checksum = crc32(&pp_data);

        // Compute offsets (after header)
        let token_offset = HEADER_SIZE as u32;
        let merge_offset = token_offset + token_data.len() as u32;
        let daac_offset = merge_offset + merge_data.len() as u32;
        let prefix_offset = daac_offset + daac_data.len() as u32;
        let pp_offset = prefix_offset + prefix_data.len() as u32;

        // Write header (88 bytes total)
        // 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + (5 × 8) = 40 + 40 = 80... need 8 more
        // Actually: magic(4) + version(4) + encoder(4) + pretok(4) + norm(4) + pp_type(4)
        //         + vocab(4) + merges(4) + base(4) + reserved(4) = 40 bytes
        //         + 5 sections × 8 bytes = 40 bytes
        //         Total = 80 bytes... let me recalculate for 88
        // We need: 40 bytes metadata + 5 sections × 8 = 80 bytes
        // For 88: add another reserved u32 (4) + padding (4) = 88
        writer.write_all(MAGIC)?;
        writer.write_all(&VERSION.to_le_bytes())?;
        writer.write_all(&(encoder_type as u32).to_le_bytes())?;
        writer.write_all(&(pretokenizer_type as u32).to_le_bytes())?;
        writer.write_all(&normalizer.to_u32().to_le_bytes())?;
        writer.write_all(&post_processor.type_id().to_le_bytes())?;
        writer.write_all(&(decoder.vocab_size() as u32).to_le_bytes())?;
        writer.write_all(&((encoder.vocab_size() - encoder.num_base_tokens()) as u32).to_le_bytes())?;
        writer.write_all(&(encoder.num_base_tokens() as u32).to_le_bytes())?;
        writer.write_all(&0u32.to_le_bytes())?; // Reserved

        // Interleaved offsets and checksums (5 sections × 8 bytes = 40 bytes)
        writer.write_all(&token_offset.to_le_bytes())?;
        writer.write_all(&token_checksum.to_le_bytes())?;
        writer.write_all(&merge_offset.to_le_bytes())?;
        writer.write_all(&merge_checksum.to_le_bytes())?;
        writer.write_all(&daac_offset.to_le_bytes())?;
        writer.write_all(&daac_checksum.to_le_bytes())?;
        writer.write_all(&prefix_offset.to_le_bytes())?;
        writer.write_all(&prefix_checksum.to_le_bytes())?;
        writer.write_all(&pp_offset.to_le_bytes())?;
        writer.write_all(&pp_checksum.to_le_bytes())?;

        // Header total: 40 + 40 = 80 bytes... but we said 88
        // Let me add padding
        writer.write_all(&0u64.to_le_bytes())?; // 8 bytes padding to reach 88

        // Write sections
        writer.write_all(&token_data)?;
        writer.write_all(&merge_data)?;
        writer.write_all(&daac_data)?;
        writer.write_all(&prefix_data)?;
        writer.write_all(&pp_data)?;

        Ok(())
    }

    /// Load a tokenizer from a file.
    ///
    /// This loads pre-built DAAC state for instant use without rebuilding.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, SerdeError> {
        let file = std::fs::File::open(path)?;
        let mut reader = std::io::BufReader::new(file);
        Self::load(&mut reader)
    }

    /// Load a tokenizer from a reader.
    pub fn load<R: Read>(reader: &mut R) -> Result<Self, SerdeError> {
        // Read entire file
        let mut data = Vec::new();
        reader.read_to_end(&mut data)?;

        if data.len() < HEADER_SIZE {
            return Err(SerdeError::InvalidData("file too small"));
        }

        // Parse header
        if &data[0..4] != MAGIC {
            return Err(SerdeError::InvalidMagic);
        }

        let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
        if version != VERSION {
            return Err(SerdeError::UnsupportedVersion(version));
        }

        let encoder_type = u32::from_le_bytes(data[8..12].try_into().unwrap());
        let encoder_type = EncoderType::from_u32(encoder_type)
            .ok_or(SerdeError::InvalidEncoderType(encoder_type))?;

        let pretokenizer_type = u32::from_le_bytes(data[12..16].try_into().unwrap());
        let pretokenizer_type = PretokType::from_u32(pretokenizer_type)
            .ok_or(SerdeError::InvalidPretokenizer(pretokenizer_type))?;

        let normalizer_type = u32::from_le_bytes(data[16..20].try_into().unwrap());
        let normalizer = Normalizer::from_u32(normalizer_type)
            .ok_or(SerdeError::InvalidNormalizer(normalizer_type))?;

        let pp_type = u32::from_le_bytes(data[20..24].try_into().unwrap());

        let vocab_size = u32::from_le_bytes(data[24..28].try_into().unwrap()) as usize;
        let _num_merges = u32::from_le_bytes(data[28..32].try_into().unwrap()) as usize;
        let num_base_tokens = u32::from_le_bytes(data[32..36].try_into().unwrap()) as usize;
        // data[36..40] is reserved

        // Section offsets and checksums (5 sections × 8 bytes = 40 bytes)
        let token_offset = u32::from_le_bytes(data[40..44].try_into().unwrap()) as usize;
        let token_checksum = u32::from_le_bytes(data[44..48].try_into().unwrap());
        let merge_offset = u32::from_le_bytes(data[48..52].try_into().unwrap()) as usize;
        let merge_checksum = u32::from_le_bytes(data[52..56].try_into().unwrap());
        let daac_offset = u32::from_le_bytes(data[56..60].try_into().unwrap()) as usize;
        let daac_checksum = u32::from_le_bytes(data[60..64].try_into().unwrap());
        let prefix_offset = u32::from_le_bytes(data[64..68].try_into().unwrap()) as usize;
        let prefix_checksum = u32::from_le_bytes(data[68..72].try_into().unwrap());
        let pp_offset = u32::from_le_bytes(data[72..76].try_into().unwrap()) as usize;
        let pp_checksum = u32::from_le_bytes(data[76..80].try_into().unwrap());
        // data[80..88] is padding

        // Extract and verify sections
        let token_data = &data[token_offset..merge_offset];
        if crc32(token_data) != token_checksum {
            return Err(SerdeError::ChecksumMismatch { section: "token_data" });
        }

        let merge_data = &data[merge_offset..daac_offset];
        if crc32(merge_data) != merge_checksum {
            return Err(SerdeError::ChecksumMismatch { section: "merge_data" });
        }

        let daac_data = &data[daac_offset..prefix_offset];
        if crc32(daac_data) != daac_checksum {
            return Err(SerdeError::ChecksumMismatch { section: "daac_data" });
        }

        let prefix_data = &data[prefix_offset..pp_offset];
        if crc32(prefix_data) != prefix_checksum {
            return Err(SerdeError::ChecksumMismatch { section: "prefix_data" });
        }

        let pp_data = &data[pp_offset..];
        if crc32(pp_data) != pp_checksum {
            return Err(SerdeError::ChecksumMismatch { section: "pp_data" });
        }

        // Deserialize post-processor
        let post_processor = PostProcessor::deserialize(pp_type, pp_data)
            .ok_or(SerdeError::InvalidPostProcessor(pp_type))?;

        // Deserialize decoder
        let (decoder_offsets, decoder_data) = deserialize_decoder(token_data, vocab_size)?;

        // Build encoder based on type
        // OPTIMIZATION: For Simple/SentencePiece, build lookups directly from decoder
        // without intermediate Vec<Vec<u8>> allocation (4x faster for large vocabs)
        let encoder = match encoder_type {
            EncoderType::Backtracking => {
                // Backtracking still needs token_bytes for now
                let token_bytes: Vec<Vec<u8>> = (0..vocab_size)
                    .map(|i| {
                        let start = decoder_offsets[i] as usize;
                        let end = decoder_offsets[i + 1] as usize;
                        decoder_data[start..end].to_vec()
                    })
                    .collect();

                let split_table = deserialize_splits(merge_data)?;
                let (daac, _) = DoubleArrayAhoCorasick::deserialize(daac_data)
                    .ok_or(SerdeError::InvalidData("failed to deserialize DAAC"))?;
                let next_prefix_match = deserialize_prefix_match(prefix_data)?;

                // Rebuild pair_lookup from split_table
                let pair_lookup = rebuild_pair_lookup(&split_table, num_base_tokens);

                // Extract token lengths from decoder offsets
                let token_lengths: Vec<u8> = (0..vocab_size)
                    .map(|i| {
                        let start = decoder_offsets[i] as usize;
                        let end = decoder_offsets[i + 1] as usize;
                        (end - start).min(255) as u8
                    })
                    .collect();

                let enc = BacktrackingBytePairEncoder::from_parts(
                    split_table,
                    pair_lookup,
                    token_lengths,
                    num_base_tokens,
                    daac,
                    next_prefix_match,
                    &token_bytes,
                );
                Encoder::Backtracking(enc)
            }
            EncoderType::Simple => {
                // OPTIMIZED: Build lookups directly from decoder (single copy)
                // Simple encoder doesn't use token_lengths, so we ignore it
                let (byte_lut, token_cache, _) = build_token_lookups(&decoder_offsets, &decoder_data, vocab_size);
                let merges = deserialize_merges(merge_data)?;

                let enc = BytePairEncoder::from_parts(
                    &merges,
                    byte_lut,
                    token_cache,
                    vocab_size,
                    num_base_tokens,
                );
                Encoder::Simple(enc)
            }
            EncoderType::WordPiece => {
                // WordPiece needs token_bytes for continuation prefix matching
                let token_bytes: Vec<Vec<u8>> = (0..vocab_size)
                    .map(|i| {
                        let start = decoder_offsets[i] as usize;
                        let end = decoder_offsets[i + 1] as usize;
                        decoder_data[start..end].to_vec()
                    })
                    .collect();

                let (unk_token, continuation_prefix) = deserialize_wordpiece_config(merge_data)?;
                let (daac, _) = DoubleArrayAhoCorasick::deserialize(daac_data)
                    .ok_or(SerdeError::InvalidData("failed to deserialize DAAC"))?;

                let enc = WordPieceEncoder::from_parts(
                    daac,
                    unk_token,
                    continuation_prefix,
                    vocab_size,
                    &token_bytes,
                );
                Encoder::WordPiece(enc)
            }
            EncoderType::SentencePiece => {
                // OPTIMIZED: Build lookups directly from decoder (single copy)
                let (byte_lut, token_cache, token_lengths) = build_token_lookups(&decoder_offsets, &decoder_data, vocab_size);
                let merges = deserialize_merges(merge_data)?;

                let enc = SentencePieceBPE::from_parts(
                    &merges,
                    byte_lut,
                    token_cache,
                    token_lengths,
                    vocab_size,
                    num_base_tokens,
                );
                Encoder::SentencePiece(enc)
            }
            EncoderType::Unigram => {
                // Unigram needs token_bytes for token_cache
                let token_bytes: Vec<Vec<u8>> = (0..vocab_size)
                    .map(|i| {
                        let start = decoder_offsets[i] as usize;
                        let end = decoder_offsets[i + 1] as usize;
                        decoder_data[start..end].to_vec()
                    })
                    .collect();

                let (scores, unk_token, byte_tokens, token_lengths) = deserialize_unigram_config(merge_data)?;
                let (daac, _) = DoubleArrayAhoCorasick::deserialize(daac_data)
                    .ok_or(SerdeError::InvalidData("failed to deserialize DAAC"))?;

                let enc = UnigramEncoder::from_parts(
                    daac,
                    scores,
                    unk_token,
                    byte_tokens,
                    token_lengths,
                    &token_bytes,
                );
                Encoder::Unigram(enc)
            }
        };

        // Build decoder
        let decoder = Decoder::from_parts(decoder_data, decoder_offsets);

        Ok(Tokenizer::new(encoder, decoder, pretokenizer_type, normalizer, post_processor))
    }
}

/// Serialize the decoder's flat buffer.
fn serialize_decoder(decoder: &Decoder) -> Vec<u8> {
    let (data, offsets) = decoder.as_parts();

    // Format: num_offsets (u32) + offsets + data
    let mut buf = Vec::with_capacity(4 + offsets.len() * 4 + data.len());

    buf.extend_from_slice(&(offsets.len() as u32).to_le_bytes());
    for &offset in offsets {
        buf.extend_from_slice(&offset.to_le_bytes());
    }
    buf.extend_from_slice(data);

    buf
}

/// Maximum token length to cache for early exit lookup.
const MAX_CACHED_TOKEN_LEN: usize = 16;

/// Build token lookups directly from decoder data (single copy, no intermediate Vec<Vec<u8>>).
/// Returns: (byte_lut, token_cache, token_lengths)
fn build_token_lookups(
    decoder_offsets: &[u32],
    decoder_data: &[u8],
    vocab_size: usize,
) -> ([TokenId; 256], FoldHashMap<Vec<u8>, TokenId>, Vec<u16>) {
    // Build byte_lut array for single-byte tokens
    let mut byte_lut = [u32::MAX; 256];

    // Pre-count short tokens for HashMap capacity
    let short_count: usize = (0..vocab_size)
        .filter(|&i| {
            let len = (decoder_offsets[i + 1] - decoder_offsets[i]) as usize;
            len <= MAX_CACHED_TOKEN_LEN
        })
        .count();

    let mut token_cache: FoldHashMap<Vec<u8>, TokenId> =
        FoldHashMap::with_capacity_and_hasher(short_count, Default::default());

    // Build token_lengths in the same pass
    let mut token_lengths: Vec<u16> = Vec::with_capacity(vocab_size);

    for i in 0..vocab_size {
        let start = decoder_offsets[i] as usize;
        let end = decoder_offsets[i + 1] as usize;
        let bytes = &decoder_data[start..end];
        let len = bytes.len();

        // Token length
        token_lengths.push(len as u16);

        // Single-byte lookup
        if len == 1 && byte_lut[bytes[0] as usize] == u32::MAX {
            byte_lut[bytes[0] as usize] = i as TokenId;
        }

        // Short token lookup (single copy into HashMap)
        if len <= MAX_CACHED_TOKEN_LEN {
            token_cache.insert(bytes.to_vec(), i as TokenId);
        }
    }

    (byte_lut, token_cache, token_lengths)
}

/// Deserialize the decoder's flat buffer.
/// Note: We read u32s manually because the slice may not be aligned.
fn deserialize_decoder(data: &[u8], vocab_size: usize) -> Result<(Vec<u32>, Vec<u8>), SerdeError> {
    if data.len() < 4 {
        return Err(SerdeError::InvalidData("decoder data too small"));
    }

    let num_offsets = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    if num_offsets != vocab_size + 1 {
        return Err(SerdeError::InvalidData("offset count mismatch"));
    }

    let offsets_end = 4 + num_offsets * 4;
    if data.len() < offsets_end {
        return Err(SerdeError::InvalidData("decoder data truncated"));
    }

    // Read offsets manually to handle unaligned data
    let mut offsets = Vec::with_capacity(num_offsets);
    for i in 0..num_offsets {
        let start = 4 + i * 4;
        offsets.push(u32::from_le_bytes(data[start..start + 4].try_into().unwrap()));
    }

    let token_data = data[offsets_end..].to_vec();

    Ok((offsets, token_data))
}

/// Serialize the split table.
fn serialize_splits(splits: &[Split]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(splits.len() * 8);
    for split in splits {
        buf.extend_from_slice(&split.left.to_le_bytes());
        buf.extend_from_slice(&split.right.to_le_bytes());
    }
    buf
}

/// Deserialize the split table.
/// Note: We read manually to handle unaligned data from file reads.
fn deserialize_splits(data: &[u8]) -> Result<Vec<Split>, SerdeError> {
    if data.len() % size_of::<Split>() != 0 {
        return Err(SerdeError::InvalidData("split data size not aligned"));
    }

    let num_splits = data.len() / size_of::<Split>();
    let mut splits = Vec::with_capacity(num_splits);

    for i in 0..num_splits {
        let start = i * 8;
        let left = u32::from_le_bytes(data[start..start + 4].try_into().unwrap());
        let right = u32::from_le_bytes(data[start + 4..start + 8].try_into().unwrap());
        splits.push(Split { left, right });
    }

    Ok(splits)
}

/// Serialize the next_prefix_match table.
fn serialize_prefix_match(prefixes: &[TokenId]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(prefixes.len() * 4);
    for &prefix in prefixes {
        buf.extend_from_slice(&prefix.to_le_bytes());
    }
    buf
}

/// Deserialize the next_prefix_match table.
fn deserialize_prefix_match(data: &[u8]) -> Result<Vec<TokenId>, SerdeError> {
    if data.len() % 4 != 0 {
        return Err(SerdeError::InvalidData("prefix data size not aligned"));
    }

    let num_prefixes = data.len() / 4;
    let mut prefixes = Vec::with_capacity(num_prefixes);

    for i in 0..num_prefixes {
        let start = i * 4;
        prefixes.push(u32::from_le_bytes(data[start..start + 4].try_into().unwrap()));
    }

    Ok(prefixes)
}

/// Pack two token IDs into a single u64 key.
#[inline(always)]
fn pack_pair(left: TokenId, right: TokenId) -> u64 {
    ((left as u64) << 32) | (right as u64)
}

/// Unpack u64 key back to two token IDs.
#[inline(always)]
fn unpack_pair(packed: u64) -> (TokenId, TokenId) {
    let left = (packed >> 32) as TokenId;
    let right = (packed & 0xFFFF_FFFF) as TokenId;
    (left, right)
}

/// Serialize Simple encoder's pair_lookup as merge list with merged IDs.
///
/// Format: (left: u32, right: u32, merged_id: u32) per merge, sorted by rank.
/// This allows fast deserialization without rebuilding the token_cache map.
fn serialize_pair_lookup(enc: &BytePairEncoder) -> Vec<u8> {
    let pair_lookup = enc.pair_lookup();

    // Collect all merges with their ranks and merged IDs
    let mut merges: Vec<(u32, TokenId, TokenId, TokenId)> = pair_lookup
        .iter()
        .map(|(&packed, &(merged, rank))| {
            let (left, right) = unpack_pair(packed);
            (rank, left, right, merged)
        })
        .collect();

    // Sort by rank to preserve merge order
    merges.sort_by_key(|(rank, _, _, _)| *rank);

    // Serialize as (left, right, merged_id) tuples
    let mut buf = Vec::with_capacity(merges.len() * 12);
    for (_, left, right, merged) in merges {
        buf.extend_from_slice(&left.to_le_bytes());
        buf.extend_from_slice(&right.to_le_bytes());
        buf.extend_from_slice(&merged.to_le_bytes());
    }
    buf
}

/// Serialize SentencePiece encoder's pair_lookup as merge list with merged IDs.
///
/// Format: (left: u32, right: u32, merged_id: u32) per merge, sorted by rank.
/// This allows fast deserialization without rebuilding the token_cache map.
fn serialize_sentencepiece_config(enc: &SentencePieceBPE) -> Vec<u8> {
    let pair_lookup = enc.pair_lookup();

    // Collect all merges with their ranks and merged IDs
    let mut merges: Vec<(u32, TokenId, TokenId, TokenId)> = pair_lookup
        .iter()
        .map(|(&packed, &(merged, rank))| {
            let (left, right) = unpack_pair(packed);
            (rank, left, right, merged)
        })
        .collect();

    // Sort by rank to preserve merge order
    merges.sort_by_key(|(rank, _, _, _)| *rank);

    // Serialize as (left, right, merged_id) tuples
    let mut buf = Vec::with_capacity(merges.len() * 12);
    for (_, left, right, merged) in merges {
        buf.extend_from_slice(&left.to_le_bytes());
        buf.extend_from_slice(&right.to_le_bytes());
        buf.extend_from_slice(&merged.to_le_bytes());
    }
    buf
}

/// Deserialize merge list for Simple/SentencePiece encoder.
///
/// Format: (left: u32, right: u32, merged_id: u32) per merge.
/// Returns tuples of (left, right, merged_id) for direct pair_lookup construction.
fn deserialize_merges(data: &[u8]) -> Result<Vec<(TokenId, TokenId, TokenId)>, SerdeError> {
    if data.len() % 12 != 0 {
        return Err(SerdeError::InvalidData("merge data size not aligned (expected 12 bytes per merge)"));
    }

    let num_merges = data.len() / 12;
    let mut merges = Vec::with_capacity(num_merges);

    for i in 0..num_merges {
        let start = i * 12;
        let left = u32::from_le_bytes(data[start..start + 4].try_into().unwrap());
        let right = u32::from_le_bytes(data[start + 4..start + 8].try_into().unwrap());
        let merged = u32::from_le_bytes(data[start + 8..start + 12].try_into().unwrap());
        merges.push((left, right, merged));
    }

    Ok(merges)
}

/// Rebuild pair_lookup from split_table using packed u64 keys.
fn rebuild_pair_lookup(
    splits: &[Split],
    num_base_tokens: usize,
) -> FoldHashMap<u64, TokenId> {
    let mut lookup = FoldHashMap::default();

    for (id, split) in splits.iter().enumerate().skip(num_base_tokens) {
        lookup.insert(pack_pair(split.left, split.right), id as TokenId);
    }

    lookup
}

/// Serialize WordPiece encoder config (unk_token + continuation_prefix).
///
/// Format: unk_token (u32) + prefix_len (u32) + prefix bytes
fn serialize_wordpiece_config(enc: &WordPieceEncoder) -> Vec<u8> {
    let prefix = enc.continuation_prefix();
    let mut buf = Vec::with_capacity(8 + prefix.len());
    buf.extend_from_slice(&enc.unk_token().to_le_bytes());
    buf.extend_from_slice(&(prefix.len() as u32).to_le_bytes());
    buf.extend_from_slice(prefix);
    buf
}

/// Deserialize WordPiece encoder config.
fn deserialize_wordpiece_config(data: &[u8]) -> Result<(TokenId, Vec<u8>), SerdeError> {
    if data.len() < 8 {
        return Err(SerdeError::InvalidData("wordpiece config too small"));
    }

    let unk_token = u32::from_le_bytes(data[0..4].try_into().unwrap());
    let prefix_len = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;

    if data.len() < 8 + prefix_len {
        return Err(SerdeError::InvalidData("wordpiece prefix truncated"));
    }

    let continuation_prefix = data[8..8 + prefix_len].to_vec();
    Ok((unk_token, continuation_prefix))
}

/// Serialize Unigram encoder config.
///
/// Format:
/// - vocab_size (u32)
/// - unk_token (u32)
/// - byte_tokens (256 × u32 = 1024 bytes)
/// - scores (vocab_size × f32)
/// - token_lengths (vocab_size × u16)
fn serialize_unigram_config(enc: &UnigramEncoder) -> Vec<u8> {
    let scores = enc.scores();
    let byte_tokens = enc.byte_tokens();
    let token_lengths = enc.token_lengths();
    let vocab_size = enc.vocab_size();

    // Calculate buffer size
    let buf_size = 4 + 4 + (256 * 4) + (vocab_size * 4) + (vocab_size * 2);
    let mut buf = Vec::with_capacity(buf_size);

    // vocab_size
    buf.extend_from_slice(&(vocab_size as u32).to_le_bytes());
    // unk_token
    buf.extend_from_slice(&enc.unk_token().to_le_bytes());
    // byte_tokens (256 u32s)
    for &bt in byte_tokens.iter() {
        buf.extend_from_slice(&bt.to_le_bytes());
    }
    // scores (f32 array)
    for &score in scores {
        buf.extend_from_slice(&score.to_le_bytes());
    }
    // token_lengths (u16 array)
    for &len in token_lengths {
        buf.extend_from_slice(&len.to_le_bytes());
    }

    buf
}

/// Deserialize Unigram encoder config.
fn deserialize_unigram_config(data: &[u8]) -> Result<(Vec<f32>, TokenId, [TokenId; 256], Vec<u16>), SerdeError> {
    if data.len() < 8 + 1024 {
        return Err(SerdeError::InvalidData("unigram config too small"));
    }

    let vocab_size = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    let unk_token = u32::from_le_bytes(data[4..8].try_into().unwrap());

    // Read byte_tokens (256 u32s starting at offset 8)
    let mut byte_tokens = [0u32; 256];
    for i in 0..256 {
        let start = 8 + i * 4;
        byte_tokens[i] = u32::from_le_bytes(data[start..start + 4].try_into().unwrap());
    }

    // Read scores (vocab_size f32s starting at offset 8 + 1024)
    let scores_offset = 8 + 1024;
    let expected_len = scores_offset + vocab_size * 4 + vocab_size * 2;
    if data.len() < expected_len {
        return Err(SerdeError::InvalidData("unigram config truncated"));
    }

    let mut scores = Vec::with_capacity(vocab_size);
    for i in 0..vocab_size {
        let start = scores_offset + i * 4;
        scores.push(f32::from_le_bytes(data[start..start + 4].try_into().unwrap()));
    }

    // Read token_lengths (vocab_size u16s starting after scores)
    let lengths_offset = scores_offset + vocab_size * 4;
    let mut token_lengths = Vec::with_capacity(vocab_size);
    for i in 0..vocab_size {
        let start = lengths_offset + i * 2;
        token_lengths.push(u16::from_le_bytes(data[start..start + 2].try_into().unwrap()));
    }

    Ok((scores, unk_token, byte_tokens, token_lengths))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TokenId;

    #[test]
    fn test_crc32() {
        assert_eq!(crc32(b""), 0);
        assert_eq!(crc32(b"hello"), crc32(b"hello"));
        assert_ne!(crc32(b"hello"), crc32(b"world"));
    }

    #[test]
    fn test_pretokenizer_type_roundtrip() {
        for typ in [
            PretokType::None,
            PretokType::Gpt2,
            PretokType::Cl100k,
            PretokType::O200k,
        ] {
            assert_eq!(PretokType::from_u32(typ as u32), Some(typ));
        }
    }

    fn make_test_tokenizer() -> Tokenizer {
        let base_tokens: Vec<Vec<u8>> = (0u8..=255).map(|b| vec![b]).collect();
        let merges: Vec<(TokenId, TokenId)> = vec![
            (b'a' as u32, b'b' as u32), // ab
            (b'c' as u32, b'd' as u32), // cd
            (256, 257),                  // abcd
        ];
        let (encoder, token_bytes) = crate::encoder::BacktrackingBytePairEncoder::from_merges(&merges, &base_tokens);
        let decoder = crate::decoder::Decoder::new(token_bytes);
        Tokenizer::new(Encoder::Backtracking(encoder), decoder, PretokType::Gpt2, Normalizer::None, PostProcessor::None)
    }

    fn make_simple_test_tokenizer() -> Tokenizer {
        let base_tokens: Vec<Vec<u8>> = (0u8..=255).map(|b| vec![b]).collect();
        let merges: Vec<(TokenId, TokenId)> = vec![
            (b'a' as u32, b'b' as u32), // ab
            (b'c' as u32, b'd' as u32), // cd
            (256, 257),                  // abcd
        ];
        let (encoder, token_bytes) = BytePairEncoder::from_merges(&merges, &base_tokens);
        let decoder = crate::decoder::Decoder::new(token_bytes);
        Tokenizer::new(Encoder::Simple(encoder), decoder, PretokType::Gpt2, Normalizer::None, PostProcessor::None)
    }

    #[test]
    fn test_save_load_roundtrip() {
        let tokenizer = make_test_tokenizer();

        // Save to memory buffer
        let mut buf = Vec::new();
        tokenizer
            .save(&mut buf)
            .expect("save failed");

        // Load from buffer
        let mut cursor = std::io::Cursor::new(&buf);
        let loaded = Tokenizer::load(&mut cursor).expect("load failed");

        // Verify same vocab size
        assert_eq!(tokenizer.vocab_size(), loaded.vocab_size());

        // Verify encoding matches
        let test_texts = ["Hello world", "abcd", "test 123", "abcdabcd"];
        for text in test_texts {
            let original_tokens = tokenizer.encode(text, false);
            let loaded_tokens = loaded.encode(text, false);
            assert_eq!(
                original_tokens, loaded_tokens,
                "encoding mismatch for '{}'",
                text
            );
        }

        // Verify decoding matches
        let tokens = tokenizer.encode("Hello world", false);
        let original_decoded = tokenizer.decode(&tokens);
        let loaded_decoded = loaded.decode(&tokens);
        assert_eq!(original_decoded, loaded_decoded);
    }

    #[test]
    fn test_save_load_file() {
        let tokenizer = make_test_tokenizer();

        let temp_path = std::env::temp_dir().join("tokie_test.bin");

        // Save to file
        tokenizer
            .to_file(&temp_path)
            .expect("to_file failed");

        // Load from file
        let loaded = Tokenizer::from_file(&temp_path).expect("from_file failed");

        // Verify encoding matches
        let text = "Hello world test";
        assert_eq!(tokenizer.encode(text, false), loaded.encode(text, false));

        // Cleanup
        std::fs::remove_file(&temp_path).ok();
    }

    #[test]
    fn test_load_invalid_magic() {
        let mut bad_data = vec![0u8; HEADER_SIZE + 100];
        bad_data[0..4].copy_from_slice(b"BADM");
        let mut cursor = std::io::Cursor::new(&bad_data);
        let result = Tokenizer::load(&mut cursor);
        assert!(matches!(result, Err(SerdeError::InvalidMagic)));
    }

    #[test]
    fn test_load_unsupported_version() {
        let mut data = Vec::new();
        data.extend_from_slice(MAGIC);
        data.extend_from_slice(&99u32.to_le_bytes()); // Bad version
        data.resize(HEADER_SIZE + 100, 0);

        let mut cursor = std::io::Cursor::new(&data);
        let result = Tokenizer::load(&mut cursor);
        assert!(matches!(result, Err(SerdeError::UnsupportedVersion(99))));
    }

    #[test]
    fn test_simple_encoder_save_load_roundtrip() {
        let tokenizer = make_simple_test_tokenizer();

        // Verify it's a Simple encoder
        assert_eq!(tokenizer.encoder_type(), EncoderType::Simple);

        // Save to memory buffer
        let mut buf = Vec::new();
        tokenizer
            .save(&mut buf)
            .expect("save failed");

        // Load from buffer
        let mut cursor = std::io::Cursor::new(&buf);
        let loaded = Tokenizer::load(&mut cursor).expect("load failed");

        // Verify it loaded as Simple encoder
        assert_eq!(loaded.encoder_type(), EncoderType::Simple);

        // Verify same vocab size
        assert_eq!(tokenizer.vocab_size(), loaded.vocab_size());

        // Verify encoding matches
        let test_texts = ["Hello world", "abcd", "test 123", "abcdabcd"];
        for text in test_texts {
            let original_tokens = tokenizer.encode(text, false);
            let loaded_tokens = loaded.encode(text, false);
            assert_eq!(
                original_tokens, loaded_tokens,
                "encoding mismatch for '{}'",
                text
            );
        }

        // Verify decoding matches
        let tokens = tokenizer.encode("Hello world", false);
        let original_decoded = tokenizer.decode(&tokens);
        let loaded_decoded = loaded.decode(&tokens);
        assert_eq!(original_decoded, loaded_decoded);
    }
}
