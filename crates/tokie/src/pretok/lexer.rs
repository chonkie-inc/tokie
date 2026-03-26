//! Hand-coded lexer for fast pretokenization.
//!
//! This module provides a single `Pretok` struct that can be configured
//! to behave like GPT-2, CL100K, O200K, or custom patterns.
//!
//! Achieves ~400 MiB/s throughput with full Unicode support.

use unicode_general_category::{get_general_category, GeneralCategory};

// ============================================================================
// Configuration
// ============================================================================

/// How letters can be prefixed in the pattern.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LetterPrefix {
    /// Only space can prefix letters: ` ?\p{L}+` (GPT-2)
    SpaceOnly,
    /// Any non-newline, non-letter, non-number: `[^\r\n\p{L}\p{N}]?\p{L}+` (CL100K, O200K)
    NonNewlineNonAlnum,
}

/// How letters are scanned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LetterScanning {
    /// Simple: any letter `\p{L}+` (GPT-2, CL100K)
    Simple,
    /// Case-aware: `Upper*Lower+` or `Upper+Lower*` for CamelCase splitting (O200K)
    CaseAware,
}

/// Where contractions appear in the pattern.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContractionPosition {
    /// Contractions are standalone patterns at start: `'s|'t|...` (GPT-2, CL100K)
    Standalone,
    /// Contractions are suffixes on words: `word(?i:'s)?` (O200K)
    Suffix,
}

/// Pretokenizer for BPE tokenizers.
///
/// Splits text into pieces before subword tokenization. Configure behavior
/// using the fields or use predefined constants for common tokenizers.
///
/// # Example
///
/// ```
/// use tokie::pretok::Pretok;
///
/// // Use predefined config
/// let pieces: Vec<&str> = Pretok::GPT2.split("Hello world").collect();
/// assert_eq!(pieces, vec!["Hello", " world"]);
///
/// // CL100K chunks numbers
/// let pieces: Vec<&str> = Pretok::CL100K.split("12345").collect();
/// assert_eq!(pieces, vec!["123", "45"]);
/// ```
///
/// # Configuration Groups
///
/// The fields are organized into logical groups:
///
/// **Letter handling**: `letter_prefix`, `letter_scanning`
/// **Contraction handling**: `contraction_position`, `case_insensitive_contractions`
/// **Number handling**: `max_number_digits`, `space_prefix_numbers`, `keep_alphanumeric_together`
/// **Punctuation handling**: `individual_punctuation`, `punct_trailing_newlines`, `punct_trailing_slash`
/// **Whitespace handling**: `include_leading_space`, `add_prefix_space`, `newline_lookahead`, `newline_anchored_whitespace`
#[derive(Clone, Copy, Debug)]
pub struct Pretok {
    // --- Letter handling ---
    /// How letters can be prefixed: `SpaceOnly` (GPT-2) or `NonNewlineNonAlnum` (CL100K/O200K).
    pub letter_prefix: LetterPrefix,
    /// How letters are scanned: `Simple` or `CaseAware` for CamelCase splitting.
    pub letter_scanning: LetterScanning,

    // --- Contraction handling ---
    /// Where contractions appear: `Standalone` ('s separate) or `Suffix` (word's together).
    pub contraction_position: ContractionPosition,
    /// Whether contractions are case-insensitive (`'S` matches like `'s`).
    pub case_insensitive_contractions: bool,

    // --- Number handling ---
    /// Maximum digits per number token (0 = unlimited, 3 = chunk into groups of 3).
    pub max_number_digits: u8,
    /// Whether space can prefix numbers: " 123" (GPT-2: true, CL100K/O200K: false).
    pub space_prefix_numbers: bool,
    /// Whether letters and numbers stay together: "18th" (BERT: true, others: false).
    pub keep_alphanumeric_together: bool,

    // --- Punctuation handling ---
    /// Whether each punctuation is separate: "..." → [".", ".", "."] (BERT: true).
    pub individual_punctuation: bool,
    /// Whether punctuation can include trailing newlines (CL100K/O200K: true).
    pub punct_trailing_newlines: bool,
    /// Whether punctuation can include trailing slashes (O200K only).
    pub punct_trailing_slash: bool,

    // --- Whitespace handling ---
    /// Whether leading space is part of tokens: " world" vs "world" (BERT: false).
    pub include_leading_space: bool,
    /// Whether to prepend space to input so first word gets leading space (GPT-2: true).
    pub add_prefix_space: bool,
    /// Whether newline sequences use lookahead for grouping (GPT-2: true).
    pub newline_lookahead: bool,
    /// Whether whitespace runs anchor on the last newline: `\s*[\r\n]+` (CL100K/Llama/Qwen: true).
    pub newline_anchored_whitespace: bool,
    /// Whether combining marks (\p{M}) are scanned as part of letter runs (DeepSeek: true).
    pub include_marks_in_letters: bool,
    /// Whether to skip whitespace backtracking before digits (DeepSeek: true).
    /// In multi-stage pretokenizers, digit groups are extracted first, so whitespace
    /// before digits should be consumed greedily rather than leaving one space for
    /// the next token's prefix.
    pub no_backtrack_before_digit: bool,
    /// Whether punct prefix only matches ASCII letters (DeepSeek: true).
    /// When true, `[punct][A-Za-z]+` only scans ASCII letters after punctuation,
    /// stopping at non-ASCII letters like `ë`. When false, `\p{L}+` scans all
    /// Unicode letters after any prefix.
    pub punct_prefix_ascii_only: bool,
}

impl Pretok {
    /// GPT-2 compatible configuration.
    ///
    /// Used by: GPT-2, GPT-J, GPT-Neo
    pub const GPT2: Self = Self {
        // Letter handling
        letter_prefix: LetterPrefix::SpaceOnly,
        letter_scanning: LetterScanning::Simple,
        // Contraction handling
        contraction_position: ContractionPosition::Standalone,
        case_insensitive_contractions: false,
        // Number handling
        max_number_digits: 0,
        space_prefix_numbers: true,
        keep_alphanumeric_together: false,
        // Punctuation handling
        individual_punctuation: false,
        punct_trailing_newlines: false,
        punct_trailing_slash: false,
        // Whitespace handling
        include_leading_space: true,
        add_prefix_space: true,
        newline_lookahead: true,
        newline_anchored_whitespace: false,
        include_marks_in_letters: false,
        no_backtrack_before_digit: false,
        punct_prefix_ascii_only: false,
    };

    /// CL100K compatible configuration.
    ///
    /// Used by: GPT-3.5, GPT-4
    pub const CL100K: Self = Self {
        // Letter handling
        letter_prefix: LetterPrefix::NonNewlineNonAlnum,
        letter_scanning: LetterScanning::Simple,
        // Contraction handling
        contraction_position: ContractionPosition::Standalone,
        case_insensitive_contractions: true,
        // Number handling
        max_number_digits: 3,
        space_prefix_numbers: false,
        keep_alphanumeric_together: false,
        // Punctuation handling
        individual_punctuation: false,
        punct_trailing_newlines: true,
        punct_trailing_slash: false,
        // Whitespace handling
        include_leading_space: true,
        add_prefix_space: false,
        newline_lookahead: false,
        newline_anchored_whitespace: true,
        include_marks_in_letters: false,
        no_backtrack_before_digit: false,
        punct_prefix_ascii_only: false,
    };

    /// O200K compatible configuration.
    ///
    /// Used by: GPT-4o, o1, o3
    pub const O200K: Self = Self {
        // Letter handling
        letter_prefix: LetterPrefix::NonNewlineNonAlnum,
        letter_scanning: LetterScanning::CaseAware,
        // Contraction handling
        contraction_position: ContractionPosition::Suffix,
        case_insensitive_contractions: true,
        // Number handling
        max_number_digits: 3,
        space_prefix_numbers: false,
        keep_alphanumeric_together: false,
        // Punctuation handling
        individual_punctuation: false,
        punct_trailing_newlines: true,
        punct_trailing_slash: true,
        // Whitespace handling
        include_leading_space: true,
        add_prefix_space: false,
        newline_lookahead: false,
        newline_anchored_whitespace: true,
        include_marks_in_letters: false,
        no_backtrack_before_digit: false,
        punct_prefix_ascii_only: false,
    };

    /// Voyage compatible configuration.
    ///
    /// Used by: Voyage 3, Voyage 3 Large, Voyage Code 3
    ///
    /// Similar to CL100K but with single-digit number matching:
    /// - Tab/punctuation can prefix letters: `.hello` → \[`.hello`\]
    /// - Numbers matched one digit at a time: `12345` → [`1`, `2`, `3`, `4`, `5`]
    /// - Case-insensitive contractions
    pub const VOYAGE: Self = Self {
        // Letter handling
        letter_prefix: LetterPrefix::NonNewlineNonAlnum,
        letter_scanning: LetterScanning::Simple,
        // Contraction handling
        contraction_position: ContractionPosition::Standalone,
        case_insensitive_contractions: true,
        // Number handling
        max_number_digits: 1, // Single digits (unlike CL100K's 3)
        space_prefix_numbers: false,
        keep_alphanumeric_together: false,
        // Punctuation handling
        individual_punctuation: false,
        punct_trailing_newlines: true,
        punct_trailing_slash: false,
        // Whitespace handling
        include_leading_space: true,
        add_prefix_space: false,
        newline_lookahead: false,
        newline_anchored_whitespace: true,
        include_marks_in_letters: false,
        no_backtrack_before_digit: false,
        punct_prefix_ascii_only: false,
    };

    /// DeepSeek compatible configuration.
    ///
    /// Used by: DeepSeek-V3, DeepSeek-R1
    ///
    /// Similar to CL100K but includes combining marks in letter runs:
    /// - `[\p{L}\p{M}]+` instead of `\p{L}+` (Thai, Arabic combining marks stay with letters)
    /// - No contractions
    pub const DEEPSEEK: Self = Self {
        // Letter handling
        letter_prefix: LetterPrefix::NonNewlineNonAlnum,
        letter_scanning: LetterScanning::Simple,
        // Contraction handling
        contraction_position: ContractionPosition::Standalone,
        case_insensitive_contractions: true,
        // Number handling
        max_number_digits: 3,
        space_prefix_numbers: false,
        keep_alphanumeric_together: false,
        // Punctuation handling
        individual_punctuation: false,
        punct_trailing_newlines: true,
        punct_trailing_slash: false,
        // Whitespace handling
        include_leading_space: true,
        add_prefix_space: false,
        newline_lookahead: false,
        newline_anchored_whitespace: true,
        include_marks_in_letters: true,
        no_backtrack_before_digit: true,
        punct_prefix_ascii_only: true,
    };

    /// BERT compatible configuration.
    ///
    /// Used by: BERT, DistilBERT, GTE, BGE, E5, MiniLM (with WordPiece encoder)
    ///
    /// BERT-style pretokenization:
    /// - Splits on whitespace (whitespace not included in tokens)
    /// - Punctuation separated into individual tokens
    /// - No contractions handling
    /// - No number chunking
    pub const BERT: Self = Self {
        // Letter handling
        letter_prefix: LetterPrefix::SpaceOnly,
        letter_scanning: LetterScanning::Simple,
        // Contraction handling
        contraction_position: ContractionPosition::Standalone,
        case_insensitive_contractions: false,
        // Number handling
        max_number_digits: 0,
        space_prefix_numbers: false,
        keep_alphanumeric_together: true,
        // Punctuation handling
        individual_punctuation: true,
        punct_trailing_newlines: false,
        punct_trailing_slash: false,
        // Whitespace handling
        include_leading_space: false,
        add_prefix_space: false,
        newline_lookahead: false,
        newline_anchored_whitespace: false,
        include_marks_in_letters: false,
        no_backtrack_before_digit: false,
        punct_prefix_ascii_only: false,
    };

    /// Split text into pre-tokens.
    #[inline]
    pub fn split<'a>(&self, text: &'a str) -> PretokIter<'a> {
        PretokIter {
            bytes: text.as_bytes(),
            len: text.len(),
            pos: 0,
            config: *self,
        }
    }

    /// Split text and collect into a Vec.
    pub fn split_to_vec<'a>(&self, text: &'a str) -> Vec<&'a str> {
        self.split(text).collect()
    }
}

impl Default for Pretok {
    fn default() -> Self {
        Self::GPT2
    }
}

// ============================================================================
// Iterator
// ============================================================================

/// Iterator over pre-tokens.
pub struct PretokIter<'a> {
    bytes: &'a [u8],
    len: usize,
    pos: usize,
    config: Pretok,
}

impl<'a> Iterator for PretokIter<'a> {
    type Item = &'a str;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let bytes = self.bytes;

        // Skip whitespace when include_leading_space is false (BERT mode)
        if !self.config.include_leading_space {
            while self.pos < self.len {
                let b = unsafe { *bytes.get_unchecked(self.pos) };
                match b {
                    b' ' | b'\t' | b'\r' | b'\n' | 0x0B | 0x0C => {
                        self.pos += 1;
                    }
                    0x80..=0xFF => {
                        let (c, char_len) = decode_utf8(unsafe { bytes.get_unchecked(self.pos..) });
                        if c.is_whitespace() {
                            self.pos += char_len;
                        } else {
                            break;
                        }
                    }
                    _ => break,
                }
            }
        }

        if self.pos >= self.len {
            return None;
        }

        let start = self.pos;
        let b = unsafe { *bytes.get_unchecked(self.pos) };

        match b {
            b'a'..=b'z' | b'A'..=b'Z' => {
                self.pos += 1;
                self.scan_alphanumeric_from_letter();
                self.try_contraction_suffix_if_needed();
            }
            b'0'..=b'9' => {
                self.pos += 1;
                self.scan_alphanumeric_from_number();
            }
            b' ' => {
                self.handle_space(start);
            }
            b'\'' => {
                // BERT mode: each punctuation is separate, no contraction handling
                if self.config.individual_punctuation {
                    self.pos += 1;
                    // Don't scan_other - just return the single apostrophe
                } else if self.config.contraction_position == ContractionPosition::Standalone {
                    // For standalone contractions (GPT-2, CL100K), check for 's, 't, etc.
                    let len = self.check_contraction();
                    if len > 0 {
                        self.pos += len;
                    } else if self.config.letter_prefix == LetterPrefix::NonNewlineNonAlnum
                        && self.try_scan_prefix_letters(1)
                    {
                        // Apostrophe as letter prefix: 'être (CL100K/Llama/Qwen pattern)
                    } else {
                        self.pos += 1;
                        self.scan_other();
                    }
                } else {
                    // O200K: apostrophe could prefix letters or is just punctuation
                    if self.config.letter_prefix == LetterPrefix::NonNewlineNonAlnum
                        && self.try_scan_prefix_letters(1)
                    {
                        // Apostrophe as letter prefix
                    } else {
                        self.pos += 1;
                        self.scan_other();
                    }
                }
            }
            b'\r' | b'\n' => {
                self.scan_whitespace(start);
            }
            b'\t' | 0x0B | 0x0C => {
                // CL100K/O200K: tabs can prefix letters: `[^\r\n\p{L}\p{N}]?\p{L}+`
                if self.config.letter_prefix == LetterPrefix::NonNewlineNonAlnum
                    && self.try_scan_prefix_letters(1)
                {
                    // Tab + letters combined into one token
                } else {
                    self.scan_whitespace(start);
                }
            }
            0x80..=0xFF => {
                self.handle_unicode(start);
            }
            _ => {
                // Other ASCII (punctuation)
                self.handle_other(start);
            }
        }

        Some(unsafe { std::str::from_utf8_unchecked(&bytes[start..self.pos]) })
    }
}

// ============================================================================
// Character Classification Helpers
// ============================================================================

#[inline(always)]
fn is_ascii_letter(b: u8) -> bool {
    let lower = b | 0x20;
    lower >= b'a' && lower <= b'z'
}

#[inline]
fn is_unicode_letter(c: char) -> bool {
    matches!(
        get_general_category(c),
        GeneralCategory::UppercaseLetter
            | GeneralCategory::LowercaseLetter
            | GeneralCategory::TitlecaseLetter
            | GeneralCategory::ModifierLetter
            | GeneralCategory::OtherLetter
    )
}

#[inline]
fn is_unicode_mark(c: char) -> bool {
    matches!(
        get_general_category(c),
        GeneralCategory::NonspacingMark
            | GeneralCategory::SpacingMark
            | GeneralCategory::EnclosingMark
    )
}

#[inline]
fn is_unicode_number(c: char) -> bool {
    matches!(
        get_general_category(c),
        GeneralCategory::DecimalNumber
            | GeneralCategory::LetterNumber
            | GeneralCategory::OtherNumber
    )
}

#[inline(always)]
fn decode_utf8(bytes: &[u8]) -> (char, usize) {
    let b0 = bytes[0];
    if b0 < 0x80 {
        (b0 as char, 1)
    } else if b0 < 0xE0 {
        let c = ((b0 as u32 & 0x1F) << 6) | (bytes[1] as u32 & 0x3F);
        (unsafe { char::from_u32_unchecked(c) }, 2)
    } else if b0 < 0xF0 {
        let c = ((b0 as u32 & 0x0F) << 12)
            | ((bytes[1] as u32 & 0x3F) << 6)
            | (bytes[2] as u32 & 0x3F);
        (unsafe { char::from_u32_unchecked(c) }, 3)
    } else {
        let c = ((b0 as u32 & 0x07) << 18)
            | ((bytes[1] as u32 & 0x3F) << 12)
            | ((bytes[2] as u32 & 0x3F) << 6)
            | (bytes[3] as u32 & 0x3F);
        (unsafe { char::from_u32_unchecked(c) }, 4)
    }
}

/// Check if a character is a CJK ideograph (used by BERT pretokenizer).
///
/// This matches the ranges used by HuggingFace's `_is_chinese_char`:
/// <https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/pre_tokenizers/bert.rs>
#[inline]
fn is_cjk_character(c: char) -> bool {
    let cp = c as u32;
    matches!(cp,
        0x4E00..=0x9FFF       // CJK Unified Ideographs
        | 0x3400..=0x4DBF     // CJK Unified Ideographs Extension A
        | 0x20000..=0x2A6DF   // CJK Unified Ideographs Extension B
        | 0x2A700..=0x2B73F   // CJK Unified Ideographs Extension C
        | 0x2B740..=0x2B81F   // CJK Unified Ideographs Extension D
        | 0x2B820..=0x2CEAF   // CJK Unified Ideographs Extension E
        | 0xF900..=0xFAFF     // CJK Compatibility Ideographs
        | 0x2F800..=0x2FA1F   // CJK Compatibility Ideographs Supplement
    )
}

/// Classify a Unicode character.
#[derive(Clone, Copy, PartialEq, Eq)]
enum UnicodeClass {
    Letter,
    Number,
    Newline,
    Whitespace,
    CJK,
    Punctuation,
    Other,
}

/// Check if a Unicode character is punctuation (for BERT pretokenizer).
///
/// Matches HuggingFace's `_is_punctuation`: ASCII punct ranges + Unicode P* categories.
#[inline]
fn is_unicode_punctuation(c: char) -> bool {
    let cp = c as u32;
    // ASCII punctuation ranges (same as HF)
    if (33..=47).contains(&cp)
        || (58..=64).contains(&cp)
        || (91..=96).contains(&cp)
        || (123..=126).contains(&cp)
    {
        return true;
    }
    // Unicode Punctuation categories
    matches!(
        get_general_category(c),
        GeneralCategory::ConnectorPunctuation
            | GeneralCategory::DashPunctuation
            | GeneralCategory::ClosePunctuation
            | GeneralCategory::FinalPunctuation
            | GeneralCategory::InitialPunctuation
            | GeneralCategory::OtherPunctuation
            | GeneralCategory::OpenPunctuation
    )
}

#[inline]
fn classify_unicode(c: char) -> UnicodeClass {
    if c == '\r' || c == '\n' {
        UnicodeClass::Newline
    } else if c.is_whitespace() {
        UnicodeClass::Whitespace
    } else if is_cjk_character(c) {
        UnicodeClass::CJK
    } else if is_unicode_letter(c) {
        UnicodeClass::Letter
    } else if is_unicode_number(c) {
        UnicodeClass::Number
    } else if is_unicode_punctuation(c) {
        UnicodeClass::Punctuation
    } else {
        UnicodeClass::Other
    }
}

/// Letter case for O200K case-aware scanning.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum LetterCase {
    /// Uppercase or Titlecase: \p{Lu} | \p{Lt}
    Upper,
    /// Lowercase: \p{Ll}
    Lower,
    /// Other letters and marks: \p{Lm} | \p{Lo} | \p{M}
    Other,
    /// Not a letter
    None,
}

/// Classify a character's case for O200K patterns.
#[inline]
fn classify_letter_case(c: char) -> LetterCase {
    match get_general_category(c) {
        GeneralCategory::UppercaseLetter | GeneralCategory::TitlecaseLetter => LetterCase::Upper,
        GeneralCategory::LowercaseLetter => LetterCase::Lower,
        GeneralCategory::ModifierLetter
        | GeneralCategory::OtherLetter
        | GeneralCategory::NonspacingMark
        | GeneralCategory::SpacingMark
        | GeneralCategory::EnclosingMark => LetterCase::Other,
        _ => LetterCase::None,
    }
}

/// Check if byte is ASCII uppercase.
#[inline(always)]
fn is_ascii_upper(b: u8) -> bool {
    b >= b'A' && b <= b'Z'
}

/// Check if byte is ASCII lowercase.
#[inline(always)]
fn is_ascii_lower(b: u8) -> bool {
    b >= b'a' && b <= b'z'
}

// ============================================================================
// Scanning Functions
// ============================================================================

impl PretokIter<'_> {
    /// Scan letters based on config (simple or case-aware).
    #[inline]
    fn scan_letters(&mut self) {
        if self.config.letter_scanning == LetterScanning::CaseAware {
            self.scan_letters_case_aware();
        } else {
            self.scan_letters_simple();
        }
    }

    /// Try to append contraction suffix if O200K mode.
    /// This consolidates the repeated pattern throughout the code.
    #[inline]
    fn try_contraction_suffix_if_needed(&mut self) {
        if self.config.contraction_position == ContractionPosition::Suffix {
            self.try_contraction_suffix();
        }
    }

    /// Try to scan prefix + letters if next char is a letter.
    /// Returns true if letters were scanned (prefix consumed).
    /// `prefix_len` is the number of bytes of prefix already consumed.
    #[inline]
    fn try_scan_prefix_letters(&mut self, prefix_len: usize) -> bool {
        let bytes = self.bytes;
        if self.pos + prefix_len >= self.len {
            return false;
        }
        let next_b = unsafe { *bytes.get_unchecked(self.pos + prefix_len) };
        if is_ascii_letter(next_b) {
            self.pos += prefix_len + 1;
            self.scan_letters();
            self.try_contraction_suffix_if_needed();
            true
        } else if next_b >= 0x80 {
            let (c, char_len) = decode_utf8(unsafe { bytes.get_unchecked(self.pos + prefix_len..) });
            if is_unicode_letter(c) {
                self.pos += prefix_len + char_len;
                self.scan_letters();
                self.try_contraction_suffix_if_needed();
                true
            } else {
                false
            }
        } else {
            false
        }
    }

    /// Try to scan prefix + ASCII letters only: `[prefix][A-Za-z]+`
    /// Used by DeepSeek where punct prefix only groups with ASCII letters.
    #[inline]
    fn try_scan_prefix_ascii_letters(&mut self, prefix_len: usize) -> bool {
        if self.pos + prefix_len >= self.len {
            return false;
        }
        let next_b = unsafe { *self.bytes.get_unchecked(self.pos + prefix_len) };
        if is_ascii_letter(next_b) {
            self.pos += prefix_len + 1;
            // Scan only ASCII letters
            while self.pos < self.len {
                let b = unsafe { *self.bytes.get_unchecked(self.pos) };
                if is_ascii_letter(b) {
                    self.pos += 1;
                } else {
                    break;
                }
            }
            true
        } else {
            false
        }
    }

    /// Simple letter scanning: `\p{L}+` or `[\p{L}\p{M}]+`
    /// In BERT mode, stops at CJK and punctuation but continues through marks.
    /// When `include_marks_in_letters` is true, also continues through combining marks.
    #[inline]
    fn scan_letters_simple(&mut self) {
        let bytes = self.bytes;
        let len = self.len;
        let bert_mode = self.config.individual_punctuation;
        let include_marks = self.config.include_marks_in_letters;

        while self.pos < len {
            let b = unsafe { *bytes.get_unchecked(self.pos) };
            if is_ascii_letter(b) {
                self.pos += 1;
            } else if b < 0x80 {
                return;
            } else {
                let (c, char_len) = decode_utf8(unsafe { bytes.get_unchecked(self.pos..) });
                if bert_mode {
                    // In BERT mode: continue through letters and marks,
                    // stop at CJK, punctuation, whitespace, numbers
                    match classify_unicode(c) {
                        UnicodeClass::Letter | UnicodeClass::Other => {
                            self.pos += char_len;
                        }
                        _ => return,
                    }
                } else if is_unicode_letter(c) {
                    self.pos += char_len;
                } else if include_marks && is_unicode_mark(c) {
                    self.pos += char_len;
                } else {
                    return;
                }
            }
        }
    }

    /// Case-aware letter scanning for O200K.
    ///
    /// O200K has two letter patterns:
    /// - `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+` - Upper*Lower+
    /// - `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*` - Upper+Lower*
    /// - `\p{Ll}+` - all lowercase
    ///
    /// If we start with lowercase, we only scan lowercase (stop at uppercase).
    /// If we start with uppercase, we scan uppercase then optional lowercase.
    #[inline]
    fn scan_letters_case_aware(&mut self) {
        let bytes = self.bytes;

        // Check what the first letter was (already consumed)
        let first_pos = self.pos - 1;
        let first_b = unsafe { *bytes.get_unchecked(first_pos) };

        let started_lower = if first_b < 0x80 {
            is_ascii_lower(first_b)
        } else {
            let (c, _) = decode_utf8(unsafe { bytes.get_unchecked(first_pos..) });
            classify_letter_case(c) == LetterCase::Lower
        };

        if started_lower {
            // Pattern: \p{Ll}+ - scan lowercase only, stop at uppercase
            self.scan_lowercase_only();
        } else {
            // Pattern: Upper*Lower* or Upper+Lower*
            // Scan uppercase/other, then optional lowercase
            self.scan_upper_then_lower();
        }
    }

    /// Scan lowercase letters only (stop at uppercase).
    #[inline]
    fn scan_lowercase_only(&mut self) {
        let bytes = self.bytes;
        let len = self.len;

        while self.pos < len {
            let b = unsafe { *bytes.get_unchecked(self.pos) };
            if is_ascii_lower(b) {
                self.pos += 1;
            } else if b < 0x80 {
                // Uppercase or non-letter - stop
                return;
            } else {
                let (c, char_len) = decode_utf8(unsafe { bytes.get_unchecked(self.pos..) });
                let case = classify_letter_case(c);
                match case {
                    LetterCase::Lower => {
                        self.pos += char_len;
                    }
                    LetterCase::Other => {
                        // \p{Lm}, \p{Lo}, \p{M} - continue in lowercase pattern
                        self.pos += char_len;
                    }
                    _ => return, // Uppercase or non-letter - stop
                }
            }
        }
    }

    /// Scan uppercase letters, then optional lowercase.
    #[inline]
    fn scan_upper_then_lower(&mut self) {
        let bytes = self.bytes;
        let len = self.len;

        // First, consume uppercase/other letters
        while self.pos < len {
            let b = unsafe { *bytes.get_unchecked(self.pos) };
            if is_ascii_upper(b) {
                self.pos += 1;
            } else if is_ascii_lower(b) {
                // Hit lowercase - consume it and remaining lowercase
                self.pos += 1;
                self.scan_lower_and_other();
                return;
            } else if b < 0x80 {
                // Non-letter - stop
                return;
            } else {
                let (c, char_len) = decode_utf8(unsafe { bytes.get_unchecked(self.pos..) });
                let case = classify_letter_case(c);
                match case {
                    LetterCase::Upper => {
                        self.pos += char_len;
                    }
                    LetterCase::Lower => {
                        self.pos += char_len;
                        self.scan_lower_and_other();
                        return;
                    }
                    LetterCase::Other => {
                        self.pos += char_len;
                    }
                    LetterCase::None => return,
                }
            }
        }
    }

    /// Scan lowercase and "other" letters (for O200K pattern continuation).
    #[inline]
    fn scan_lower_and_other(&mut self) {
        let bytes = self.bytes;
        let len = self.len;

        while self.pos < len {
            let b = unsafe { *bytes.get_unchecked(self.pos) };
            if is_ascii_lower(b) {
                self.pos += 1;
            } else if b < 0x80 {
                // Uppercase or non-letter - stop
                return;
            } else {
                let (c, char_len) = decode_utf8(unsafe { bytes.get_unchecked(self.pos..) });
                let case = classify_letter_case(c);
                match case {
                    LetterCase::Lower | LetterCase::Other => {
                        self.pos += char_len;
                    }
                    _ => return,
                }
            }
        }
    }

    /// Scan numbers with optional chunking.
    #[inline]
    fn scan_numbers(&mut self, mut count: usize) {
        let bytes = self.bytes;
        let len = self.len;
        let limit = if self.config.max_number_digits == 0 {
            usize::MAX
        } else {
            self.config.max_number_digits as usize
        };

        while self.pos < len && count < limit {
            let b = unsafe { *bytes.get_unchecked(self.pos) };
            if b >= b'0' && b <= b'9' {
                self.pos += 1;
                count += 1;
            } else if b < 0x80 {
                return;
            } else {
                let (c, char_len) = decode_utf8(unsafe { bytes.get_unchecked(self.pos..) });
                if is_unicode_number(c) {
                    self.pos += char_len;
                    count += 1;
                } else {
                    return;
                }
            }
        }
    }

    /// Scan alphanumeric starting from a letter.
    /// If keep_alphanumeric_together is true, continues scanning numbers after letters.
    #[inline]
    fn scan_alphanumeric_from_letter(&mut self) {
        self.scan_letters();
        if self.config.keep_alphanumeric_together {
            // Continue scanning letters and numbers together
            self.scan_alphanumeric_continuation();
        }
    }

    /// Scan alphanumeric starting from a number.
    /// If keep_alphanumeric_together is true, continues scanning letters after numbers.
    #[inline]
    fn scan_alphanumeric_from_number(&mut self) {
        self.scan_numbers(1);
        if self.config.keep_alphanumeric_together {
            // Continue scanning letters and numbers together
            self.scan_alphanumeric_continuation();
        }
    }

    /// Continue scanning both letters and numbers until neither is found.
    /// In BERT mode, stops at CJK and punctuation but continues through marks.
    #[inline]
    fn scan_alphanumeric_continuation(&mut self) {
        let bytes = self.bytes;
        let len = self.len;
        let bert_mode = self.config.individual_punctuation;

        while self.pos < len {
            let b = unsafe { *bytes.get_unchecked(self.pos) };
            if is_ascii_letter(b) || (b >= b'0' && b <= b'9') {
                self.pos += 1;
            } else if b < 0x80 {
                return;
            } else {
                let (c, char_len) = decode_utf8(unsafe { bytes.get_unchecked(self.pos..) });
                if bert_mode {
                    match classify_unicode(c) {
                        UnicodeClass::Letter | UnicodeClass::Number | UnicodeClass::Other => {
                            self.pos += char_len;
                        }
                        _ => return,
                    }
                } else if is_unicode_letter(c) || is_unicode_number(c) {
                    self.pos += char_len;
                } else {
                    return;
                }
            }
        }
    }

    /// Scan word characters in BERT mode: letters, numbers, marks, and other non-punct chars.
    /// Stops at whitespace, punctuation, and CJK characters.
    #[inline]
    fn scan_bert_word_chars(&mut self) {
        let bytes = self.bytes;
        let len = self.len;

        while self.pos < len {
            let b = unsafe { *bytes.get_unchecked(self.pos) };
            if is_ascii_letter(b) || (b >= b'0' && b <= b'9') {
                self.pos += 1;
            } else if b < 0x80 {
                // ASCII non-alnum: whitespace or punctuation — stop
                return;
            } else {
                let (c, char_len) = decode_utf8(unsafe { bytes.get_unchecked(self.pos..) });
                match classify_unicode(c) {
                    UnicodeClass::Letter | UnicodeClass::Number | UnicodeClass::Other => {
                        self.pos += char_len;
                    }
                    _ => return, // Whitespace, Newline, CJK, Punctuation — stop
                }
            }
        }
    }

    /// Scan "other" characters (punctuation, symbols).
    #[inline]
    fn scan_other(&mut self) {
        // BERT mode: each punctuation is a separate token
        if self.config.individual_punctuation {
            return;
        }

        let bytes = self.bytes;
        let len = self.len;

        while self.pos < len {
            let b = unsafe { *bytes.get_unchecked(self.pos) };
            // Continue if punctuation or apostrophe (but not whitespace/letter/number)
            if b == b'\''
                || (b > b' ' && b < 0x80 && !is_ascii_letter(b) && !(b >= b'0' && b <= b'9'))
            {
                self.pos += 1;
            } else if b >= 0x80 {
                let (c, char_len) = decode_utf8(unsafe { bytes.get_unchecked(self.pos..) });
                match classify_unicode(c) {
                    UnicodeClass::Other | UnicodeClass::Punctuation => {
                        self.pos += char_len;
                    }
                    _ => break,
                }
            } else {
                break;
            }
        }

        // Optionally consume trailing newlines and/or slashes
        if self.config.punct_trailing_newlines || self.config.punct_trailing_slash {
            while self.pos < len {
                let b = unsafe { *bytes.get_unchecked(self.pos) };
                if (self.config.punct_trailing_newlines && (b == b'\r' || b == b'\n'))
                    || (self.config.punct_trailing_slash && b == b'/')
                {
                    self.pos += 1;
                } else {
                    break;
                }
            }
        }
    }

    /// Try to consume a contraction suffix (for O200K): `(?i:'s|'t|'re|'ve|'m|'ll|'d)?`
    /// Returns the number of bytes consumed (0 if no contraction found).
    #[inline]
    fn try_contraction_suffix(&mut self) -> usize {
        let bytes = self.bytes;
        let pos = self.pos;
        let len = self.len;

        if pos >= len || bytes[pos] != b'\'' {
            return 0;
        }

        if pos + 1 >= len {
            return 0;
        }

        // Case-insensitive check
        let b = bytes[pos + 1] | 0x20;

        let contraction_len = match b {
            b's' | b't' | b'm' | b'd' => 2,
            b'r' if pos + 2 < len && (bytes[pos + 2] | 0x20) == b'e' => 3,
            b'v' if pos + 2 < len && (bytes[pos + 2] | 0x20) == b'e' => 3,
            b'l' if pos + 2 < len && (bytes[pos + 2] | 0x20) == b'l' => 3,
            _ => 0,
        };

        if contraction_len > 0 {
            self.pos += contraction_len;
        }
        contraction_len
    }

    /// Unified whitespace scanner.
    ///
    /// Handles all whitespace modes:
    /// - `newline_anchored_whitespace` (CL100K/Llama/Qwen): `\s*[\r\n]+|\s+(?!\S)|\s+`
    /// - `newline_lookahead` (GPT-2): `\s+(?!\S)|\s`
    #[inline]
    fn scan_whitespace(&mut self, start: usize) {
        // Consume first whitespace char (ASCII, 1 byte)
        self.pos += 1;
        self.scan_whitespace_continue(start);
    }

    /// Continue scanning whitespace after the first character has been consumed.
    #[inline]
    fn scan_whitespace_continue(&mut self, start: usize) {
        let bytes = self.bytes;
        let len = self.len;

        // Continue consuming all whitespace
        while self.pos < len {
            let b = unsafe { *bytes.get_unchecked(self.pos) };
            match b {
                b' ' | b'\t' | b'\r' | b'\n' | 0x0B | 0x0C => {
                    self.pos += 1;
                }
                0x80..=0xFF => {
                    let (c, char_len) = decode_utf8(unsafe { bytes.get_unchecked(self.pos..) });
                    if c.is_whitespace() {
                        self.pos += char_len;
                    } else {
                        break;
                    }
                }
                _ => break,
            }
        }

        if self.config.newline_anchored_whitespace {
            // \s*[\r\n]+ pattern: find last newline in the consumed whitespace, split there.
            // This handles CL100K/Llama/Qwen patterns where whitespace runs anchor on newlines.
            for i in (start..self.pos).rev() {
                let b = unsafe { *bytes.get_unchecked(i) };
                if b == b'\r' || b == b'\n' {
                    self.pos = i + 1;
                    return;
                }
            }
            // No newline found: fall through to lookahead
        }

        // \s+(?!\S) pattern: greedily match whitespace, backtrack one character at a time
        // until the character after the match is NOT \S (i.e., is whitespace or end of string).
        // This handles both GPT-2 (\s+(?!\S)|\s) and CL100K (\s+(?!\S)|\s+) patterns.
        let consumed = self.pos - start;
        if consumed > 1 && self.pos < len {
            let next_b = unsafe { *bytes.get_unchecked(self.pos) };
            let is_next_ws = next_b == b' '
                || next_b == b'\t'
                || next_b == b'\r'
                || next_b == b'\n'
                || next_b == 0x0B
                || next_b == 0x0C
                || (next_b >= 0x80 && {
                    let (c, _) = decode_utf8(unsafe { bytes.get_unchecked(self.pos..) });
                    c.is_whitespace()
                });
            if !is_next_ws {
                // In multi-stage pretokenizers (DeepSeek), digits are extracted first,
                // so whitespace before digits should be kept together, not backtracked.
                let skip_backtrack = self.config.no_backtrack_before_digit
                    && (next_b.is_ascii_digit()
                        || (next_b >= 0x80 && {
                            let (c, _) = decode_utf8(unsafe { bytes.get_unchecked(self.pos..) });
                            is_unicode_number(c)
                        }));

                if !skip_backtrack {
                    // Back up by one character (not one byte) — find the start of the
                    // last character in the consumed whitespace run.
                    let mut back = self.pos - 1;
                    while back > start && (unsafe { *bytes.get_unchecked(back) } & 0xC0) == 0x80 {
                        back -= 1; // skip UTF-8 continuation bytes
                    }
                    // Only back up if we'd still have at least one character
                    if back > start {
                        self.pos = back;
                    }
                }
            }
        }
    }

    /// Handle space character.
    #[inline]
    fn handle_space(&mut self, start: usize) {
        let bytes = self.bytes;
        let len = self.len;

        if self.pos + 1 >= len {
            self.pos += 1;
            return;
        }

        let next_b = unsafe { *bytes.get_unchecked(self.pos + 1) };

        // Space + letters (handles both ASCII and Unicode letters)
        if self.try_scan_prefix_letters(1) {
            return;
        }

        // Handle other cases based on next byte
        match next_b {
            b'0'..=b'9' => {
                if self.config.space_prefix_numbers {
                    self.pos += 2;
                    self.scan_numbers(1);
                } else {
                    self.pos += 1; // Space alone
                }
            }
            b' ' | b'\t' | b'\r' | b'\n' => {
                self.scan_whitespace(start);
            }
            0x21..=0x7E => {
                // Other printable ASCII (punctuation, symbols)
                self.pos += 2;
                self.scan_other();
            }
            0x80..=0xFF => {
                // Unicode (non-letter, since try_scan_prefix_letters returned false)
                let (c, char_len) = decode_utf8(unsafe { bytes.get_unchecked(self.pos + 1..) });
                match classify_unicode(c) {
                    UnicodeClass::Number => {
                        if self.config.space_prefix_numbers {
                            self.pos += 1 + char_len;
                            self.scan_numbers(1);
                        } else {
                            self.pos += 1;
                        }
                    }
                    UnicodeClass::Whitespace => {
                        // In CL100K/Qwen regex, non-newline whitespace can prefix
                        // letters via [^\r\n\p{L}\p{N}]?\p{L}+. If \u{3000} is
                        // followed by a letter, emit space alone so \u{3000} starts
                        // a new chunk as a letter prefix.
                        if self.config.letter_prefix == LetterPrefix::NonNewlineNonAlnum {
                            let after = self.pos + 1 + char_len;
                            if after < self.len {
                                let ab = unsafe { *bytes.get_unchecked(after) };
                                let has_letter = is_ascii_letter(ab)
                                    || (ab >= 0x80 && {
                                        let (ac, _) = decode_utf8(unsafe {
                                            bytes.get_unchecked(after..)
                                        });
                                        is_unicode_letter(ac)
                                    });
                                if has_letter {
                                    self.pos += 1; // just the space
                                    return;
                                }
                            }
                        }
                        self.scan_whitespace(start);
                    }
                    UnicodeClass::Newline => {
                        self.scan_whitespace(start);
                    }
                    UnicodeClass::CJK | UnicodeClass::Punctuation | UnicodeClass::Other | UnicodeClass::Letter => {
                        // Letter case already handled by try_scan_prefix_letters
                        self.pos += 1 + char_len;
                        self.scan_other();
                    }
                }
            }
            _ => {
                // Control character
                self.pos += 1;
            }
        }
    }

    /// Handle other ASCII punctuation.
    #[inline]
    fn handle_other(&mut self, _start: usize) {
        // Check if this can be a prefix for letters (CL100K/O200K style)
        if self.config.letter_prefix == LetterPrefix::NonNewlineNonAlnum {
            if self.config.punct_prefix_ascii_only {
                // DeepSeek: [punct][A-Za-z]+ — only ASCII letters after punct
                if self.try_scan_prefix_ascii_letters(1) {
                    return;
                }
            } else if self.try_scan_prefix_letters(1) {
                return;
            }
        }
        // Just punctuation
        self.pos += 1;
        self.scan_other();
    }

    /// Handle Unicode character at current position.
    #[inline]
    fn handle_unicode(&mut self, start: usize) {
        let bytes = self.bytes;
        let (c, char_len) = decode_utf8(unsafe { bytes.get_unchecked(self.pos..) });
        self.pos += char_len;

        match classify_unicode(c) {
            UnicodeClass::CJK => {
                if self.config.individual_punctuation {
                    // BERT mode: each CJK character is its own word (no grouping)
                    // Just return the single character
                } else {
                    // Non-BERT: treat CJK as letters
                    self.scan_letters();
                    self.try_contraction_suffix_if_needed();
                }
            }
            UnicodeClass::Letter => {
                self.scan_letters();
                self.try_contraction_suffix_if_needed();
            }
            UnicodeClass::Number => self.scan_numbers(1),
            UnicodeClass::Newline => self.scan_whitespace_continue(start),
            UnicodeClass::Whitespace => {
                // In CL100K/Qwen/Llama regex, [^\r\n\p{L}\p{N}]?\p{L}+ matches non-newline
                // whitespace as a letter prefix (the class excludes \r\n but not \s).
                // So \u{3000}a → one chunk, not two.
                if self.config.letter_prefix == LetterPrefix::NonNewlineNonAlnum
                    && self.try_scan_prefix_letters(0)
                {
                    return;
                }
                // First char already consumed (self.pos += char_len above).
                // Continue scanning any remaining whitespace.
                self.scan_whitespace_continue(start);
            }
            UnicodeClass::Punctuation => {
                if self.config.individual_punctuation {
                    // BERT mode: each punctuation is a separate token
                    // Just return the single character (already consumed)
                } else {
                    // Check for letter prefix (CL100K/O200K style)
                    if self.config.letter_prefix == LetterPrefix::NonNewlineNonAlnum
                        && self.try_scan_prefix_letters(0)
                    {
                        return;
                    }
                    self.scan_other();
                }
            }
            UnicodeClass::Other => {
                if self.config.individual_punctuation {
                    // BERT mode: non-punct/non-letter chars (marks, symbols) attach to words
                    // Continue scanning as part of the word
                    self.scan_bert_word_chars();
                } else {
                    // Check for letter prefix (CL100K/O200K style)
                    if self.config.letter_prefix == LetterPrefix::NonNewlineNonAlnum
                        && self.try_scan_prefix_letters(0)
                    {
                        return;
                    }
                    self.scan_other();
                }
            }
        }
    }

    /// Check for contraction pattern.
    #[inline]
    fn check_contraction(&self) -> usize {
        let bytes = self.bytes;
        let pos = self.pos;
        let len = self.len;

        if pos + 1 >= len {
            return 0;
        }

        let mut b = unsafe { *bytes.get_unchecked(pos + 1) };
        if self.config.case_insensitive_contractions {
            b |= 0x20; // lowercase
        }

        match b {
            b's' | b't' | b'm' | b'd' => 2,
            b'r' if pos + 2 < len => {
                let mut b2 = unsafe { *bytes.get_unchecked(pos + 2) };
                if self.config.case_insensitive_contractions {
                    b2 |= 0x20;
                }
                if b2 == b'e' {
                    3
                } else {
                    0
                }
            }
            b'v' if pos + 2 < len => {
                let mut b2 = unsafe { *bytes.get_unchecked(pos + 2) };
                if self.config.case_insensitive_contractions {
                    b2 |= 0x20;
                }
                if b2 == b'e' {
                    3
                } else {
                    0
                }
            }
            b'l' if pos + 2 < len => {
                let mut b2 = unsafe { *bytes.get_unchecked(pos + 2) };
                if self.config.case_insensitive_contractions {
                    b2 |= 0x20;
                }
                if b2 == b'l' {
                    3
                } else {
                    0
                }
            }
            _ => 0,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpt2_basic() {
        assert_eq!(Pretok::GPT2.split_to_vec("Hello world"), vec!["Hello", " world"]);
    }

    #[test]
    fn test_gpt2_contractions() {
        assert_eq!(Pretok::GPT2.split_to_vec("don't"), vec!["don", "'t"]);
        assert_eq!(Pretok::GPT2.split_to_vec("DON'T"), vec!["DON", "'", "T"]); // case-sensitive
    }

    #[test]
    fn test_gpt2_numbers() {
        assert_eq!(Pretok::GPT2.split_to_vec("12345"), vec!["12345"]); // no chunking
        assert_eq!(Pretok::GPT2.split_to_vec("test 123"), vec!["test", " 123"]); // space prefix
    }

    #[test]
    fn test_gpt2_whitespace() {
        assert_eq!(Pretok::GPT2.split_to_vec("a  b"), vec!["a", " ", " b"]);
    }

    #[test]
    fn test_cl100k_basic() {
        assert_eq!(Pretok::CL100K.split_to_vec("Hello world"), vec!["Hello", " world"]);
    }

    #[test]
    fn test_cl100k_contractions() {
        assert_eq!(Pretok::CL100K.split_to_vec("don't"), vec!["don", "'t"]);
        assert_eq!(Pretok::CL100K.split_to_vec("DON'T"), vec!["DON", "'T"]); // case-insensitive
    }

    #[test]
    fn test_cl100k_numbers() {
        assert_eq!(Pretok::CL100K.split_to_vec("12345"), vec!["123", "45"]); // chunked
        assert_eq!(Pretok::CL100K.split_to_vec("test 123"), vec!["test", " ", "123"]); // no space prefix
    }

    #[test]
    fn test_cl100k_letter_prefix() {
        assert_eq!(Pretok::CL100K.split_to_vec(".hello"), vec![".hello"]); // punct + letters
    }

    #[test]
    fn test_unicode() {
        assert_eq!(Pretok::GPT2.split_to_vec("Привет мир"), vec!["Привет", " мир"]);
        assert_eq!(Pretok::CL100K.split_to_vec("你好世界"), vec!["你好世界"]);
    }

    // O200K tests
    #[test]
    fn test_o200k_basic() {
        assert_eq!(Pretok::O200K.split_to_vec("Hello world"), vec!["Hello", " world"]);
    }

    #[test]
    fn test_o200k_contractions_suffix() {
        // O200K: contractions attach to preceding word
        assert_eq!(Pretok::O200K.split_to_vec("don't"), vec!["don't"]);
        assert_eq!(Pretok::O200K.split_to_vec("DON'T"), vec!["DON'T"]);
        assert_eq!(Pretok::O200K.split_to_vec("I'm"), vec!["I'm"]);
        assert_eq!(Pretok::O200K.split_to_vec("we'll"), vec!["we'll"]);
        assert_eq!(Pretok::O200K.split_to_vec("they're"), vec!["they're"]);
        assert_eq!(Pretok::O200K.split_to_vec("you've"), vec!["you've"]);
        assert_eq!(Pretok::O200K.split_to_vec("she'd"), vec!["she'd"]);
    }

    #[test]
    fn test_o200k_camelcase() {
        // O200K: case-aware scanning - splits on Upper->Lower boundary when followed by uppercase
        assert_eq!(Pretok::O200K.split_to_vec("CamelCase"), vec!["Camel", "Case"]);
        assert_eq!(Pretok::O200K.split_to_vec("JSONParser"), vec!["JSONParser"]); // no split - all uppercase before lowercase
        assert_eq!(Pretok::O200K.split_to_vec("parseJSON"), vec!["parse", "JSON"]);
        assert_eq!(Pretok::O200K.split_to_vec("XMLHttpRequest"), vec!["XMLHttp", "Request"]);
    }

    #[test]
    fn test_o200k_numbers() {
        // O200K: chunked numbers (same as CL100K)
        assert_eq!(Pretok::O200K.split_to_vec("12345"), vec!["123", "45"]);
        assert_eq!(Pretok::O200K.split_to_vec("test 123"), vec!["test", " ", "123"]);
    }

    #[test]
    fn test_o200k_apostrophe_only() {
        // O200K: apostrophe can prefix letters via [^\r\n\p{L}\p{N}]?\p{L}+
        assert_eq!(Pretok::O200K.split_to_vec("'hello'"), vec!["'hello", "'"]);
    }

    // BERT tests
    #[test]
    fn test_bert_basic() {
        // BERT: whitespace is delimiter only, not included in tokens
        assert_eq!(Pretok::BERT.split_to_vec("Hello world"), vec!["Hello", "world"]);
        assert_eq!(Pretok::BERT.split_to_vec("hello, world!"), vec!["hello", ",", "world", "!"]);
    }

    #[test]
    fn test_bert_whitespace_stripped() {
        // BERT: all whitespace is stripped
        assert_eq!(Pretok::BERT.split_to_vec("  hello  world  "), vec!["hello", "world"]);
        assert_eq!(Pretok::BERT.split_to_vec("a\tb\nc"), vec!["a", "b", "c"]);
    }

    #[test]
    fn test_bert_punctuation() {
        // BERT: punctuation is split into separate tokens
        assert_eq!(Pretok::BERT.split_to_vec("hello,world"), vec!["hello", ",", "world"]);
        assert_eq!(Pretok::BERT.split_to_vec("test."), vec!["test", "."]);
        assert_eq!(Pretok::BERT.split_to_vec("(test)"), vec!["(", "test", ")"]);
    }

    #[test]
    fn test_bert_numbers() {
        // BERT: numbers are not chunked
        assert_eq!(Pretok::BERT.split_to_vec("12345"), vec!["12345"]);
        assert_eq!(Pretok::BERT.split_to_vec("test 123"), vec!["test", "123"]);
    }

    #[test]
    fn test_bert_individual_punctuation() {
        // BERT: each punctuation is a separate token (not grouped)
        assert_eq!(Pretok::BERT.split_to_vec("***"), vec!["*", "*", "*"]);
        assert_eq!(Pretok::BERT.split_to_vec("hello***world"), vec!["hello", "*", "*", "*", "world"]);
        assert_eq!(Pretok::BERT.split_to_vec("..."), vec![".", ".", "."]);
        assert_eq!(Pretok::BERT.split_to_vec("a!!b"), vec!["a", "!", "!", "b"]);
    }

    #[test]
    fn test_bert_alphanumeric_together() {
        // BERT: letters and numbers are kept together
        assert_eq!(Pretok::BERT.split_to_vec("18th"), vec!["18th"]);
        assert_eq!(Pretok::BERT.split_to_vec("the 18th brumaire"), vec!["the", "18th", "brumaire"]);
        assert_eq!(Pretok::BERT.split_to_vec("a1b2c3"), vec!["a1b2c3"]);
        assert_eq!(Pretok::BERT.split_to_vec("123abc456"), vec!["123abc456"]);
        assert_eq!(Pretok::BERT.split_to_vec("test123"), vec!["test123"]);
    }

    // Voyage tests
    #[test]
    fn test_voyage_basic() {
        assert_eq!(Pretok::VOYAGE.split_to_vec("Hello world"), vec!["Hello", " world"]);
    }

    #[test]
    fn test_voyage_numbers() {
        // Voyage: single digits (unlike CL100K's groups of 3)
        assert_eq!(Pretok::VOYAGE.split_to_vec("12345"), vec!["1", "2", "3", "4", "5"]);
        assert_eq!(Pretok::VOYAGE.split_to_vec("test 123"), vec!["test", " ", "1", "2", "3"]);
    }

    #[test]
    fn test_voyage_tab_prefix() {
        // Voyage (like CL100K): tabs can prefix letters
        assert_eq!(Pretok::VOYAGE.split_to_vec("\ttabs"), vec!["\ttabs"]);
        assert_eq!(Pretok::VOYAGE.split_to_vec("\t\ttabs\tand"), vec!["\t", "\ttabs", "\tand"]);
    }

    #[test]
    fn test_voyage_contractions() {
        // Voyage: case-insensitive standalone contractions
        assert_eq!(Pretok::VOYAGE.split_to_vec("don't"), vec!["don", "'t"]);
        assert_eq!(Pretok::VOYAGE.split_to_vec("DON'T"), vec!["DON", "'T"]);
    }

    // BERT CJK tests
    #[test]
    fn test_bert_cjk_individual() {
        // BERT: each CJK character is a separate token
        assert_eq!(Pretok::BERT.split_to_vec("你好世界"), vec!["你", "好", "世", "界"]);
    }

    #[test]
    fn test_bert_cjk_mixed_with_ascii() {
        // CJK chars break words
        assert_eq!(Pretok::BERT.split_to_vec("hello你好world"), vec!["hello", "你", "好", "world"]);
        assert_eq!(Pretok::BERT.split_to_vec("test政策test"), vec!["test", "政", "策", "test"]);
    }

    #[test]
    fn test_bert_cjk_with_punctuation() {
        assert_eq!(Pretok::BERT.split_to_vec("你好!"), vec!["你", "好", "!"]);
        assert_eq!(Pretok::BERT.split_to_vec("[你好]"), vec!["[", "你", "好", "]"]);
    }

    #[test]
    fn test_bert_cjk_spacing_marks() {
        // Tamil: spacing marks (Mc) should stay with letters, not split
        // மதியிய = ம + த + ி(Mc) + ய + ி(Mc) + ய → single word
        assert_eq!(Pretok::BERT.split_to_vec("மதியிய"), vec!["மதியிய"]);
    }

    #[test]
    fn test_bert_unicode_punctuation() {
        // Unicode punctuation should be split (like ASCII punct)
        // « and » are InitialPunctuation/FinalPunctuation
        assert_eq!(Pretok::BERT.split_to_vec("«hello»"), vec!["«", "hello", "»"]);
    }

    #[test]
    fn test_non_bert_cjk_grouped() {
        // Non-BERT modes: CJK characters are grouped as letters
        assert_eq!(Pretok::CL100K.split_to_vec("你好世界"), vec!["你好世界"]);
        assert_eq!(Pretok::GPT2.split_to_vec("你好世界"), vec!["你好世界"]);
    }

    // CL100K tab prefix tests (regression)
    #[test]
    fn test_cl100k_tab_prefix() {
        // CL100K: tabs can prefix letters
        assert_eq!(Pretok::CL100K.split_to_vec("\ttabs"), vec!["\ttabs"]);
        assert_eq!(Pretok::CL100K.split_to_vec("\t\ttabs\tand"), vec!["\t", "\ttabs", "\tand"]);
    }

    // \u{3000} (ideographic space) tests — it's whitespace but matches [^\r\n\p{L}\p{N}]
    // so CL100K regex treats it as a valid letter prefix
    #[test]
    fn test_cl100k_ideographic_space() {
        // Alone
        assert_eq!(Pretok::CL100K.split_to_vec("\u{3000}"), vec!["\u{3000}"]);
        // As letter prefix
        assert_eq!(Pretok::CL100K.split_to_vec("\u{3000}a"), vec!["\u{3000}a"]);
        assert_eq!(Pretok::CL100K.split_to_vec("\u{3000}abc"), vec!["\u{3000}abc"]);
        assert_eq!(Pretok::CL100K.split_to_vec("\u{3000}("), vec!["\u{3000}", "("]);
        // After letter
        assert_eq!(Pretok::CL100K.split_to_vec("a\u{3000}"), vec!["a", "\u{3000}"]);
        assert_eq!(Pretok::CL100K.split_to_vec("abc\u{3000}def"), vec!["abc", "\u{3000}def"]);
        assert_eq!(Pretok::CL100K.split_to_vec("Hello\u{3000}World"), vec!["Hello", "\u{3000}World"]);
        // Space + ideographic space + letter: space alone, then \u{3000} prefixes letter
        assert_eq!(Pretok::CL100K.split_to_vec(" \u{3000}a"), vec![" ", "\u{3000}a"]);
        // Two ideographic spaces + letter: first alone, second prefixes letter
        assert_eq!(Pretok::CL100K.split_to_vec("\u{3000}\u{3000}a"), vec!["\u{3000}", "\u{3000}a"]);
        // Whitespace-only runs (no letter follows)
        assert_eq!(Pretok::CL100K.split_to_vec(" \u{3000}"), vec![" \u{3000}"]);
        assert_eq!(Pretok::CL100K.split_to_vec("\u{3000}\u{3000}\u{3000}"), vec!["\u{3000}\u{3000}\u{3000}"]);
    }
}
