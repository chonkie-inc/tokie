//! Generic pretokenizer iterator parameterized by `PretokConfig`.
//!
//! All BPE-style pretokenizers (GPT-2, CL100K, O200K, Voyage, SmolLM, DeepSeek, Qwen)
//! are monomorphizations of `Core<C>` with different config constants. The compiler
//! const-propagates all config checks and dead-code-eliminates unreachable branches,
//! producing identical machine code to hand-written per-type implementations.

use std::marker::PhantomData;

use crate::core::config::*;
use crate::util::{decode_utf8, is_ascii_letter, is_digit, is_lower, is_upper, is_unicode_letter, is_unicode_mark};

pub struct Core<'a, C: PretokConfig> {
    bytes: &'a [u8],
    pos: usize,
    len: usize,
    _cfg: PhantomData<C>,
}

impl<'a, C: PretokConfig> Core<'a, C> {
    pub fn new(text: &'a str) -> Self {
        let bytes = text.as_bytes();
        Self { bytes, pos: 0, len: bytes.len(), _cfg: PhantomData }
    }

    #[inline(always)]
    fn at(&self, pos: usize) -> u8 {
        unsafe { *self.bytes.get_unchecked(pos) }
    }

    #[inline(always)]
    fn emit(&self, start: usize) -> &'a str {
        unsafe { std::str::from_utf8_unchecked(&self.bytes[start..self.pos]) }
    }

    // ---- Letter scanning ----

    /// Scan letters: `\p{L}+` (Plain) or `[\p{L}\p{M}]+` (PlainWithMarks).
    /// Not used for CamelCase — that calls `scan_letters_case_aware` instead.
    #[inline(always)]
    fn scan_letters(&mut self) {
        while self.pos < self.len {
            let b = self.at(self.pos);
            if is_ascii_letter(b) {
                self.pos += 1;
            } else if b >= 0x80 {
                let (ch, cl) = decode_utf8(&self.bytes[self.pos..]);
                if C::LETTER_MODE == LetterMode::PlainWithMarks {
                    if ch.is_alphabetic() || is_unicode_mark(ch) {
                        self.pos += cl;
                    } else { return; }
                } else {
                    if is_unicode_letter(ch) { self.pos += cl; } else { return; }
                }
            } else {
                return;
            }
        }
    }

    /// O200K CamelCase: dispatch based on first byte's case.
    #[inline(always)]
    fn scan_letters_case_aware(&mut self, first: u8) {
        if first < 0x80 {
            if is_lower(first) {
                self.scan_lowercase();
            } else {
                self.scan_upper_then_lower();
            }
        } else {
            let start = self.pos - 1;
            let (ch, _) = decode_utf8(&self.bytes[start..]);
            if ch.is_lowercase() {
                self.scan_lowercase();
            } else {
                self.scan_upper_then_lower();
            }
        }
    }

    /// Scan `[\p{Ll}\p{Lm}\p{Lo}\p{M}]+`
    #[inline(always)]
    fn scan_lowercase(&mut self) {
        while self.pos < self.len {
            let b = self.at(self.pos);
            if is_lower(b) {
                self.pos += 1;
            } else if b >= 0x80 {
                let (ch, cl) = decode_utf8(&self.bytes[self.pos..]);
                if is_unicode_mark(ch) || ch.is_lowercase()
                    || (is_unicode_letter(ch) && !ch.is_uppercase())
                {
                    self.pos += cl;
                } else { return; }
            } else { return; }
        }
    }

    /// Scan upper/titlecase/modifier/other letters, then lowercase+marks.
    #[inline(always)]
    fn scan_upper_then_lower(&mut self) {
        while self.pos < self.len {
            let b = self.at(self.pos);
            if is_upper(b) {
                self.pos += 1;
            } else if is_lower(b) {
                self.pos += 1;
                self.scan_lowercase();
                return;
            } else if b >= 0x80 {
                let (ch, cl) = decode_utf8(&self.bytes[self.pos..]);
                if ch.is_uppercase() {
                    self.pos += cl;
                } else if ch.is_lowercase() || is_unicode_mark(ch) {
                    self.pos += cl;
                    self.scan_lowercase();
                    return;
                } else if is_unicode_letter(ch) {
                    self.pos += cl;
                } else { return; }
            } else { return; }
        }
    }

    /// Scan ASCII-only letters (DeepSeek: punct prefix → ASCII letters only).
    #[inline(always)]
    fn scan_ascii_letters(&mut self) {
        while self.pos < self.len && is_ascii_letter(self.at(self.pos)) {
            self.pos += 1;
        }
    }

    // ---- Digit scanning ----

    /// Scan remaining digits after the first digit byte is already consumed.
    #[inline(always)]
    fn scan_digits(&mut self) {
        match C::DIGIT_MODE {
            DigitMode::Unlimited => {
                while self.pos < self.len && is_digit(self.at(self.pos)) {
                    self.pos += 1;
                }
            }
            DigitMode::Chunked3 => {
                // First digit already consumed; scan up to 2 more
                let mut count = 1u8;
                while self.pos < self.len && count < 3 && is_digit(self.at(self.pos)) {
                    self.pos += 1;
                    count += 1;
                }
            }
            DigitMode::Single => {
                // Single digit already consumed by caller
            }
        }
    }

    // ---- Contraction detection ----

    #[inline(always)]
    fn check_contraction(&self) -> usize {
        if self.pos >= self.len || self.bytes[self.pos] != b'\'' { return 0; }
        let rem = self.len - self.pos;
        if rem < 2 { return 0; }
        let b1 = if C::CONTRACTION_CASE == ContractionCase::Insensitive {
            self.bytes[self.pos + 1] | 0x20
        } else {
            self.bytes[self.pos + 1]
        };
        if matches!(b1, b's' | b't' | b'd' | b'm') {
            if rem == 2 || !is_ascii_letter(self.bytes[self.pos + 2]) {
                return 2;
            }
        }
        if rem < 3 { return 0; }
        let b2 = if C::CONTRACTION_CASE == ContractionCase::Insensitive {
            self.bytes[self.pos + 2] | 0x20
        } else {
            self.bytes[self.pos + 2]
        };
        if (b1 == b'l' && b2 == b'l')
            || (b1 == b'v' && b2 == b'e')
            || (b1 == b'r' && b2 == b'e')
        {
            return 3;
        }
        0
    }

    // ---- Punctuation scanning ----

    #[inline(always)]
    fn is_punct_byte(b: u8) -> bool {
        !is_ascii_letter(b) && !is_digit(b) && b != b' ' && b != b'\t' && b != b'\n' && b != b'\r' && b < 0x80
    }

    /// Check if a Unicode char is a non-letter/non-digit/non-whitespace symbol.
    #[inline(always)]
    fn is_unicode_punct_char(ch: char) -> bool {
        if C::LETTER_MODE == LetterMode::PlainWithMarks || C::LETTER_MODE == LetterMode::CamelCase {
            !ch.is_alphabetic() && !ch.is_numeric() && !ch.is_whitespace() && !is_unicode_mark(ch)
        } else {
            !is_unicode_letter(ch) && !ch.is_numeric() && !ch.is_whitespace()
        }
    }

    #[inline(always)]
    fn scan_punct(&mut self) {
        // Scan punct chars
        while self.pos < self.len {
            let b = self.at(self.pos);
            if Self::is_punct_byte(b) {
                self.pos += 1;
            } else if b >= 0x80 {
                let (ch, cl) = decode_utf8(&self.bytes[self.pos..]);
                if Self::is_unicode_punct_char(ch) {
                    self.pos += cl;
                } else { break; }
            } else { break; }
        }
        // Trailing newlines
        if C::PUNCT_TRAILING == PunctTrailing::Newlines || C::PUNCT_TRAILING == PunctTrailing::NewlinesAndSlashes {
            while self.pos < self.len {
                let b = self.at(self.pos);
                if b == b'\n' || b == b'\r' {
                    self.pos += 1;
                } else if C::PUNCT_TRAILING == PunctTrailing::NewlinesAndSlashes && b == b'/' {
                    self.pos += 1;
                } else { break; }
            }
        }
    }

    // ---- Whitespace scanning ----

    #[inline(always)]
    fn scan_whitespace(&mut self) {
        if C::WS_PATTERN == WsPattern::Cl100k {
            self.scan_whitespace_cl100k();
        } else {
            self.scan_whitespace_gpt2();
        }
    }

    /// GPT-2 style: `\s+(?!\S)|\s+` — greedy, back up one if followed by non-WS.
    #[inline(always)]
    fn scan_whitespace_gpt2(&mut self) {
        let start = self.pos;
        let mut prev_pos = self.pos;
        while self.pos < self.len {
            let c = self.at(self.pos);
            if c == b' ' || c == b'\n' || c == b'\r' || c == b'\t' {
                prev_pos = self.pos;
                self.pos += 1;
            } else if c >= 0x80 {
                let (ch, cl) = decode_utf8(&self.bytes[self.pos..]);
                if ch.is_whitespace() { prev_pos = self.pos; self.pos += cl; } else { break; }
            } else {
                break;
            }
        }
        if self.pos < self.len && prev_pos > start {
            if C::WS_EXCEPTION == WsException::Digits {
                // SmolLM: don't back up before digits
                if !is_digit(self.at(self.pos)) {
                    self.pos = prev_pos;
                }
            } else {
                self.pos = prev_pos;
            }
        }
    }

    /// CL100K style: `\s*[\r\n]|\s+(?!\S)|\s+` — prioritize newlines.
    #[inline(always)]
    fn scan_whitespace_cl100k(&mut self) {
        let start = self.pos;
        let mut last_newline_end = 0usize;
        let mut prev_pos = self.pos;
        while self.pos < self.len {
            let c = self.at(self.pos);
            if c == b'\n' || c == b'\r' {
                prev_pos = self.pos;
                self.pos += 1;
                last_newline_end = self.pos;
            } else if c == b' ' || c == b'\t' {
                prev_pos = self.pos;
                self.pos += 1;
            } else if c >= 0x80 {
                let (ch, cl) = decode_utf8(&self.bytes[self.pos..]);
                if ch.is_whitespace() { prev_pos = self.pos; self.pos += cl; } else { break; }
            } else {
                break;
            }
        }
        if last_newline_end > 0 {
            self.pos = last_newline_end;
        } else if self.pos < self.len && prev_pos > start {
            if C::WS_EXCEPTION == WsException::Cjk {
                // DeepSeek: don't back up before CJK
                let next = self.at(self.pos);
                if !is_digit(next) && !(next >= 0x80 && Self::is_cjk_start(&self.bytes[self.pos..])) {
                    self.pos = prev_pos;
                }
            } else {
                self.pos = prev_pos;
            }
        }
    }

    /// Check if bytes start a CJK character.
    #[inline(always)]
    fn is_cjk_start(bytes: &[u8]) -> bool {
        if bytes[0] < 0xE0 { return false; }
        let (ch, _) = decode_utf8(bytes);
        let cp = ch as u32;
        matches!(cp, 0x4E00..=0x9FA5 | 0x3040..=0x309F | 0x30A0..=0x30FF)
    }

    // ---- Letter entry helpers (handles contraction after letters) ----

    /// After scanning letters, handle contraction check based on config.
    /// Returns true if we should early-return (standalone contraction found).
    #[inline(always)]
    fn handle_post_letters(&mut self, start: usize) -> Option<&'a str> {
        if C::CONTRACTION_MODE == ContractionMode::Suffix {
            let clen = self.check_contraction();
            if clen > 0 { self.pos += clen; }
            None // suffix: contraction merged, no early return
        } else {
            // Standalone: if contraction follows, DON'T consume it — emit word only
            if self.check_contraction() > 0 {
                Some(self.emit(start))
            } else {
                None
            }
        }
    }

    /// Scan letters starting from current position (first letter byte already consumed).
    #[inline(always)]
    fn do_letter_scan(&mut self, first: u8) {
        if C::LETTER_MODE == LetterMode::CamelCase {
            self.scan_letters_case_aware(first);
        } else {
            self.scan_letters();
        }
    }

    /// Check if a Unicode char is a letter for this config.
    #[inline(always)]
    fn is_letter_char(ch: char) -> bool {
        if C::LETTER_MODE == LetterMode::PlainWithMarks || C::LETTER_MODE == LetterMode::CamelCase {
            ch.is_alphabetic() || is_unicode_mark(ch)
        } else {
            is_unicode_letter(ch)
        }
    }

    // ---- Punct prefix: scan letters after a prefix char ----

    /// After a non-alnum prefix char, scan following letters.
    /// DeepSeek: only ASCII letters. Others: full unicode letters.
    #[inline(always)]
    fn scan_letters_after_punct_prefix(&mut self, next: u8) {
        if C::PUNCT_PREFIX_MODE == PunctPrefixMode::AsciiOnly {
            self.pos += 1; // consume the ASCII letter
            self.scan_ascii_letters();
        } else {
            self.pos += 1; // consume the letter byte
            self.do_letter_scan(next);
        }
    }
}

// ---- Iterator implementation ----

impl<'a, C: PretokConfig> Iterator for Core<'a, C> {
    type Item = &'a str;

    #[inline(always)]
    fn next(&mut self) -> Option<&'a str> {
        if self.pos >= self.len { return None; }

        let start = self.pos;
        let b = self.at(self.pos);

        if is_ascii_letter(b) {
            // Letter start
            self.pos += 1;
            self.do_letter_scan(b);
            if let Some(piece) = self.handle_post_letters(start) {
                return Some(piece);
            }
        } else if b == b'\'' {
            // Apostrophe: contraction or punct prefix
            let clen = self.check_contraction();
            if clen > 0 && C::CONTRACTION_MODE == ContractionMode::Standalone {
                self.pos += clen;
            } else if clen == 0 || C::CONTRACTION_MODE == ContractionMode::Suffix {
                // Not a standalone contraction — treat as prefix or punct
                if self.pos + 1 < self.len {
                    let next = self.at(self.pos + 1);
                    if is_ascii_letter(next) {
                        self.pos += 1; // skip apostrophe prefix
                        self.pos += 1; // consume first letter
                        self.do_letter_scan(next);
                        if let Some(piece) = self.handle_post_letters(start) {
                            return Some(piece);
                        }
                    } else if next >= 0x80 {
                        let (ch, _) = decode_utf8(&self.bytes[self.pos + 1..]);
                        if Self::is_letter_char(ch) {
                            self.pos += 1; // skip apostrophe prefix
                            self.scan_letters();
                            if C::LETTER_MODE == LetterMode::CamelCase {
                                // For O200K, we need case-aware scan from the Unicode char
                                let clen2 = self.check_contraction();
                                if clen2 > 0 { self.pos += clen2; }
                            } else if let Some(piece) = self.handle_post_letters(start) {
                                return Some(piece);
                            }
                        } else {
                            self.pos += 1;
                            self.scan_punct();
                        }
                    } else {
                        self.pos += 1;
                        self.scan_punct();
                    }
                } else {
                    self.pos += 1;
                }
            }
        } else if is_digit(b) {
            if C::DIGIT_MODE == DigitMode::Single {
                self.pos += 1;
            } else {
                self.pos += 1;
                self.scan_digits();
            }
        } else if b == b' ' || b == b'\t' {
            // Space/tab: prefix letters, prefix punct, or whitespace run
            if self.pos + 1 < self.len {
                let next = self.at(self.pos + 1);
                if is_ascii_letter(next) {
                    self.pos += 2;
                    self.do_letter_scan(next);
                    if let Some(piece) = self.handle_post_letters(start) {
                        return Some(piece);
                    }
                } else if next >= 0x80 {
                    let (ch, _) = decode_utf8(&self.bytes[self.pos + 1..]);
                    if Self::is_letter_char(ch) {
                        self.pos += 1; // consume space prefix
                        if C::LETTER_MODE == LetterMode::CamelCase {
                            let (_, cl) = decode_utf8(&self.bytes[self.pos..]);
                            self.pos += cl;
                            if ch.is_lowercase() || is_unicode_mark(ch) {
                                self.scan_lowercase();
                            } else {
                                self.scan_upper_then_lower();
                            }
                            let clen = self.check_contraction();
                            if clen > 0 { self.pos += clen; }
                        } else {
                            self.scan_letters();
                            if let Some(piece) = self.handle_post_letters(start) {
                                return Some(piece);
                            }
                        }
                    } else if ch.is_whitespace() || ch.is_numeric() {
                        self.scan_whitespace();
                    } else {
                        self.pos += 1;
                        self.scan_punct();
                    }
                } else if C::SPACE_PREFIXES_DIGITS && is_digit(next) {
                    // GPT-2: space prefixes digits
                    self.pos += 2;
                    self.scan_digits();
                } else if Self::is_punct_byte(next) || next == b'\'' {
                    if C::PUNCT_PREFIX_MODE != PunctPrefixMode::SpaceOnly || next == b'\'' {
                        // CL100K/O200K/Voyage/Qwen: space prefixes punct
                        self.pos += 1;
                        self.scan_punct();
                    } else {
                        // GPT-2/SmolLM: space + punct → space prefixes punct group
                        self.pos += 2;
                        self.scan_punct();
                    }
                } else if is_digit(next) {
                    // Space before digit, no prefix (CL100K/Voyage/SmolLM etc.)
                    if C::WS_PATTERN == WsPattern::Cl100k {
                        self.scan_whitespace();
                    } else {
                        // GPT-2 style: SmolLM emits bare space before digit
                        self.pos += 1;
                    }
                } else {
                    // Space + more whitespace
                    self.scan_whitespace();
                }
            } else {
                self.pos += 1;
            }
        } else if b == b'\n' || b == b'\r' {
            if C::WS_PATTERN == WsPattern::Cl100k {
                // CL100K: consume all consecutive newlines
                self.pos += 1;
                while self.pos < self.len {
                    let c = self.at(self.pos);
                    if c == b'\n' || c == b'\r' { self.pos += 1; }
                    else { break; }
                }
            } else {
                // GPT-2: whitespace run with lookahead
                self.scan_whitespace();
            }
        } else if b >= 0x80 {
            let (ch, cl) = decode_utf8(&self.bytes[self.pos..]);
            if Self::is_letter_char(ch) {
                self.pos += cl;
                if C::LETTER_MODE == LetterMode::CamelCase {
                    if ch.is_lowercase() { self.scan_lowercase(); } else { self.scan_upper_then_lower(); }
                } else {
                    self.scan_letters();
                }
                if let Some(piece) = self.handle_post_letters(start) {
                    return Some(piece);
                }
            } else if is_unicode_mark(ch) && C::LETTER_MODE == LetterMode::CamelCase {
                // O200K: mark at start → scan as lowercase
                self.pos += cl;
                self.scan_lowercase();
                let clen = self.check_contraction();
                if clen > 0 { self.pos += clen; }
            } else if ch.is_numeric() {
                self.pos += cl;
                self.scan_digits();
            } else if ch.is_whitespace() {
                self.scan_whitespace();
            } else {
                // Non-ASCII symbol: try prefix letters, else punct group
                self.pos += cl;
                if self.pos < self.len {
                    let next = self.at(self.pos);
                    if is_ascii_letter(next) {
                        self.scan_letters_after_punct_prefix(next);
                        if C::LETTER_MODE == LetterMode::CamelCase {
                            let clen = self.check_contraction();
                            if clen > 0 { self.pos += clen; }
                        } else if let Some(piece) = self.handle_post_letters(start) {
                            return Some(piece);
                        }
                    } else if next >= 0x80 {
                        let (ch2, _) = decode_utf8(&self.bytes[self.pos..]);
                        if Self::is_letter_char(ch2) {
                            if C::LETTER_MODE == LetterMode::CamelCase {
                                self.scan_lowercase();
                                let clen = self.check_contraction();
                                if clen > 0 { self.pos += clen; }
                            } else {
                                self.scan_letters();
                            }
                        } else if Self::is_unicode_punct_char(ch2) {
                            self.scan_punct();
                        }
                    } else if Self::is_punct_byte(next) {
                        self.scan_punct();
                    }
                }
            }
        } else {
            // Other ASCII punct
            if C::PUNCT_PREFIX_MODE == PunctPrefixMode::SpaceOnly {
                // GPT-2/SmolLM: punct doesn't prefix letters, just scan punct group
                self.pos += 1;
                self.scan_punct();
            } else if self.pos + 1 < self.len {
                let next = self.at(self.pos + 1);
                if is_ascii_letter(next) {
                    self.pos += 1; // skip punct prefix
                    self.scan_letters_after_punct_prefix(next);
                    if C::LETTER_MODE == LetterMode::CamelCase {
                        let clen = self.check_contraction();
                        if clen > 0 { self.pos += clen; }
                    } else if let Some(piece) = self.handle_post_letters(start) {
                        return Some(piece);
                    }
                } else if next >= 0x80 {
                    let (ch, cl) = decode_utf8(&self.bytes[self.pos + 1..]);
                    if Self::is_letter_char(ch) {
                        self.pos += 1; // skip punct prefix
                        if C::LETTER_MODE == LetterMode::CamelCase {
                            self.pos += cl;
                            if ch.is_lowercase() || is_unicode_mark(ch) {
                                self.scan_lowercase();
                            } else {
                                self.scan_upper_then_lower();
                            }
                            let clen = self.check_contraction();
                            if clen > 0 { self.pos += clen; }
                        } else {
                            self.scan_letters();
                            if let Some(piece) = self.handle_post_letters(start) {
                                return Some(piece);
                            }
                        }
                    } else {
                        self.pos += 1;
                        self.scan_punct();
                    }
                } else {
                    self.pos += 1;
                    self.scan_punct();
                }
            } else {
                self.pos += 1;
            }
        }

        debug_assert!(self.pos > start, "no progress at pos {start}");
        Some(self.emit(start))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::configs::*;

    type Gpt2<'a> = Core<'a, Gpt2Config>;
    type Cl100k<'a> = Core<'a, Cl100kConfig>;
    type O200k<'a> = Core<'a, O200kConfig>;
    type Voyage<'a> = Core<'a, VoyageConfig>;
    type SmolLM<'a> = Core<'a, SmolLMConfig>;
    type DeepSeek<'a> = Core<'a, DeepSeekConfig>;
    type Qwen<'a> = Core<'a, QwenConfig>;

    // GPT-2
    #[test] fn gpt2_basic() { assert_eq!(Gpt2::new("Hello world").collect::<Vec<_>>(), vec!["Hello", " world"]); }
    #[test] fn gpt2_punct() { assert_eq!(Gpt2::new("Hello, world!").collect::<Vec<_>>(), vec!["Hello", ",", " world", "!"]); }
    #[test] fn gpt2_numbers() { assert_eq!(Gpt2::new("test 123").collect::<Vec<_>>(), vec!["test", " 123"]); }
    #[test] fn gpt2_contraction() { assert_eq!(Gpt2::new("don't").collect::<Vec<_>>(), vec!["don", "'t"]); }
    #[test] fn gpt2_contraction_ll() { assert_eq!(Gpt2::new("I'll").collect::<Vec<_>>(), vec!["I", "'ll"]); }
    #[test] fn gpt2_contraction_ve() { assert_eq!(Gpt2::new("I've").collect::<Vec<_>>(), vec!["I", "'ve"]); }
    #[test] fn gpt2_contraction_re() { assert_eq!(Gpt2::new("we're").collect::<Vec<_>>(), vec!["we", "'re"]); }
    #[test] fn gpt2_space_digit() { assert_eq!(Gpt2::new("x 42").collect::<Vec<_>>(), vec!["x", " 42"]); }
    #[test] fn gpt2_double_newline() { assert_eq!(Gpt2::new("a\n\nb").collect::<Vec<_>>(), vec!["a", "\n", "\n", "b"]); }
    #[test] fn gpt2_newline_spaces() { assert_eq!(Gpt2::new("a\n  b").collect::<Vec<_>>(), vec!["a", "\n ", " b"]); }
    #[test] fn gpt2_multi_spaces() { assert_eq!(Gpt2::new("a  b").collect::<Vec<_>>(), vec!["a", " ", " b"]); }
    #[test] fn gpt2_space_punct() { assert_eq!(Gpt2::new("a <b").collect::<Vec<_>>(), vec!["a", " <", "b"]); }
    #[test] fn gpt2_triple_spaces() { assert_eq!(Gpt2::new("a   b").collect::<Vec<_>>(), vec!["a", "  ", " b"]); }

    // CL100K
    #[test] fn cl100k_basic() { assert_eq!(Cl100k::new("Hello world").collect::<Vec<_>>(), vec!["Hello", " world"]); }
    #[test] fn cl100k_digits() { assert_eq!(Cl100k::new("12345").collect::<Vec<_>>(), vec!["123", "45"]); }
    #[test] fn cl100k_case_insensitive() { assert_eq!(Cl100k::new("DON'T").collect::<Vec<_>>(), vec!["DON", "'T"]); }
    #[test] fn cl100k_punct_prefix() { assert_eq!(Cl100k::new("$hello").collect::<Vec<_>>(), vec!["$hello"]); }
    #[test] fn cl100k_space_prefix() { assert_eq!(Cl100k::new("a b").collect::<Vec<_>>(), vec!["a", " b"]); }
    #[test] fn cl100k_newline() { assert_eq!(Cl100k::new("a\nb").collect::<Vec<_>>(), vec!["a", "\n", "b"]); }
    #[test] fn cl100k_space_newline() { assert_eq!(Cl100k::new("a \nb").collect::<Vec<_>>(), vec!["a", " \n", "b"]); }

    // O200K
    #[test] fn o200k_basic() { assert_eq!(O200k::new("Hello world").collect::<Vec<_>>(), vec!["Hello", " world"]); }
    #[test] fn o200k_suffix() { assert_eq!(O200k::new("don't").collect::<Vec<_>>(), vec!["don't"]); }
    #[test] fn o200k_suffix_cases() {
        assert_eq!(O200k::new("DON'T").collect::<Vec<_>>(), vec!["DON'T"]);
        assert_eq!(O200k::new("I'm").collect::<Vec<_>>(), vec!["I'm"]);
        assert_eq!(O200k::new("we'll").collect::<Vec<_>>(), vec!["we'll"]);
    }
    #[test] fn o200k_camelcase() {
        assert_eq!(O200k::new("CamelCase").collect::<Vec<_>>(), vec!["Camel", "Case"]);
        assert_eq!(O200k::new("JSONParser").collect::<Vec<_>>(), vec!["JSONParser"]);
        assert_eq!(O200k::new("parseJSON").collect::<Vec<_>>(), vec!["parse", "JSON"]);
        assert_eq!(O200k::new("XMLHttpRequest").collect::<Vec<_>>(), vec!["XMLHttp", "Request"]);
    }
    #[test] fn o200k_digits() { assert_eq!(O200k::new("12345").collect::<Vec<_>>(), vec!["123", "45"]); }
    #[test] fn o200k_apost_prefix() { assert_eq!(O200k::new("'hello'").collect::<Vec<_>>(), vec!["'hello", "'"]); }

    // Voyage
    #[test] fn voyage_basic() { assert_eq!(Voyage::new("Hello world").collect::<Vec<_>>(), vec!["Hello", " world"]); }
    #[test] fn voyage_single_digits() { assert_eq!(Voyage::new("12345").collect::<Vec<_>>(), vec!["1", "2", "3", "4", "5"]); }
    #[test] fn voyage_contractions() { assert_eq!(Voyage::new("don't").collect::<Vec<_>>(), vec!["don", "'t"]); }
    #[test] fn voyage_punct_prefix() { assert_eq!(Voyage::new("$hello").collect::<Vec<_>>(), vec!["$hello"]); }

    // SmolLM
    #[test] fn smollm_basic() { assert_eq!(SmolLM::new("Hello world").collect::<Vec<_>>(), vec!["Hello", " world"]); }
    #[test] fn smollm_single_digits() { assert_eq!(SmolLM::new("12345").collect::<Vec<_>>(), vec!["1", "2", "3", "4", "5"]); }
    #[test] fn smollm_space_no_digit() {
        assert_eq!(SmolLM::new("test 123").collect::<Vec<_>>(), vec!["test", " ", "1", "2", "3"]);
        assert_eq!(SmolLM::new("a 1 b").collect::<Vec<_>>(), vec!["a", " ", "1", " b"]);
    }
    #[test] fn smollm_contraction() { assert_eq!(SmolLM::new("don't").collect::<Vec<_>>(), vec!["don", "'t"]); }
    #[test] fn smollm_whitespace() {
        assert_eq!(SmolLM::new("a\n\nb").collect::<Vec<_>>(), vec!["a", "\n", "\n", "b"]);
        assert_eq!(SmolLM::new("a  b").collect::<Vec<_>>(), vec!["a", " ", " b"]);
    }
    #[test] fn smollm_newline_digits() { assert_eq!(SmolLM::new("abc\n123").collect::<Vec<_>>(), vec!["abc", "\n", "1", "2", "3"]); }

    // DeepSeek
    #[test] fn deepseek_basic() { assert_eq!(DeepSeek::new("Hello world").collect::<Vec<_>>(), vec!["Hello", " world"]); }
    #[test] fn deepseek_digits() { assert_eq!(DeepSeek::new("12345").collect::<Vec<_>>(), vec!["123", "45"]); }
    #[test] fn deepseek_contractions() {
        assert_eq!(DeepSeek::new("don't").collect::<Vec<_>>(), vec!["don", "'t"]);
        assert_eq!(DeepSeek::new("DON'T").collect::<Vec<_>>(), vec!["DON", "'T"]);
    }
    #[test] fn deepseek_marks() { assert_eq!(DeepSeek::new("ก\u{0E31}น").collect::<Vec<_>>(), vec!["ก\u{0E31}น"]); }
    #[test] fn deepseek_punct_prefix() { assert_eq!(DeepSeek::new("$hello").collect::<Vec<_>>(), vec!["$hello"]); }

    // Qwen
    #[test] fn qwen_basic() { assert_eq!(Qwen::new("Hello world").collect::<Vec<_>>(), vec!["Hello", " world"]); }
    #[test] fn qwen_single_digits() { assert_eq!(Qwen::new("12345").collect::<Vec<_>>(), vec!["1", "2", "3", "4", "5"]); }
    #[test] fn qwen_contractions() {
        assert_eq!(Qwen::new("don't").collect::<Vec<_>>(), vec!["don", "'t"]);
        assert_eq!(Qwen::new("DON'T").collect::<Vec<_>>(), vec!["DON", "'T"]);
    }
    #[test] fn qwen_marks() { assert_eq!(Qwen::new("ก\u{0E31}น").collect::<Vec<_>>(), vec!["ก\u{0E31}น"]); }
    #[test] fn qwen_punct_prefix() { assert_eq!(Qwen::new("$hello").collect::<Vec<_>>(), vec!["$hello"]); }
    #[test] fn qwen_newline() {
        assert_eq!(Qwen::new("a\nb").collect::<Vec<_>>(), vec!["a", "\n", "b"]);
        assert_eq!(Qwen::new("a \nb").collect::<Vec<_>>(), vec!["a", " \n", "b"]);
    }
}
