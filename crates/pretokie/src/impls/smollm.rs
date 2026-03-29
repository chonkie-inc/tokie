//! GPT-2 with individual digit isolation — single-pass, zero allocation.
//!
//! Used by: SmolLM2, models with Sequence[Digits, ByteLevel]
//!
//! Same as GPT-2 but:
//! - Each digit is a separate piece: `12345` → `["1", "2", "3", "4", "5"]`
//! - Space does NOT prefix digits: `test 123` → `["test", " ", "1", "2", "3"]`
//! - Contractions are case-sensitive and standalone (same as GPT-2)

use crate::util::{decode_utf8, is_ascii_letter, is_digit};

pub struct SmolLM<'a> {
    bytes: &'a [u8],
    pos: usize,
    len: usize,
}

impl<'a> SmolLM<'a> {
    pub fn new(text: &'a str) -> Self {
        let bytes = text.as_bytes();
        Self { bytes, pos: 0, len: bytes.len() }
    }

    #[inline(always)]
    fn at(&self, pos: usize) -> u8 {
        unsafe { *self.bytes.get_unchecked(pos) }
    }

    #[inline(always)]
    fn scan_letters(&mut self) {
        while self.pos < self.len {
            let b = self.at(self.pos);
            if is_ascii_letter(b) {
                self.pos += 1;
            } else if b >= 0x80 {
                let (ch, cl) = decode_utf8(&self.bytes[self.pos..]);
                if ch.is_alphabetic() { self.pos += cl; } else { return; }
            } else {
                return;
            }
        }
    }

    #[inline(always)]
    fn check_contraction(&self) -> usize {
        if self.pos >= self.len || self.bytes[self.pos] != b'\'' { return 0; }
        let rem = self.len - self.pos;
        if rem < 2 { return 0; }
        let b1 = self.bytes[self.pos + 1];
        if matches!(b1, b's' | b't' | b'd' | b'm') {
            if rem == 2 || !is_ascii_letter(self.bytes[self.pos + 2]) {
                return 2;
            }
        }
        if rem < 3 { return 0; }
        let b2 = self.bytes[self.pos + 2];
        if (b1 == b'l' && b2 == b'l')
            || (b1 == b'v' && b2 == b'e')
            || (b1 == b'r' && b2 == b'e')
        {
            return 3;
        }
        0
    }

    #[inline(always)]
    fn scan_punct(&mut self) {
        while self.pos < self.len {
            let b = self.at(self.pos);
            if is_ascii_letter(b) || is_digit(b) || b == b' ' || b == b'\n' || b == b'\r' || b >= 0x80 {
                break;
            }
            self.pos += 1;
        }
    }

    #[inline(always)]
    fn emit(&self, start: usize) -> &'a str {
        unsafe { std::str::from_utf8_unchecked(&self.bytes[start..self.pos]) }
    }
}

impl<'a> Iterator for SmolLM<'a> {
    type Item = &'a str;

    #[inline(always)]
    fn next(&mut self) -> Option<&'a str> {
        if self.pos >= self.len { return None; }

        let start = self.pos;
        let b = self.at(self.pos);

        if is_ascii_letter(b) {
            self.pos += 1;
            self.scan_letters();
            if self.check_contraction() > 0 {
                return Some(self.emit(start));
            }
        } else if b == b' ' {
            self.pos += 1;
            if self.pos < self.len {
                let next = self.at(self.pos);
                if is_ascii_letter(next) {
                    self.pos += 1;
                    self.scan_letters();
                    if self.check_contraction() > 0 {
                        return Some(self.emit(start));
                    }
                } else if next >= 0x80 {
                    let (ch, _) = decode_utf8(&self.bytes[self.pos..]);
                    if ch.is_alphabetic() {
                        self.scan_letters();
                    }
                } else if next != b' ' && next != b'\n' && next != b'\r' && !is_digit(next) {
                    // Space prefixes punctuation (but NOT digits in SmolLM)
                    self.pos += 1;
                    self.scan_punct();
                }
                // else: bare space
            }
        } else if b == b'\'' {
            let clen = self.check_contraction();
            if clen > 0 {
                self.pos += clen;
            } else {
                self.pos += 1;
                self.scan_punct();
            }
        } else if is_digit(b) {
            // Single digit only
            self.pos += 1;
        } else if b == b'\n' || b == b'\r' {
            self.pos += 1;
            while self.pos < self.len {
                let c = self.at(self.pos);
                if c == b'\n' || c == b'\r' || c == b' ' { self.pos += 1; }
                else { break; }
            }
        } else if b >= 0x80 {
            let (ch, cl) = decode_utf8(&self.bytes[self.pos..]);
            self.pos += cl;
            if ch.is_alphabetic() {
                self.scan_letters();
            } else if ch.is_whitespace() {
                while self.pos < self.len {
                    let c = self.at(self.pos);
                    if c == b' ' || c == b'\n' || c == b'\r' { self.pos += 1; }
                    else if c >= 0x80 {
                        let (ch2, cl2) = decode_utf8(&self.bytes[self.pos..]);
                        if ch2.is_whitespace() { self.pos += cl2; } else { break; }
                    } else { break; }
                }
            }
        } else {
            self.pos += 1;
            self.scan_punct();
        }

        debug_assert!(self.pos > start, "no progress at pos {start}");
        Some(self.emit(start))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split(text: &str) -> Vec<&str> {
        SmolLM::new(text).collect()
    }

    #[test]
    fn basic() {
        assert_eq!(split("Hello world"), vec!["Hello", " world"]);
    }

    #[test]
    fn single_digits() {
        assert_eq!(split("12345"), vec!["1", "2", "3", "4", "5"]);
    }

    #[test]
    fn space_no_digit_prefix() {
        assert_eq!(split("test 123"), vec!["test", " ", "1", "2", "3"]);
        assert_eq!(split("a 1 b"), vec!["a", " ", "1", " b"]);
    }

    #[test]
    fn contractions() {
        assert_eq!(split("don't"), vec!["don", "'t"]);
        assert_eq!(split("it's"), vec!["it", "'s"]);
    }

    #[test]
    fn whitespace() {
        assert_eq!(split("a\n\nb"), vec!["a", "\n\n", "b"]);
        assert_eq!(split("a  b"), vec!["a", " ", " b"]);
    }

    #[test]
    fn newline_then_digits() {
        assert_eq!(split("abc\n123"), vec!["abc", "\n", "1", "2", "3"]);
    }
}
