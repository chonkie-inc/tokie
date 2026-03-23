//! Regex-based pretokenizer for cl100k, o200k, and custom patterns.
//!
//! This module provides a regex-automata based pretokenizer that handles
//! negative lookahead patterns by rewriting them as multiple patterns.
//!
//! Used as fallback for patterns without hand-coded lexers.

use regex_automata::{meta::Regex, util::captures::Captures, Anchored, Input};

/// Regex-based pretokenizer.
///
/// Achieves ~146 MiB/s on GPT-2 patterns. Use hand-coded lexers for
/// higher performance when available.
pub struct RegexPretok {
    regex: Regex,
    lookahead: Vec<bool>,
}

impl RegexPretok {
    /// Create a new pretokenizer from patterns.
    ///
    /// Each pattern is a tuple of (pattern_str, is_lookahead).
    /// If is_lookahead is true, the last character of matches will be dropped.
    pub fn new(patterns: &[(&str, bool)]) -> Result<Self, regex_automata::meta::BuildError> {
        let pats: Vec<&str> = patterns.iter().map(|(p, _)| *p).collect();
        let lookahead: Vec<bool> = patterns.iter().map(|(_, l)| *l).collect();
        let regex = Regex::new_many(&pats)?;
        Ok(Self { regex, lookahead })
    }

    /// Create a cl100k (GPT-3.5/GPT-4) compatible pretokenizer.
    ///
    /// Original pattern includes case-insensitive contractions and number chunking.
    pub fn cl100k() -> Self {
        Self::new(&[
            (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+$", false),
            (r"\s+\s", true),
            (r"\s+", false),
        ]).expect("valid cl100k pattern")
    }

    /// Create an o200k (GPT-4o) compatible pretokenizer.
    pub fn o200k() -> Self {
        let pat1 = [
            r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
            r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
            r"\p{N}{1,3}",
            r" ?[^\s\p{L}\p{N}]+[\r\n/]*",
            r"\s*[\r\n]+",
            r"\s+$",
        ]
        .join("|");

        Self::new(&[(&pat1, false), (r"\s+\s", true), (r"\s+", false)])
            .expect("valid o200k pattern")
    }

    /// Create a GPT-2 compatible pretokenizer (regex version).
    ///
    /// Use [`Pretok::GPT2`](super::Pretok::GPT2) for better performance.
    pub fn gpt2() -> Self {
        Self::new(&[
            (
                r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+$",
                false,
            ),
            (r"\s+\s", true),
            (r"\s+", false),
        ])
        .expect("valid GPT-2 pattern")
    }

    /// Split text into pre-tokens.
    pub fn split<'a>(&'a self, text: &'a str) -> RegexPretokIter<'a> {
        RegexPretokIter {
            pretokenizer: self,
            text,
            pos: 0,
            caps: Captures::matches(self.regex.group_info().clone()),
        }
    }

    /// Split text and collect into a Vec.
    pub fn split_to_vec<'a>(&'a self, text: &'a str) -> Vec<&'a str> {
        self.split(text).collect()
    }
}

/// Iterator over pre-tokens from regex pretokenizer.
pub struct RegexPretokIter<'a> {
    pretokenizer: &'a RegexPretok,
    text: &'a str,
    pos: usize,
    caps: Captures,
}

impl<'a> Iterator for RegexPretokIter<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.text.len() {
            return None;
        }

        let input = Input::new(&self.text[self.pos..]).anchored(Anchored::Yes);
        self.caps.clear();
        self.pretokenizer.regex.captures(input, &mut self.caps);

        let m = self.caps.get_match()?;
        let start = self.pos;
        let mut end = self.pos + m.range().end;

        // If this is a lookahead pattern, drop the last character
        if self.pretokenizer.lookahead[m.pattern().as_usize()] {
            if let Some(last_char) = self.text[start..end].chars().next_back() {
                end -= last_char.len_utf8();
            }
        }

        // Safety: ensure we make progress
        if end <= start {
            self.pos = start + 1;
            return self.next();
        }

        self.pos = end;
        Some(&self.text[start..end])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cl100k_numbers() {
        let pre = RegexPretok::cl100k();
        // cl100k chunks numbers into groups of 1-3 digits
        let tokens: Vec<_> = pre.split("12345").collect();
        assert_eq!(tokens, vec!["123", "45"]);
    }

    #[test]
    fn test_gpt2_basic() {
        let pre = RegexPretok::gpt2();
        let tokens: Vec<_> = pre.split("Hello world").collect();
        assert_eq!(tokens, vec!["Hello", " world"]);
    }

    #[test]
    fn test_gpt2_contractions() {
        let pre = RegexPretok::gpt2();
        let tokens: Vec<_> = pre.split("How's it going?").collect();
        assert_eq!(tokens, vec!["How", "'s", " it", " going", "?"]);
    }
}

#[cfg(test)]
mod o200k_tests {
    use super::*;

    #[test]
    fn test_o200k_camelcase() {
        let o200k = RegexPretok::o200k();

        assert_eq!(o200k.split("CamelCase").collect::<Vec<_>>(), vec!["Camel", "Case"]);
        assert_eq!(o200k.split("JSONParser").collect::<Vec<_>>(), vec!["JSONParser"]);
        assert_eq!(o200k.split("parseJSON").collect::<Vec<_>>(), vec!["parse", "JSON"]);
        assert_eq!(o200k.split("XMLHttpRequest").collect::<Vec<_>>(), vec!["XMLHttp", "Request"]);
        assert_eq!(o200k.split("don't").collect::<Vec<_>>(), vec!["don't"]);
        assert_eq!(o200k.split("Hello world").collect::<Vec<_>>(), vec!["Hello", " world"]);
    }
}
