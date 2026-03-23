//! Benchmark normalizer throughput
//!
//! Tests different normalization strategies to understand performance.

use std::borrow::Cow;
use std::time::Instant;

// ============================================================================
// Fast ASCII lowercase
// ============================================================================

/// Check if string is pure ASCII
#[inline]
fn is_ascii(bytes: &[u8]) -> bool {
    // Process 8 bytes at a time
    let mut i = 0;
    while i + 8 <= bytes.len() {
        let chunk = u64::from_ne_bytes(bytes[i..i+8].try_into().unwrap());
        if chunk & 0x8080808080808080 != 0 {
            return false;
        }
        i += 8;
    }
    // Check remaining bytes
    while i < bytes.len() {
        if bytes[i] & 0x80 != 0 {
            return false;
        }
        i += 1;
    }
    true
}

/// Check if ASCII string has any uppercase
#[inline]
fn has_ascii_uppercase(bytes: &[u8]) -> bool {
    for &b in bytes {
        if b >= b'A' && b <= b'Z' {
            return true;
        }
    }
    false
}

/// Fast ASCII lowercase (branchless, auto-vectorizes)
#[inline]
fn ascii_lowercase_inplace(bytes: &mut [u8]) {
    for b in bytes.iter_mut() {
        // Branchless: if A-Z, set bit 5
        let is_upper = (*b >= b'A') & (*b <= b'Z');
        *b |= (is_upper as u8) << 5;
    }
}

/// ASCII-optimized lowercase with Unicode fallback
fn fast_lowercase<'a>(text: &'a str) -> Cow<'a, str> {
    let bytes = text.as_bytes();

    // Fast path: pure ASCII
    if is_ascii(bytes) {
        if !has_ascii_uppercase(bytes) {
            return Cow::Borrowed(text);
        }
        // ASCII with uppercase - do fast lowercase
        let mut result = text.to_owned().into_bytes();
        ascii_lowercase_inplace(&mut result);
        // SAFETY: ASCII lowercase preserves valid UTF-8
        return Cow::Owned(unsafe { String::from_utf8_unchecked(result) });
    }

    // Slow path: Unicode
    Cow::Owned(text.to_lowercase())
}

/// Simple: check if any uppercase, then call std
fn simple_lowercase<'a>(text: &'a str) -> Cow<'a, str> {
    // Check if any uppercase letter exists (ASCII or Unicode)
    let has_upper = text.chars().any(|c| c.is_uppercase());
    if !has_upper {
        return Cow::Borrowed(text);
    }
    Cow::Owned(text.to_lowercase())
}

/// Even simpler: just use make_ascii_lowercase for ASCII, fallback for Unicode
fn make_ascii_lower<'a>(text: &'a str) -> Cow<'a, str> {
    let bytes = text.as_bytes();

    // Check if pure ASCII and has uppercase
    if is_ascii(bytes) {
        if !has_ascii_uppercase(bytes) {
            return Cow::Borrowed(text);
        }
        let mut s = text.to_owned();
        // SAFETY: make_ascii_lowercase only modifies ASCII bytes
        unsafe { s.as_bytes_mut().make_ascii_lowercase() };
        return Cow::Owned(s);
    }

    // Has non-ASCII - check if any uppercase exists
    if !text.chars().any(|c| c.is_uppercase()) {
        return Cow::Borrowed(text);
    }
    Cow::Owned(text.to_lowercase())
}

/// Hybrid: process ASCII prefix fast, then Unicode
fn hybrid_lowercase<'a>(text: &'a str) -> Cow<'a, str> {
    let bytes = text.as_bytes();

    // Find first non-ASCII byte
    let ascii_len = bytes.iter().position(|&b| b & 0x80 != 0).unwrap_or(bytes.len());

    if ascii_len == bytes.len() {
        // Pure ASCII
        if !has_ascii_uppercase(bytes) {
            return Cow::Borrowed(text);
        }
        let mut result = text.to_owned().into_bytes();
        ascii_lowercase_inplace(&mut result);
        return Cow::Owned(unsafe { String::from_utf8_unchecked(result) });
    }

    // Mixed: ASCII prefix + Unicode suffix
    let mut result = String::with_capacity(text.len());

    // Fast ASCII prefix
    if ascii_len > 0 {
        let ascii_part = &text[..ascii_len];
        let mut ascii_bytes = ascii_part.to_owned().into_bytes();
        ascii_lowercase_inplace(&mut ascii_bytes);
        result.push_str(unsafe { std::str::from_utf8_unchecked(&ascii_bytes) });
    }

    // Unicode suffix
    result.push_str(&text[ascii_len..].to_lowercase());

    Cow::Owned(result)
}

/// BertNormalizer operations
struct BertNormalizer {
    lowercase: bool,
    clean_text: bool,
    handle_chinese_chars: bool,
    strip_accents: bool,
}

impl BertNormalizer {
    fn normalize<'a>(&self, text: &'a str) -> Cow<'a, str> {
        // Fast path: check if any transformation needed
        let needs_transform = self.lowercase && text.chars().any(|c| c.is_uppercase())
            || self.clean_text && text.bytes().any(|b| b < 0x20 && b != b'\t' && b != b'\n' && b != b'\r')
            || self.handle_chinese_chars && text.chars().any(is_chinese_char)
            || self.strip_accents && text.chars().any(|c| c.is_ascii() == false);

        if !needs_transform {
            return Cow::Borrowed(text);
        }

        let mut result = String::with_capacity(text.len() + text.len() / 10);

        for c in text.chars() {
            // Clean text: remove control chars, normalize whitespace
            if self.clean_text {
                if c == '\t' || c == '\n' || c == '\r' {
                    result.push(' ');
                    continue;
                }
                if c.is_control() {
                    continue;
                }
            }

            // Handle Chinese chars: add spaces around them
            if self.handle_chinese_chars && is_chinese_char(c) {
                result.push(' ');
                if self.lowercase {
                    for lc in c.to_lowercase() {
                        result.push(lc);
                    }
                } else {
                    result.push(c);
                }
                result.push(' ');
                continue;
            }

            // Lowercase
            if self.lowercase {
                for lc in c.to_lowercase() {
                    result.push(lc);
                }
            } else {
                result.push(c);
            }
        }

        Cow::Owned(result)
    }
}

/// Check if character is Chinese/CJK
#[inline]
fn is_chinese_char(c: char) -> bool {
    let cp = c as u32;
    matches!(cp,
        0x4E00..=0x9FFF |      // CJK Unified Ideographs
        0x3400..=0x4DBF |      // CJK Unified Ideographs Extension A
        0x20000..=0x2A6DF |    // CJK Unified Ideographs Extension B
        0x2A700..=0x2B73F |    // CJK Unified Ideographs Extension C
        0x2B740..=0x2B81F |    // CJK Unified Ideographs Extension D
        0x2B820..=0x2CEAF |    // CJK Unified Ideographs Extension E
        0xF900..=0xFAFF |      // CJK Compatibility Ideographs
        0x2F800..=0x2FA1F |    // CJK Compatibility Ideographs Supplement
        0x3000..=0x303F |      // CJK Symbols and Punctuation
        0xFF00..=0xFFEF        // Halfwidth and Fullwidth Forms
    )
}

/// Simple lowercase-only normalizer
fn lowercase_only(text: &str) -> Cow<str> {
    if text.chars().any(|c| c.is_uppercase()) {
        Cow::Owned(text.to_lowercase())
    } else {
        Cow::Borrowed(text)
    }
}

/// NFC normalization using unicode-normalization crate pattern
fn nfc_normalize(text: &str) -> Cow<str> {
    use unicode_normalization::{is_nfc, UnicodeNormalization};
    if is_nfc(text) {
        Cow::Borrowed(text)
    } else {
        Cow::Owned(text.nfc().collect())
    }
}

fn bench<F>(name: &str, text: &str, iterations: u32, f: F)
where F: Fn(&str) -> Cow<str>
{
    // Warmup
    for _ in 0..10 {
        let _ = f(text);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = f(text);
    }
    let elapsed = start.elapsed();

    let total_bytes = text.len() as u64 * iterations as u64;
    let throughput = total_bytes as f64 / elapsed.as_secs_f64() / 1_000_000.0;

    // Check if it borrowed or allocated
    let result = f(text);
    let allocated = matches!(result, Cow::Owned(_));

    println!("{:<25} {:>8.1} MB/s  (allocated: {})", name, throughput, allocated);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string("benches/data/war_and_peace.txt")?;
    let iterations = 100;

    println!("Text size: {} bytes ({:.1} MB)", text.len(), text.len() as f64 / 1_000_000.0);
    println!("Iterations: {}\n", iterations);

    // Check if text is ASCII
    let ascii_pct = text.bytes().filter(|&b| b < 128).count() as f64 / text.len() as f64 * 100.0;
    let upper_pct = text.bytes().filter(|&b| b >= b'A' && b <= b'Z').count() as f64 / text.len() as f64 * 100.0;
    println!("ASCII: {:.1}%, Uppercase: {:.1}%\n", ascii_pct, upper_pct);

    // Test with English text (mostly ASCII, some uppercase)
    println!("=== English Text (War and Peace) ===\n");

    bench("None (passthrough)", &text, iterations, |t| Cow::Borrowed(t));
    bench("std to_lowercase()", &text, iterations, lowercase_only);
    bench("simple_lowercase", &text, iterations, simple_lowercase);
    bench("make_ascii_lower", &text, iterations, make_ascii_lower);
    bench("fast_lowercase", &text, iterations, fast_lowercase);
    bench("NFC", &text, iterations, nfc_normalize);

    // Test with lowercase text (should be fast path - no allocation!)
    let lower_text = text.to_lowercase();
    println!("\n=== Pre-lowercased Text (should NOT allocate) ===\n");

    bench("None (passthrough)", &lower_text, iterations, |t| Cow::Borrowed(t));
    bench("std to_lowercase()", &lower_text, iterations, lowercase_only);
    bench("simple_lowercase", &lower_text, iterations, simple_lowercase);
    bench("make_ascii_lower", &lower_text, iterations, make_ascii_lower);

    // Test with pure ASCII (generated)
    let ascii_text: String = (0..text.len()).map(|i| {
        let c = (i % 52) as u8;
        if c < 26 { (b'A' + c) as char } else { (b'a' + c - 26) as char }
    }).collect();
    println!("\n=== Pure ASCII (50% uppercase, MUST allocate) ===\n");

    bench("None (passthrough)", &ascii_text, iterations, |t| Cow::Borrowed(t));
    bench("std to_lowercase()", &ascii_text, iterations, lowercase_only);
    bench("simple_lowercase", &ascii_text, iterations, simple_lowercase);
    bench("make_ascii_lower", &ascii_text, iterations, make_ascii_lower);

    // Test with pure ASCII lowercase (should NOT allocate)
    let ascii_lower: String = (0..text.len()).map(|i| (b'a' + (i % 26) as u8) as char).collect();
    println!("\n=== Pure ASCII lowercase (should NOT allocate) ===\n");

    bench("None (passthrough)", &ascii_lower, iterations, |t| Cow::Borrowed(t));
    bench("std to_lowercase()", &ascii_lower, iterations, lowercase_only);
    bench("simple_lowercase", &ascii_lower, iterations, simple_lowercase);
    bench("make_ascii_lower", &ascii_lower, iterations, make_ascii_lower);

    Ok(())
}
