//! Compatibility checking for BPE token pairs
//!
//! Two tokens are "compatible" if no merge rule exists that would have
//! combined bytes at the token boundary into a higher-priority token.

use fnv::FnvHashMap;

use crate::types::{Split, TokenId};

/// Check if two tokens can appear adjacent in a valid BPE encoding.
///
/// This works by "unwinding" the merge history to check if any merge
/// should have happened at the boundary between the tokens.
///
/// # Arguments
/// * `pair_lookup` - Maps (token1, token2) to their merged token
/// * `split_table` - For each token, the two tokens it was merged from
/// * `token1` - The left token
/// * `token2` - The right token
///
/// # Returns
/// `true` if the tokens can be adjacent, `false` if a merge should have happened
pub fn is_valid_token_pair(
    pair_lookup: &FnvHashMap<(TokenId, TokenId), TokenId>,
    split_table: &[Split],
    mut token1: TokenId,
    mut token2: TokenId,
) -> bool {
    let mut limit = u32::MAX;

    loop {
        // Check: would these two boundary pieces merge?
        if let Some(&combined) = pair_lookup.get(&(token1, token2)) {
            if combined < limit {
                // A higher-priority merge exists - NOT compatible!
                return false;
            }
        }

        // Unwind the higher-ID token to check inner boundaries
        if token1 > token2 {
            limit = token1;
            let right = split_table[token1 as usize].right;
            if right == token1 {
                // token1 is a base token, switch to unwinding token2
                limit = token2 + 1;
                let left = split_table[token2 as usize].left;
                if left + 1 == limit {
                    return true; // Both fully unwound, compatible!
                }
                token2 = left;
            } else {
                token1 = right; // Check right part of token1 against token2
            }
        } else {
            limit = token2 + 1;
            let left = split_table[token2 as usize].left;
            if left + 1 == limit {
                // token2 is a base token, switch to unwinding token1
                limit = token1;
                let right = split_table[token1 as usize].right;
                if right == limit {
                    return true; // Both fully unwound, compatible!
                }
                token1 = right;
            } else {
                token2 = left; // Check token1 against left part of token2
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_tokens_compatible() {
        // Base tokens with no merges should always be compatible
        let pair_lookup = FnvHashMap::default();
        let split_table = vec![
            Split::base(0), // 'a'
            Split::base(1), // 'b'
            Split::base(2), // 'c'
        ];

        assert!(is_valid_token_pair(&pair_lookup, &split_table, 0, 1));
        assert!(is_valid_token_pair(&pair_lookup, &split_table, 1, 2));
    }

    #[test]
    fn test_merge_makes_incompatible() {
        // If a+b merges to ab, then (a, b) should be incompatible
        let mut pair_lookup = FnvHashMap::default();
        pair_lookup.insert((0, 1), 3); // a + b -> ab

        let split_table = vec![
            Split::base(0),      // 'a' id=0
            Split::base(1),      // 'b' id=1
            Split::base(2),      // 'c' id=2
            Split::merge(0, 1),  // 'ab' id=3
        ];

        // (a, b) should NOT be compatible because they merge to ab
        assert!(!is_valid_token_pair(&pair_lookup, &split_table, 0, 1));

        // (b, c) should be compatible (no merge rule)
        assert!(is_valid_token_pair(&pair_lookup, &split_table, 1, 2));

        // (ab, c) should be compatible (no merge rule for ab+c)
        assert!(is_valid_token_pair(&pair_lookup, &split_table, 3, 2));
    }
}
