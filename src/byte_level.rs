//! GPT-2's reversible bytes<->unicode alphabet.
//!
//! BPE should never see raw control bytes or spaces, so every byte `0..256` is
//! mapped to a *printable* unicode codepoint before merging, and mapped back on
//! decode. This makes `decode(encode(s)) == s` hold for arbitrary UTF-8, no
//! matter what merges were learned. Mirrors `talos/src/tokenizer.rs`.

use std::collections::HashMap;

/// byte -> printable alphabet char.
pub fn byte_encoder() -> [char; 256] {
    // Bytes that are already printable map to themselves.
    let mut bs: Vec<u32> = Vec::new();
    bs.extend(b'!' as u32..=b'~' as u32);
    bs.extend(0xA1u32..=0xAC);
    bs.extend(0xAEu32..=0xFF);

    let mut cs: Vec<u32> = bs.clone();
    // The remaining (non-printable) bytes get codepoints 256, 257, ... in order.
    let mut n = 0u32;
    for b in 0u32..256 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }

    let mut table = ['\0'; 256];
    for (b, c) in bs.iter().zip(cs.iter()) {
        table[*b as usize] = char::from_u32(*c).unwrap();
    }
    table
}

/// alphabet char -> byte (inverse of [`byte_encoder`]).
pub fn byte_decoder(enc: &[char; 256]) -> HashMap<char, u8> {
    enc.iter()
        .enumerate()
        .map(|(b, &c)| (c, b as u8))
        .collect()
}

/// Encode a string's UTF-8 bytes into the byte-level alphabet.
pub fn encode_bytes(text: &str, enc: &[char; 256]) -> String {
    text.bytes().map(|b| enc[b as usize]).collect()
}

/// Decode byte-level alphabet chars back into the original UTF-8 string.
pub fn decode_bytes(piece: &str, dec: &HashMap<char, u8>) -> String {
    let bytes: Vec<u8> = piece.chars().filter_map(|c| dec.get(&c).copied()).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// The full 256-char alphabet, sorted by codepoint for a deterministic vocab.
pub fn alphabet(enc: &[char; 256]) -> Vec<char> {
    let mut a: Vec<char> = enc.to_vec();
    a.sort_unstable();
    a
}
