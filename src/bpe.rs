//! Byte-level BPE: the merge-learning trainer, plus encode/decode.
//!
//! Training is the part that Hephaistos currently outsources to the HF
//! `tokenizers` crate. Here it is by hand:
//!
//!   1. Pre-tokenize the corpus into "words" (a leading space stays with its
//!      word, GPT-2 style), and byte-level encode each word.
//!   2. Count the frequency of every adjacent symbol pair over the corpus.
//!   3. Merge the most frequent pair into one new symbol; record the rule.
//!   4. Repeat until the vocab reaches `vocab_size` (or no pair beats
//!      `min_frequency`).
//!
//! The learned `merges` are an ordered list — their index *is* their priority,
//! which is exactly what encode replays.

use std::collections::HashMap;

use serde_json::{json, Value};

use crate::byte_level;

/// A trained byte-level BPE model: vocabulary + ordered merge rules.
pub struct BpeModel {
    /// id -> token piece, in the byte-level alphabet.
    pub vocab: Vec<String>,
    /// Merge rules in priority order (index = rank, lower = applied first).
    pub merges: Vec<(String, String)>,

    // Derived, rebuilt on load; not serialized.
    token_to_id: HashMap<String, u32>,
    merge_rank: HashMap<(String, String), u32>,
    encoder: [char; 256],
    decoder: HashMap<char, u8>,
}

/// Split text into words, keeping a leading space attached to the word that
/// follows it (so " world" is one unit). Only *splits* — never drops or edits a
/// byte — which is what keeps `decode(encode(s)) == s` intact.
fn pretokenize(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut cur = String::new();
    for c in text.chars() {
        if c == ' ' && !cur.is_empty() {
            words.push(std::mem::take(&mut cur));
        }
        cur.push(c);
    }
    if !cur.is_empty() {
        words.push(cur);
    }
    words
}

/// Replace every adjacent `(a, b)` in `symbols` with the merged `ab`.
fn apply_merge(symbols: &[String], a: &str, b: &str, merged: &str) -> Vec<String> {
    let mut out = Vec::with_capacity(symbols.len());
    let mut i = 0;
    while i < symbols.len() {
        if i + 1 < symbols.len() && symbols[i] == a && symbols[i + 1] == b {
            out.push(merged.to_string());
            i += 2;
        } else {
            out.push(symbols[i].clone());
            i += 1;
        }
    }
    out
}

impl BpeModel {
    /// Train a byte-level BPE on `corpus`, learning merges until the vocabulary
    /// reaches `vocab_size` or no pair occurs at least `min_frequency` times.
    pub fn train(corpus: &str, vocab_size: usize, min_frequency: u64) -> BpeModel {
        let encoder = byte_level::byte_encoder();

        // Vocab starts as the full 256-char byte alphabet (nothing is ever OOV).
        let mut vocab: Vec<String> = byte_level::alphabet(&encoder)
            .into_iter()
            .map(|c| c.to_string())
            .collect();

        // Byte-level encode each word, then fold identical words into counts so
        // pair counting is over the *unique* words, weighted by frequency.
        let mut freq: HashMap<String, u64> = HashMap::new();
        for word in pretokenize(corpus) {
            *freq.entry(byte_level::encode_bytes(&word, &encoder)).or_insert(0) += 1;
        }
        let mut words: Vec<(Vec<String>, u64)> = freq
            .into_iter()
            .map(|(w, c)| (w.chars().map(|ch| ch.to_string()).collect(), c))
            .collect();

        let mut merges: Vec<(String, String)> = Vec::new();

        while vocab.len() < vocab_size {
            // 1. Count every adjacent pair across the corpus.
            let mut pair_counts: HashMap<(&str, &str), u64> = HashMap::new();
            for (symbols, count) in &words {
                for pair in symbols.windows(2) {
                    *pair_counts.entry((&pair[0], &pair[1])).or_insert(0) += count;
                }
            }

            // 2. Pick the most frequent pair; ties break to the lexicographically
            //    smaller pair so training is deterministic.
            let best = pair_counts.iter().fold(
                None::<((&str, &str), u64)>,
                |acc, (&p, &c)| match acc {
                    Some((bp, bc)) if bc > c || (bc == c && bp <= p) => Some((bp, bc)),
                    _ => Some((p, c)),
                },
            );

            let ((a, b), count) = match best {
                Some(x) => x,
                None => break, // corpus has no pairs left to merge
            };
            if count < min_frequency {
                break;
            }
            let (a, b) = (a.to_string(), b.to_string());
            let merged = format!("{a}{b}");

            // 3. Record the rule, extend the vocab, and rewrite the corpus.
            merges.push((a.clone(), b.clone()));
            vocab.push(merged.clone());
            for (symbols, _) in &mut words {
                *symbols = apply_merge(symbols, &a, &b, &merged);
            }
        }

        BpeModel::finalize(vocab, merges, encoder)
    }

    /// Rebuild the derived lookup tables from `vocab` + `merges`.
    fn finalize(vocab: Vec<String>, merges: Vec<(String, String)>, encoder: [char; 256]) -> BpeModel {
        let token_to_id = vocab
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();
        let merge_rank = merges
            .iter()
            .enumerate()
            .map(|(i, (a, b))| ((a.clone(), b.clone()), i as u32))
            .collect();
        let decoder = byte_level::byte_decoder(&encoder);
        BpeModel { vocab, merges, token_to_id, merge_rank, encoder, decoder }
    }

    /// Number of tokens in the vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Greedily apply the learned merges to one byte-level word: repeatedly
    /// merge the adjacent pair with the lowest rank until none is left.
    fn merge_word(&self, mut symbols: Vec<String>) -> Vec<String> {
        loop {
            // Find the mergeable adjacent pair with the best (lowest) rank.
            let mut best: Option<(usize, u32)> = None;
            for (i, pair) in symbols.windows(2).enumerate() {
                if let Some(&rank) = self.merge_rank.get(&(pair[0].clone(), pair[1].clone())) {
                    if best.map_or(true, |(_, r)| rank < r) {
                        best = Some((i, rank));
                    }
                }
            }
            let Some((i, _)) = best else { break };
            let merged = format!("{}{}", symbols[i], symbols[i + 1]);
            symbols.splice(i..i + 2, [merged]);
        }
        symbols
    }

    /// Encode text into token ids.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        for word in pretokenize(text) {
            let symbols: Vec<String> = byte_level::encode_bytes(&word, &self.encoder)
                .chars()
                .map(|c| c.to_string())
                .collect();
            for tok in self.merge_word(symbols) {
                // Every symbol is a byte-alphabet char or a learned merge, both
                // of which are in the vocab, so this lookup never misses.
                ids.push(self.token_to_id[&tok]);
            }
        }
        ids
    }

    /// Decode token ids back into the original string.
    pub fn decode(&self, ids: &[u32]) -> String {
        let joined: String = ids
            .iter()
            .map(|&id| self.vocab[id as usize].as_str())
            .collect();
        byte_level::decode_bytes(&joined, &self.decoder)
    }

    /// Serialize to an HF `tokenizer.json`-shaped string: the exact format
    /// `Hephaistos/src/gguf.rs` parses (`model.vocab` as token->id, `model.merges`
    /// as `"a b"` strings) and that `talos` consumes via GGUF.
    ///
    /// A `<unk>` token is appended if absent: byte-level BPE never emits it (the
    /// full 256-byte alphabet means nothing is OOV), but the GGUF writer requires
    /// one for the special-token ids.
    pub fn to_hf_json(&self) -> String {
        let mut vocab_map = serde_json::Map::new();
        for (i, tok) in self.vocab.iter().enumerate() {
            vocab_map.insert(tok.clone(), json!(i));
        }
        let mut added_tokens = Vec::new();
        if !self.token_to_id.contains_key("<unk>") {
            let id = vocab_map.len();
            vocab_map.insert("<unk>".to_string(), json!(id));
            added_tokens.push(json!({ "id": id, "content": "<unk>", "special": true }));
        }
        let merges: Vec<String> = self.merges.iter().map(|(a, b)| format!("{a} {b}")).collect();

        let doc = json!({
            "model": { "type": "BPE", "vocab": Value::Object(vocab_map), "merges": merges },
            "added_tokens": added_tokens,
        });
        serde_json::to_string_pretty(&doc).expect("model serializes")
    }

    /// Load from an HF `tokenizer.json`-shaped string, honoring the file's token
    /// ids. Reads `model.vocab` + `model.merges`; ignores everything else. Also
    /// loads tokenizers written elsewhere (e.g. Posaidon's) as long as they are
    /// byte-level BPE.
    pub fn from_hf_json(json: &str) -> serde_json::Result<BpeModel> {
        let doc: Value = serde_json::from_str(json)?;
        let model = &doc["model"];

        let vocab_obj = model["vocab"].as_object().expect("model.vocab object");
        let mut vocab = vec![String::new(); vocab_obj.len()];
        for (tok, id) in vocab_obj {
            let id = id.as_u64().expect("vocab id") as usize;
            if id >= vocab.len() {
                vocab.resize(id + 1, String::new());
            }
            vocab[id] = tok.clone();
        }

        let merges = model["merges"]
            .as_array()
            .expect("model.merges array")
            .iter()
            .map(|m| match m.as_array() {
                // ["a", "b"] form
                Some(p) => (p[0].as_str().unwrap().to_string(), p[1].as_str().unwrap().to_string()),
                // "a b" form (token pieces never contain a literal space)
                None => {
                    let (a, b) = m.as_str().unwrap().split_once(' ').expect("merge \"a b\"");
                    (a.to_string(), b.to_string())
                }
            })
            .collect();

        Ok(BpeModel::finalize(vocab, merges, byte_level::byte_encoder()))
    }
}
