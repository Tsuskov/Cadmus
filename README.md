# Cadmus

A byte-level BPE tokenizer trained **from scratch** in Rust — the merge-learning
algorithm by hand, no [`tokenizers`](https://github.com/huggingface/tokenizers)
crate.

Named for [Cadmus](https://en.wikipedia.org/wiki/Cadmus), who legend says brought
the alphabet to Greece: turning a stream of text into a set of symbols.

## Why

My `Hephaistos` → `talos` pipeline builds every layer of an LLM by hand — except
one. Hephaistos trains its tokenizer merges with HuggingFace's `tokenizers`
crate; talos hand-writes only the *encode/decode*. The actual BPE **trainer** —
learning the merges from a corpus — was the last outsourced link. Cadmus is that
trainer.

## What BPE training is

Byte Pair Encoding starts from raw bytes and repeatedly fuses the most common
adjacent pair into a new symbol:

1. **Pre-tokenize** the corpus into words (a leading space stays with its word,
   GPT-2 style) and byte-level encode each — every byte `0..256` maps to a
   printable unicode char, so the merger never sees control bytes or spaces and
   `decode(encode(s)) == s` holds for *any* UTF-8.
2. **Count** every adjacent symbol pair over the whole corpus.
3. **Merge** the most frequent pair into one new token; record the rule.
4. **Repeat** until the vocabulary reaches `vocab_size` (or no pair beats
   `min_frequency`).

The learned `merges` are an ordered list — index = priority — which is exactly
what encoding replays: greedily merge the lowest-rank pair until none remains.

Everything lives in [`src/bpe.rs`](src/bpe.rs); the reversible byte alphabet is
in [`src/byte_level.rs`](src/byte_level.rs).

## Usage

```sh
# Train a 320-token model on a corpus. The model is saved as an HF
# `tokenizer.json` (the shape Hephaistos/gguf and llama.cpp read).
cargo run -- train data/sample.txt 320 tokenizer.json

# Encode / decode with it.
cargo run -- encode tokenizer.json "the cat sat on the mat"   # -> 83 256 268 ...
cargo run -- decode tokenizer.json 83 256 268 291 287 261 286 # -> the cat sat on the mat
```

As a library:

```rust
use cadmus::BpeModel;

let model = BpeModel::train(&corpus, 8_000, 2);
let ids = model.encode("hello world");
assert_eq!(model.decode(&ids), "hello world");
std::fs::write("tokenizer.json", model.to_hf_json())?; // HF-shaped, gguf-ready
```

## Verifying it works

```sh
cargo test
```

- **byte-level roundtrip** — `decode(encode(s)) == s` for umlauts and emoji the
  model never saw in training
- **known first merge** — the obvious most-frequent pair is learned first
- **determinism** — training twice yields identical vocab + merges
- **compression** — learned merges shrink a familiar sentence vs. pure bytes
- **save/load** — a reloaded model encodes identically

## Wired into Hephaistos

Cadmus **replaces** the HuggingFace `tokenizers` crate in
[Hephaistos](https://github.com/Tsuskov/Hephaistos): its `data.rs` now trains and
encodes via `cadmus::BpeModel`, and `to_hf_json` emits the exact
`tokenizer.json` shape its `gguf.rs` parses (with a `<unk>` token injected, which
byte-level BPE never actually emits but the GGUF writer requires). Verified both
ways — Cadmus loads the old HF-written tokenizers and decodes their existing
token bins, and `gguf.rs` parses a Cadmus-written tokenizer.

Scope note: this is a *consistent* drop-in, not a byte-identical clone of the HF
path. Cadmus uses a simplified pre-tokenizer (leading-space-attaches-to-word) and
lexicographic tie-breaking, so it learns its own merges rather than reproducing
HF's ids exactly — which is the point for a from-scratch stack.
