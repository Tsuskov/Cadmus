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
# Train a 320-token model on a corpus.
cargo run -- train data/sample.txt 320 model.json

# Encode / decode with it.
cargo run -- encode model.json "the cat sat on the mat"   # -> 83 256 268 ...
cargo run -- decode model.json 83 256 268 291 287 261 286 # -> the cat sat on the mat
```

As a library:

```rust
use cadmus::BpeModel;

let model = BpeModel::train(&corpus, 8_000, 2);
let ids = model.encode("hello world");
assert_eq!(model.decode(&ids), "hello world");
std::fs::write("model.json", model.to_json())?;
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

## Not (yet) done

The output isn't wired into Hephaistos as a drop-in for the `tokenizers` crate —
that would mean emitting an HF-`tokenizer.json`-shaped file its `gguf.rs` already
parses. Left as the natural next step.
