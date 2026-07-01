//! Cadmus — a byte-level BPE tokenizer trained from scratch.
//!
//! The one link your `Hephaistos` → `talos` pipeline still outsources to the HF
//! `tokenizers` crate is the *merge trainer*. This is that trainer, by hand,
//! plus a matching encode/decode. See [`bpe::BpeModel`].

pub mod bpe;
pub mod byte_level;

pub use bpe::BpeModel;
