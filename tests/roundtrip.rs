//! Every layer is checked: byte-level roundtrip, known merges on a toy corpus,
//! determinism, and a full train -> encode -> decode roundtrip on real UTF-8.

use cadmus::BpeModel;

const CORPUS: &str = "the cat sat on the mat. the cat ran. \
    a cat and a hat. the rat sat on the hat.";

#[test]
fn roundtrip_holds_for_arbitrary_utf8() {
    let model = BpeModel::train(CORPUS, 320, 2);
    // Includes umlauts and an emoji — bytes the model never saw in training.
    for s in ["the cat sat", "grüße aus münchen", "🌱 wächst", "", " leading space"] {
        assert_eq!(model.decode(&model.encode(s)), s, "roundtrip failed for {s:?}");
    }
}

#[test]
fn learns_the_obvious_merge_first() {
    // "ab" is by far the most frequent pair, so it must be the first merge.
    let model = BpeModel::train("ababab ab ab", 300, 2);
    assert_eq!(model.merges.first(), Some(&("a".to_string(), "b".to_string())));
    // ...and "ab" then becomes a single token.
    assert_eq!(model.encode("ab").len(), 1);
}

#[test]
fn training_is_deterministic() {
    let a = BpeModel::train(CORPUS, 320, 2);
    let b = BpeModel::train(CORPUS, 320, 2);
    assert_eq!(a.merges, b.merges);
    assert_eq!(a.vocab, b.vocab);
}

#[test]
fn merging_shrinks_the_token_count() {
    let raw = BpeModel::train(CORPUS, 256, 2); // no merges: pure byte-level
    let merged = BpeModel::train(CORPUS, 320, 2); // 64 merges learned
    let text = "the cat sat on the mat";
    assert!(
        merged.encode(text).len() < raw.encode(text).len(),
        "learned merges should compress a familiar sentence"
    );
}

#[test]
fn json_save_load_preserves_behavior() {
    let model = BpeModel::train(CORPUS, 320, 2);
    let reloaded = BpeModel::from_json(&model.to_json()).unwrap();
    let text = "the cat sat on the hat";
    assert_eq!(model.encode(text), reloaded.encode(text));
}
