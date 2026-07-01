//! Cadmus CLI: train a BPE model, or encode/decode with a trained one.
//!
//!   cadmus train  <corpus.txt> <vocab_size> <out.json>
//!   cadmus encode <model.json> <text...>
//!   cadmus decode <model.json> <id id id...>

use std::fs;
use std::process::exit;

use cadmus::BpeModel;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("train") => train(&args[1..]),
        Some("encode") => encode(&args[1..]),
        Some("decode") => decode(&args[1..]),
        _ => usage(),
    }
}

fn usage() -> ! {
    eprintln!(
        "usage:\n  \
         cadmus train  <corpus.txt> <vocab_size> <out.json>\n  \
         cadmus encode <model.json> <text...>\n  \
         cadmus decode <model.json> <id id id...>"
    );
    exit(2);
}

fn train(args: &[String]) {
    let [corpus, vocab_size, out] = args else { usage() };
    let text = fs::read_to_string(corpus).unwrap_or_else(|e| {
        eprintln!("cannot read {corpus}: {e}");
        exit(1);
    });
    let vocab_size: usize = vocab_size.parse().unwrap_or_else(|_| usage());

    let model = BpeModel::train(&text, vocab_size, 2);
    fs::write(out, model.to_json()).expect("write model");
    eprintln!(
        "trained {} tokens ({} merges) -> {out}",
        model.vocab_size(),
        model.vocab_size() - 256
    );
}

fn encode(args: &[String]) {
    let [model_path, text @ ..] = args else { usage() };
    let model = load(model_path);
    let ids = model.encode(&text.join(" "));
    let joined: Vec<String> = ids.iter().map(u32::to_string).collect();
    println!("{}", joined.join(" "));
}

fn decode(args: &[String]) {
    let [model_path, ids @ ..] = args else { usage() };
    let model = load(model_path);
    let ids: Vec<u32> = ids.iter().map(|s| s.parse().unwrap_or_else(|_| usage())).collect();
    println!("{}", model.decode(&ids));
}

fn load(path: &str) -> BpeModel {
    let json = fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("cannot read {path}: {e}");
        exit(1);
    });
    BpeModel::from_json(&json).unwrap_or_else(|e| {
        eprintln!("bad model {path}: {e}");
        exit(1);
    })
}
