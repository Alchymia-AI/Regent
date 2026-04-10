"""
Train a SentencePiece BPE tokenizer for the Regent model.

Usage:
    # Train from text files:
    python scripts/train_tokenizer.py --input data/raw/corpus.txt --vocab-size 32768

    # Train from a directory of text files:
    python scripts/train_tokenizer.py --input data/raw/ --vocab-size 32768

    # Output: data/tokenizer/regent.model, data/tokenizer/regent.vocab

The tokenizer reserves IDs 0-5 for special tokens:
    0: [PAD]
    1: [BOS]
    2: [EOS]
    3: [GROUND]  — Ver Head grounding trigger
    4: [EPG]     — EPG prefix boundary
    5: [META]    — REGENT_META marker
"""

import argparse
from pathlib import Path

import sentencepiece as spm


SPECIAL_TOKENS = ["[PAD]", "[BOS]", "[EOS]", "[GROUND]", "[EPG]", "[META]"]


def collect_text_files(input_path: str) -> list[str]:
    """Collect all .txt files from a path (file or directory)."""
    p = Path(input_path)
    if p.is_file():
        return [str(p)]
    elif p.is_dir():
        files = sorted(p.glob("**/*.txt"))
        if not files:
            raise FileNotFoundError(f"No .txt files found in {input_path}")
        return [str(f) for f in files]
    else:
        raise FileNotFoundError(f"Path not found: {input_path}")


def train_tokenizer(
    input_path: str,
    output_dir: str = "data/tokenizer",
    vocab_size: int = 32768,
    model_type: str = "bpe",
    character_coverage: float = 0.9995,
    max_sentence_length: int = 16384,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    files = collect_text_files(input_path)
    print(f"Training tokenizer on {len(files)} file(s)")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Model type: {model_type}")
    print(f"  Output: {out}/regent.model")

    # SentencePiece reserves IDs for special tokens via user_defined_symbols
    # pad_id, bos_id, eos_id are handled natively
    # We add GROUND, EPG, META as user-defined symbols
    user_symbols = SPECIAL_TOKENS[3:]  # [GROUND], [EPG], [META]

    model_prefix = str(out / "regent")

    spm.SentencePieceTrainer.Train(
        input=",".join(files),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        max_sentence_length=max_sentence_length,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        user_defined_symbols=user_symbols,
        # Normalization
        normalization_rule_name="identity",  # no NFKC normalization
        remove_extra_whitespaces=False,
        # Byte fallback for unknown chars
        byte_fallback=True,
        # Training params
        num_threads=4,
        train_extremely_large_corpus=False,
    )

    # Verify
    sp = spm.SentencePieceProcessor()
    sp.Load(f"{model_prefix}.model")
    print(f"\nTokenizer trained successfully")
    print(f"  Vocab size: {sp.GetPieceSize()}")
    print(f"  PAD={sp.pad_id()}, BOS={sp.bos_id()}, EOS={sp.eos_id()}, UNK={sp.unk_id()}")

    # Test encode/decode
    test = "The Regent is a synthetic sentient with persistent memory."
    ids = sp.Encode(test)
    decoded = sp.Decode(ids)
    print(f"\n  Test: \"{test}\"")
    print(f"  Tokens ({len(ids)}): {ids[:20]}{'...' if len(ids) > 20 else ''}")
    print(f"  Decoded: \"{decoded}\"")


def main():
    parser = argparse.ArgumentParser(description="Train Regent tokenizer")
    parser.add_argument("--input", required=True, help="Text file or directory of .txt files")
    parser.add_argument("--output", default="data/tokenizer", help="Output directory")
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--model-type", default="bpe", choices=["bpe", "unigram"])
    args = parser.parse_args()

    train_tokenizer(
        input_path=args.input,
        output_dir=args.output,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
    )


if __name__ == "__main__":
    main()
