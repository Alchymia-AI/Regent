"""
Prepare pre-training data: tokenize raw text into packed .npy token arrays.

Usage:
    python scripts/prepare_data.py \
        --input data/raw/corpus.txt \
        --tokenizer data/tokenizer/regent.model \
        --output data/processed/train.npy \
        --seq-len 2048

Input: raw text file(s) (one document per line, or continuous text)
Output: packed numpy array of token IDs, ready for training

The packing strategy:
    - Tokenize each document with BOS/EOS markers
    - Concatenate all tokens into one flat array
    - During training, the array is sliced into (seq_len) windows
    - No padding waste — every token contributes to training

For multiple splits (train/val):
    python scripts/prepare_data.py --input data/raw/train.txt --output data/processed/train.npy
    python scripts/prepare_data.py --input data/raw/val.txt --output data/processed/val.npy
"""

import argparse
from pathlib import Path

import numpy as np
import sentencepiece as spm


def tokenize_file(
    input_path: str,
    tokenizer_path: str,
    output_path: str,
    seq_len: int = 2048,
    max_documents: int | None = None,
):
    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_path)

    bos = sp.bos_id()
    eos = sp.eos_id()
    vocab_size = sp.GetPieceSize()

    print(f"Tokenizer: {tokenizer_path} (vocab={vocab_size})")
    print(f"Input: {input_path}")
    print(f"Sequence length: {seq_len}")

    input_p = Path(input_path)
    if input_p.is_dir():
        files = sorted(input_p.glob("**/*.txt"))
    else:
        files = [input_p]

    all_tokens = []
    n_docs = 0
    n_files = 0

    for fp in files:
        n_files += 1
        print(f"  Processing {fp.name}...", end="", flush=True)
        file_tokens = 0

        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Each non-empty line is a document
                ids = sp.Encode(line)
                if not ids:
                    continue

                # Add BOS/EOS
                doc_tokens = [bos] + ids + [eos]
                all_tokens.extend(doc_tokens)
                file_tokens += len(doc_tokens)
                n_docs += 1

                if max_documents and n_docs >= max_documents:
                    break

        print(f" {file_tokens:,} tokens, {n_docs:,} docs total")

        if max_documents and n_docs >= max_documents:
            break

    total = len(all_tokens)
    n_sequences = total // seq_len

    # Trim to exact multiple of seq_len (drop remainder)
    trimmed = n_sequences * seq_len
    tokens_arr = np.array(all_tokens[:trimmed], dtype=np.int32)

    # Save
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), tokens_arr)

    file_size_mb = tokens_arr.nbytes / (1024 * 1024)

    print(f"\nDone:")
    print(f"  Files processed: {n_files}")
    print(f"  Documents: {n_docs:,}")
    print(f"  Total tokens: {total:,}")
    print(f"  Usable tokens: {trimmed:,} ({n_sequences:,} sequences of {seq_len})")
    print(f"  Dropped: {total - trimmed:,} tokens (tail)")
    print(f"  Output: {output_path} ({file_size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Prepare pre-training data")
    parser.add_argument("--input", required=True, help="Text file or directory")
    parser.add_argument("--tokenizer", required=True, help="SentencePiece .model file")
    parser.add_argument("--output", required=True, help="Output .npy file")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--max-documents", type=int, default=None)
    args = parser.parse_args()

    tokenize_file(
        input_path=args.input,
        tokenizer_path=args.tokenizer,
        output_path=args.output,
        seq_len=args.seq_len,
        max_documents=args.max_documents,
    )


if __name__ == "__main__":
    main()
