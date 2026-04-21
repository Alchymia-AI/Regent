"""
Generate synthetic text corpus for pipeline smoke testing.

Outputs data/raw/train.txt and data/raw/val.txt with random sentences
that exercise the tokenizer and training loop without real data.
"""

import argparse
import random
from pathlib import Path

SUBJECTS = ["The system", "A node", "The model", "An agent", "The encoder", "A token",
            "The graph", "A layer", "The head", "An embedding", "The kernel", "A parameter"]

VERBS = ["processes", "compresses", "encodes", "transforms", "evaluates", "normalizes",
         "projects", "attends to", "retrieves", "stores", "updates", "generates"]

OBJECTS = ["the input sequence", "a hidden state", "the attention weights", "structured knowledge",
           "the recurrent buffer", "a confidence score", "the next token", "a prefix embedding",
           "the gradient signal", "behavioral parameters", "the output logits", "a memory trace"]

MODIFIERS = ["in constant memory", "without growing", "at inference time", "during training",
             "across all layers", "with fixed cost", "per token", "in a single pass",
             "using selective scan", "with gated projection", "through the backbone",
             "at the verification boundary"]


def generate_sentence() -> str:
    s = random.choice(SUBJECTS)
    v = random.choice(VERBS)
    o = random.choice(OBJECTS)
    m = random.choice(MODIFIERS)
    return f"{s} {v} {o} {m}."


def generate_document(min_sentences: int = 3, max_sentences: int = 12) -> str:
    n = random.randint(min_sentences, max_sentences)
    return " ".join(generate_sentence() for _ in range(n))


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic test corpus")
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--train-docs", type=int, default=10000)
    parser.add_argument("--val-docs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_path = out / "train.txt"
    val_path = out / "val.txt"

    with open(train_path, "w") as f:
        for _ in range(args.train_docs):
            f.write(generate_document() + "\n")

    with open(val_path, "w") as f:
        for _ in range(args.val_docs):
            f.write(generate_document() + "\n")

    print(f"Generated {args.train_docs} train docs → {train_path}")
    print(f"Generated {args.val_docs} val docs → {val_path}")


if __name__ == "__main__":
    main()
