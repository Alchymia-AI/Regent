"""
Data pipeline for training a code-focused Regent model.

Downloads, processes, and formats data for all 4 training phases
from publicly available HuggingFace datasets.

Usage:
    # Full pipeline (downloads ~50GB for Phase 1)
    PYTHONPATH=. python3 scripts/prepare_code_data.py --all

    # Individual phases
    PYTHONPATH=. python3 scripts/prepare_code_data.py --phase1 --max-docs 500000
    PYTHONPATH=. python3 scripts/prepare_code_data.py --phase2
    PYTHONPATH=. python3 scripts/prepare_code_data.py --phase3
    PYTHONPATH=. python3 scripts/prepare_code_data.py --phase4

    # Quick test (small subset)
    PYTHONPATH=. python3 scripts/prepare_code_data.py --all --max-docs 10000

Requirements:
    pip install datasets
"""

import argparse
import json
import random
from pathlib import Path


def prepare_phase1(output_dir: str, max_docs: int, val_ratio: float = 0.02):
    """
    Phase 1: Base pre-training on Python code.
    Source: codeparrot/codeparrot-clean
    """
    from datasets import load_dataset

    out = Path(output_dir) / "raw"
    out.mkdir(parents=True, exist_ok=True)

    print(f"Phase 1: Downloading codeparrot-clean (max {max_docs:,} docs)...")
    ds = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)

    docs = []
    for i, row in enumerate(ds):
        if i >= max_docs:
            break
        content = row.get("content", "").strip()
        if len(content) < 50:
            continue
        # One document per line, newlines within doc replaced with \n literal
        docs.append(content.replace("\n", "\\n"))
        if (i + 1) % 50000 == 0:
            print(f"  {i + 1:,} docs loaded...")

    random.shuffle(docs)
    split = int(len(docs) * (1 - val_ratio))
    train_docs = docs[:split]
    val_docs = docs[split:]

    train_path = out / "train.txt"
    val_path = out / "val.txt"

    with open(train_path, "w") as f:
        for doc in train_docs:
            f.write(doc + "\n")

    with open(val_path, "w") as f:
        for doc in val_docs:
            f.write(doc + "\n")

    print(f"  Train: {len(train_docs):,} docs -> {train_path}")
    print(f"  Val: {len(val_docs):,} docs -> {val_path}")


def prepare_phase2(output_dir: str, max_docs: int = 75000):
    """
    Phase 2: Code instruction fine-tuning.
    Source: ise-uiuc/Magicoder-OSS-Instruct-75K
    """
    from datasets import load_dataset

    out = Path(output_dir) / "phase2"
    out.mkdir(parents=True, exist_ok=True)

    print(f"Phase 2: Downloading Magicoder-OSS-Instruct-75K...")
    ds = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")

    train_samples = []
    val_samples = []

    for i, row in enumerate(ds):
        if i >= max_docs:
            break

        problem = row.get("problem", "").strip()
        solution = row.get("solution", "").strip()

        if not problem or not solution:
            continue

        sample = {
            "messages": [
                {"role": "user", "content": problem},
                {"role": "assistant", "content": solution},
            ]
        }

        if random.random() < 0.02:
            val_samples.append(sample)
        else:
            train_samples.append(sample)

    train_path = out / "train.jsonl"
    val_path = out / "val.jsonl"

    with open(train_path, "w") as f:
        for s in train_samples:
            f.write(json.dumps(s) + "\n")

    with open(val_path, "w") as f:
        for s in val_samples:
            f.write(json.dumps(s) + "\n")

    print(f"  Train: {len(train_samples):,} conversations -> {train_path}")
    print(f"  Val: {len(val_samples):,} conversations -> {val_path}")


def prepare_phase3(output_dir: str, max_docs: int = 50000):
    """
    Phase 3: Verification head training (correct vs buggy code).
    Source: codenet/Project_CodeNet (accepted/rejected submissions)

    Falls back to synthetic generation if CodeNet is unavailable.
    """
    out = Path(output_dir) / "phase3"
    out.mkdir(parents=True, exist_ok=True)

    train_samples = []
    val_samples = []

    try:
        from datasets import load_dataset
        print(f"Phase 3: Attempting to load CodeNet...")
        ds = load_dataset("ibm/codenet", split="train", streaming=True)

        count = 0
        for row in ds:
            if count >= max_docs:
                break

            code = row.get("code", "").strip()
            status = row.get("status", "")

            if not code or len(code) < 30:
                continue

            if status == "Accepted":
                sample = {"text": code, "label": 1, "source": "codenet_accepted"}
            elif status in ("Wrong Answer", "Runtime Error", "Compilation Error"):
                sample = {"text": code, "label": 0, "source": None}
            else:
                continue

            if random.random() < 0.02:
                val_samples.append(sample)
            else:
                train_samples.append(sample)

            count += 1
            if count % 10000 == 0:
                print(f"  {count:,} samples loaded...")

    except Exception as e:
        print(f"  CodeNet unavailable ({e}), generating synthetic grounding pairs...")
        train_samples, val_samples = _generate_synthetic_phase3(max_docs)

    train_path = out / "train.jsonl"
    val_path = out / "val.jsonl"

    with open(train_path, "w") as f:
        for s in train_samples:
            f.write(json.dumps(s) + "\n")

    with open(val_path, "w") as f:
        for s in val_samples:
            f.write(json.dumps(s) + "\n")

    print(f"  Train: {len(train_samples):,} pairs -> {train_path}")
    print(f"  Val: {len(val_samples):,} pairs -> {val_path}")


def _generate_synthetic_phase3(max_docs: int):
    """Synthetic grounding pairs for code verification."""
    correct = [
        "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
        "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
        "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1",
        "def is_palindrome(s):\n    return s == s[::-1]",
        "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)",
        "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, item):\n        self.items.append(item)\n    def pop(self):\n        return self.items.pop()",
        "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
        "def flatten(lst):\n    result = []\n    for item in lst:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result",
    ]

    buggy = [
        "def factorial(n):\n    if n <= 1:\n        return 0\n    return n * factorial(n - 1)",
        "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = a + b, a\n    return a",
        "def binary_search(arr, target):\n    lo, hi = 0, len(arr)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid\n        else:\n            hi = mid\n    return -1",
        "def is_palindrome(s):\n    return s == s[::1]",
        "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return left + right",
        "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, item):\n        self.items.append(item)\n    def pop(self):\n        return self.items.pop(0)",
        "def gcd(a, b):\n    while b:\n        a, b = b, a / b\n    return a",
        "def flatten(lst):\n    result = []\n    for item in lst:\n        result.extend(flatten(item))\n    return result",
    ]

    train_samples = []
    val_samples = []

    for _ in range(max_docs):
        if random.random() > 0.5:
            code = random.choice(correct)
            sample = {"text": code, "label": 1, "source": "synthetic_correct"}
        else:
            code = random.choice(buggy)
            sample = {"text": code, "label": 0, "source": None}

        if random.random() < 0.02:
            val_samples.append(sample)
        else:
            train_samples.append(sample)

    return train_samples, val_samples


def prepare_phase4(output_dir: str, max_docs: int = 20000):
    """
    Phase 4: DPO alignment with code preference pairs.
    Source: m-a-p/Code-Feedback
    """
    from datasets import load_dataset

    out = Path(output_dir) / "phase4"
    out.mkdir(parents=True, exist_ok=True)

    print(f"Phase 4: Downloading Code-Feedback...")
    ds = load_dataset("m-a-p/Code-Feedback", split="train")

    train_samples = []
    val_samples = []

    for i, row in enumerate(ds):
        if i >= max_docs:
            break

        messages = row.get("messages", [])
        if len(messages) < 2:
            continue

        # Extract the first user message as prompt
        prompt = None
        responses = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "").strip()
            if role == "user" and prompt is None:
                prompt = content
            elif role == "assistant" and prompt:
                responses.append(content)

        if not prompt or len(responses) < 1:
            continue

        # Use first response as chosen, generate a degraded version as rejected
        chosen = responses[0]
        rejected = _degrade_code(chosen)

        if not rejected or chosen == rejected:
            continue

        sample = {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }

        if random.random() < 0.02:
            val_samples.append(sample)
        else:
            train_samples.append(sample)

    train_path = out / "train.jsonl"
    val_path = out / "val.jsonl"

    with open(train_path, "w") as f:
        for s in train_samples:
            f.write(json.dumps(s) + "\n")

    with open(val_path, "w") as f:
        for s in val_samples:
            f.write(json.dumps(s) + "\n")

    print(f"  Train: {len(train_samples):,} preference pairs -> {train_path}")
    print(f"  Val: {len(val_samples):,} preference pairs -> {val_path}")


def _degrade_code(code: str) -> str:
    """
    Create a plausible but worse version of code for preference training.
    Applies random degradations: remove error handling, remove comments,
    use worse variable names, or truncate.
    """
    lines = code.split("\n")
    if len(lines) < 3:
        return ""

    mode = random.choice(["truncate", "strip_comments", "bad_names"])

    if mode == "truncate":
        cut = max(3, len(lines) * 2 // 3)
        return "\n".join(lines[:cut]) + "\n# ... incomplete"

    elif mode == "strip_comments":
        stripped = [l for l in lines if not l.strip().startswith("#") and not l.strip().startswith('"""')]
        if len(stripped) < len(lines) * 0.7:
            return "\n".join(stripped)
        return "\n".join(lines[:max(3, len(lines) - 3)])

    elif mode == "bad_names":
        result = code
        for old, new in [("result", "x"), ("items", "a"), ("target", "t"), ("index", "i2")]:
            if old in result:
                result = result.replace(old, new)
                break
        return result

    return ""


def main():
    parser = argparse.ArgumentParser(description="Prepare code training data for Regent")
    parser.add_argument("--output-dir", default="data", help="Base output directory")
    parser.add_argument("--max-docs", type=int, default=500000, help="Max documents for Phase 1")
    parser.add_argument("--phase1", action="store_true")
    parser.add_argument("--phase2", action="store_true")
    parser.add_argument("--phase3", action="store_true")
    parser.add_argument("--phase4", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    run_all = args.all or not (args.phase1 or args.phase2 or args.phase3 or args.phase4)

    if args.phase1 or run_all:
        prepare_phase1(args.output_dir, args.max_docs)
        print()

    if args.phase2 or run_all:
        prepare_phase2(args.output_dir)
        print()

    if args.phase3 or run_all:
        prepare_phase3(args.output_dir)
        print()

    if args.phase4 or run_all:
        prepare_phase4(args.output_dir)
        print()

    print("Data preparation complete.")
    print(f"Output directory: {args.output_dir}/")
    print()
    print("Next steps:")
    print("  1. Train tokenizer:  PYTHONPATH=. python3 scripts/train_tokenizer.py --input data/raw/train.txt --vocab-size 32768")
    print("  2. Prepare tokens:   PYTHONPATH=. python3 scripts/prepare_data.py --input data/raw/train.txt --tokenizer data/tokenizer/regent.model --output data/processed/train.npy")
    print("  3. Run pipeline:     PYTHONPATH=. python3 scripts/run_pipeline.py --config configs/regent_3b_code.yaml --start-stage 4")


if __name__ == "__main__":
    main()
