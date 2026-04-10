"""
Data scraping pipeline — collect training corpus from multiple sources.

Scrapes text from configured sources and writes one-document-per-line
output files ready for tokenization.

Sources:
    - Local text files / directories
    - URLs (web pages, stripped to text)
    - HuggingFace datasets (if `datasets` library installed)
    - Regent interaction logs (MongoDB export)

Usage:
    PYTHONPATH=. python3 scripts/scrape_corpus.py --config pipeline.yaml
    PYTHONPATH=. python3 scripts/scrape_corpus.py --urls urls.txt --output data/raw/web.txt
    PYTHONPATH=. python3 scripts/scrape_corpus.py --local /path/to/texts/ --output data/raw/local.txt
    PYTHONPATH=. python3 scripts/scrape_corpus.py --hf-dataset wikimedia/wikipedia --hf-split train --hf-column text --output data/raw/wiki.txt --max-docs 100000
"""

import argparse
import json
import re
import sys
import html
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Clean a document for training."""
    # Decode HTML entities
    text = html.unescape(text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove control characters (keep newlines for splitting)
    text = "".join(c for c in text if c.isprintable() or c in "\n\t")
    return text.strip()


def extract_text_from_html(html_content: str) -> str:
    """Extract readable text from HTML, stripping tags and scripts."""
    # Remove script and style blocks
    text = re.sub(r"<script[^>]*>[\s\S]*?</script>", "", html_content, flags=re.IGNORECASE)
    text = re.sub(r"<style[^>]*>[\s\S]*?</style>", "", text, flags=re.IGNORECASE)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode entities
    text = html.unescape(text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Source: Local files
# ---------------------------------------------------------------------------

def scrape_local(path: str, max_docs: int | None = None) -> list[str]:
    """Read text from local file(s). One document per line."""
    p = Path(path)
    docs = []

    if p.is_file():
        files = [p]
    elif p.is_dir():
        files = sorted(p.glob("**/*.txt")) + sorted(p.glob("**/*.md"))
    else:
        print(f"  Warning: path not found: {path}")
        return []

    for fp in files:
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = clean_text(line)
                if len(line) > 50:  # skip very short lines
                    docs.append(line)
                    if max_docs and len(docs) >= max_docs:
                        return docs

    return docs


# ---------------------------------------------------------------------------
# Source: URLs
# ---------------------------------------------------------------------------

def scrape_url(url: str, timeout: int = 15) -> str | None:
    """Fetch a URL and extract text."""
    try:
        req = Request(url, headers={"User-Agent": "RegentModelScraper/0.1"})
        with urlopen(req, timeout=timeout) as resp:
            content = resp.read().decode("utf-8", errors="ignore")

        if "<html" in content.lower() or "<body" in content.lower():
            text = extract_text_from_html(content)
        else:
            text = clean_text(content)

        return text if len(text) > 100 else None

    except (URLError, TimeoutError, Exception) as e:
        print(f"  Failed to fetch {url}: {e}")
        return None


def scrape_urls(urls_file: str, max_docs: int | None = None) -> list[str]:
    """Scrape text from a list of URLs (one per line)."""
    docs = []
    with open(urls_file) as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    print(f"  Scraping {len(urls)} URLs...")
    for i, url in enumerate(urls):
        text = scrape_url(url)
        if text:
            # Split long pages into paragraphs as separate docs
            paragraphs = [p.strip() for p in text.split(". ") if len(p.strip()) > 50]
            for para in paragraphs:
                docs.append(para)
                if max_docs and len(docs) >= max_docs:
                    return docs

        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{len(urls)} URLs processed, {len(docs)} docs collected")

    return docs


# ---------------------------------------------------------------------------
# Source: HuggingFace datasets
# ---------------------------------------------------------------------------

def scrape_hf_dataset(dataset_name: str, split: str = "train",
                      column: str = "text", max_docs: int | None = None) -> list[str]:
    """Load text from a HuggingFace dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  Error: `datasets` library required. Install with: pip install datasets")
        return []

    print(f"  Loading {dataset_name} (split={split}, column={column})...")
    ds = load_dataset(dataset_name, split=split, streaming=True)

    docs = []
    for i, row in enumerate(ds):
        text = row.get(column, "")
        if isinstance(text, str) and len(text) > 50:
            cleaned = clean_text(text)
            if len(cleaned) > 50:
                # Flatten to single line
                docs.append(cleaned.replace("\n", " "))

        if max_docs and len(docs) >= max_docs:
            break

        if (i + 1) % 10000 == 0:
            print(f"    {len(docs)} docs collected from {i + 1} rows...")

    return docs


# ---------------------------------------------------------------------------
# Source: Regent interaction logs (MongoDB JSON export)
# ---------------------------------------------------------------------------

def scrape_regent_logs(log_path: str, max_docs: int | None = None) -> list[str]:
    """
    Extract training text from Regent interaction logs.

    Expected format: JSONL with {messages: [{role, content}], ...}
    This is the same format that brain.service.ts produces.
    """
    docs = []
    p = Path(log_path)

    if not p.exists():
        print(f"  Warning: log path not found: {log_path}")
        return []

    files = [p] if p.is_file() else sorted(p.glob("**/*.jsonl"))

    for fp in files:
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    messages = record.get("messages", [])
                    for msg in messages:
                        content = msg.get("content", "")
                        if len(content) > 50:
                            docs.append(clean_text(content.replace("\n", " ")))
                except json.JSONDecodeError:
                    continue

                if max_docs and len(docs) >= max_docs:
                    return docs

    return docs


# ---------------------------------------------------------------------------
# Pipeline config
# ---------------------------------------------------------------------------

def run_pipeline(config_path: str):
    """Run scraping pipeline from a YAML config file."""
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg.get("output_dir", "data/raw"))
    output_dir.mkdir(parents=True, exist_ok=True)

    all_docs = []

    for source in cfg.get("sources", []):
        src_type = source["type"]
        max_docs = source.get("max_docs")
        print(f"\nProcessing source: {src_type}")

        if src_type == "local":
            docs = scrape_local(source["path"], max_docs)
        elif src_type == "urls":
            docs = scrape_urls(source["file"], max_docs)
        elif src_type == "huggingface":
            docs = scrape_hf_dataset(
                source["dataset"], source.get("split", "train"),
                source.get("column", "text"), max_docs)
        elif src_type == "regent_logs":
            docs = scrape_regent_logs(source["path"], max_docs)
        else:
            print(f"  Unknown source type: {src_type}")
            continue

        print(f"  Collected {len(docs)} documents")
        all_docs.extend(docs)

    # Shuffle and split
    import random
    random.seed(cfg.get("seed", 42))
    random.shuffle(all_docs)

    val_ratio = cfg.get("val_ratio", 0.05)
    split = int(len(all_docs) * (1 - val_ratio))

    train_docs = all_docs[:split]
    val_docs = all_docs[split:]

    # Write
    train_path = output_dir / "train.txt"
    val_path = output_dir / "val.txt"

    with open(train_path, "w") as f:
        for doc in train_docs:
            f.write(doc + "\n")

    with open(val_path, "w") as f:
        for doc in val_docs:
            f.write(doc + "\n")

    print(f"\nPipeline complete:")
    print(f"  Total documents: {len(all_docs)}")
    print(f"  Train: {len(train_docs)} → {train_path}")
    print(f"  Val: {len(val_docs)} → {val_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Scrape training corpus")
    parser.add_argument("--config", default=None, help="Pipeline config YAML")
    parser.add_argument("--local", default=None, help="Local text file/directory")
    parser.add_argument("--urls", default=None, help="File with URLs (one per line)")
    parser.add_argument("--hf-dataset", default=None, help="HuggingFace dataset name")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--hf-column", default="text")
    parser.add_argument("--regent-logs", default=None, help="Regent interaction logs (JSONL)")
    parser.add_argument("--output", default="data/raw/corpus.txt")
    parser.add_argument("--max-docs", type=int, default=None)
    args = parser.parse_args()

    if args.config:
        run_pipeline(args.config)
        return

    docs = []

    if args.local:
        print(f"Scraping local: {args.local}")
        docs.extend(scrape_local(args.local, args.max_docs))

    if args.urls:
        print(f"Scraping URLs from: {args.urls}")
        docs.extend(scrape_urls(args.urls, args.max_docs))

    if args.hf_dataset:
        print(f"Loading HF dataset: {args.hf_dataset}")
        docs.extend(scrape_hf_dataset(
            args.hf_dataset, args.hf_split, args.hf_column, args.max_docs))

    if args.regent_logs:
        print(f"Extracting from Regent logs: {args.regent_logs}")
        docs.extend(scrape_regent_logs(args.regent_logs, args.max_docs))

    if not docs:
        print("No documents collected. Specify --local, --urls, --hf-dataset, or --regent-logs")
        sys.exit(1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for doc in docs:
            f.write(doc + "\n")

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nCollected {len(docs)} documents → {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
