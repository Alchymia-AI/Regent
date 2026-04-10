"""
Data pipeline for Regent model training.

Supports three training phases:
    Phase 1 (base): Standard language modeling on text corpora
    Phase 2 (identity): Regent-specific conversations with EPG context
    Phase 3 (verification): Grounding score training for Ver Head

Data format (JSONL):
{
    "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ],
    "epg_nodes": [  // optional, for phase 2+
        {
            "key": "morning_preference",
            "value": "Prefers oat milk lattes",
            "confidence": 0.85,
            "activation": 0.7,
            "valence": 0.3,
            "emotional_weight": 0.5,
            "category": "preference"
        }
    ],
    "essence": {  // optional, for phase 2+
        "essence_index": 6.2,
        "essence_influence": 2.4,
        "truth_vs_lie": 0.8,
        "civility_vs_unruliness": 0.9,
        "good_vs_evil": 0.7,
        "curiosity": 0.6,
        "self_preservation": 0.3
    },
    "grounding_labels": [0.0, 1.0, 1.0, ...]  // optional, for phase 3
}
"""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np


# EPG category name -> integer ID
CATEGORY_MAP = {
    "identity": 0,
    "belief": 1,
    "capability": 2,
    "limitation": 3,
    "entity": 4,
    "relationship": 5,
    "trust": 6,
    "spatial": 7,
    "temporal": 8,
    "causal": 9,
    "domain": 10,
    "memory": 11,
    "outcome": 12,
    "preference": 13,
    "sensitivity": 14,
}


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class TextDataset:
    """
    Phase 1 dataset: plain text sequences for language modeling.

    Reads pre-tokenized numpy arrays (token IDs).
    """

    def __init__(self, token_file: str, seq_len: int):
        self.tokens = np.load(token_file).astype(np.int32)
        self.seq_len = seq_len
        self.n_sequences = (len(self.tokens) - 1) // seq_len

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, idx: int) -> dict:
        start = idx * self.seq_len
        end = start + self.seq_len + 1  # +1 for target shift
        chunk = self.tokens[start:end]
        return {
            "input_ids": mx.array(chunk[:-1]),
            "labels": mx.array(chunk[1:]),
        }


class RegentDataset:
    """
    Phase 2 dataset: Regent conversations with EPG context.

    Each sample includes:
        - Tokenized conversation
        - EPG node embeddings (tokenized key+value + scalar features)
        - Essence state vector
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_len: int,
        max_epg_nodes: int = 32,
        max_node_tokens: int = 32,
    ):
        self.records = load_jsonl(data_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_epg_nodes = max_epg_nodes
        self.max_node_tokens = max_node_tokens

    def __len__(self) -> int:
        return len(self.records)

    def _tokenize_conversation(self, messages: list[dict]) -> list[int]:
        """Convert messages to a flat token sequence."""
        tokens = [self.tokenizer.bos_id]
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            role_tokens = self.tokenizer.encode(f"<{role}>", add_bos=False)
            content_tokens = self.tokenizer.encode(content, add_bos=False)
            tokens.extend(role_tokens + content_tokens)
        tokens.append(self.tokenizer.eos_id)
        return tokens[: self.max_seq_len + 1]

    def _encode_epg_nodes(self, nodes: list[dict]) -> tuple[mx.array, mx.array, mx.array]:
        """Encode EPG nodes into token IDs, scalar features, and category IDs."""
        n = min(len(nodes), self.max_epg_nodes)

        node_tokens = np.zeros((n, self.max_node_tokens), dtype=np.int32)
        scalars = np.zeros((n, 5), dtype=np.float32)  # confidence, activation, valence, emotional_weight, extra
        categories = np.zeros((n,), dtype=np.int32)

        for i, node in enumerate(nodes[:n]):
            # Tokenize key+value
            ids = self.tokenizer.encode_epg_node(node["key"], node["value"], self.max_node_tokens)
            node_tokens[i, : len(ids)] = ids

            # Scalar features
            scalars[i] = [
                node.get("confidence", 0.5),
                node.get("activation", 0.5),
                node.get("valence", 0.0),
                node.get("emotional_weight", 0.5),
                0.0,  # reserved
            ]

            # Category
            categories[i] = CATEGORY_MAP.get(node.get("category", "domain"), 10)

        # Pad to max_epg_nodes
        if n < self.max_epg_nodes:
            pad_tokens = np.zeros((self.max_epg_nodes - n, self.max_node_tokens), dtype=np.int32)
            pad_scalars = np.zeros((self.max_epg_nodes - n, 5), dtype=np.float32)
            pad_cats = np.zeros((self.max_epg_nodes - n,), dtype=np.int32)
            node_tokens = np.concatenate([node_tokens, pad_tokens])
            scalars = np.concatenate([scalars, pad_scalars])
            categories = np.concatenate([categories, pad_cats])

        return mx.array(node_tokens), mx.array(scalars), mx.array(categories)

    def _encode_essence(self, essence: dict | None) -> mx.array:
        """Encode essence state into a 7-dim vector."""
        if essence is None:
            return mx.zeros((7,))

        return mx.array([
            essence.get("essence_index", 5.0),
            essence.get("essence_influence", 0.0),
            essence.get("truth_vs_lie", 0.0),
            essence.get("civility_vs_unruliness", 0.0),
            essence.get("good_vs_evil", 0.0),
            essence.get("curiosity", 0.5),
            essence.get("self_preservation", 0.3),
        ])

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]

        # Tokenize conversation
        tokens = self._tokenize_conversation(record["messages"])
        input_ids = mx.array(tokens[:-1], dtype=mx.int32)
        labels = mx.array(tokens[1:], dtype=mx.int32)

        result = {"input_ids": input_ids, "labels": labels}

        # EPG nodes
        if "epg_nodes" in record and record["epg_nodes"]:
            node_tokens, scalars, categories = self._encode_epg_nodes(record["epg_nodes"])
            result["epg_node_tokens"] = node_tokens
            result["epg_scalars"] = scalars
            result["epg_categories"] = categories

        # Essence
        if "essence" in record:
            result["essence"] = self._encode_essence(record["essence"])

        # Grounding labels (phase 3)
        if "grounding_labels" in record:
            labels_arr = record["grounding_labels"][:len(tokens) - 1]
            result["grounding_labels"] = mx.array(labels_arr, dtype=mx.float32)

        return result


def collate_batch(samples: list[dict], pad_id: int = 0) -> dict:
    """Collate samples into a padded batch."""
    batch = {}

    # Find max sequence length in batch
    max_len = max(s["input_ids"].shape[0] for s in samples)

    # Pad sequences
    input_ids = []
    labels = []
    for s in samples:
        seq_len = s["input_ids"].shape[0]
        pad_len = max_len - seq_len
        if pad_len > 0:
            input_ids.append(mx.concatenate([s["input_ids"], mx.full((pad_len,), pad_id, dtype=mx.int32)]))
            labels.append(mx.concatenate([s["labels"], mx.full((pad_len,), -100, dtype=mx.int32)]))
        else:
            input_ids.append(s["input_ids"])
            labels.append(s["labels"])

    batch["input_ids"] = mx.stack(input_ids)
    batch["labels"] = mx.stack(labels)

    # Stack optional fields if present in all samples
    if all("essence" in s for s in samples):
        batch["essence"] = mx.stack([s["essence"] for s in samples])

    if all("epg_node_tokens" in s for s in samples):
        batch["epg_node_tokens"] = mx.stack([s["epg_node_tokens"] for s in samples])
        batch["epg_scalars"] = mx.stack([s["epg_scalars"] for s in samples])
        batch["epg_categories"] = mx.stack([s["epg_categories"] for s in samples])

    if all("grounding_labels" in s for s in samples):
        gl = []
        for s in samples:
            g = s["grounding_labels"]
            pad_len = max_len - g.shape[0]
            if pad_len > 0:
                gl.append(mx.concatenate([g, mx.full((pad_len,), -1.0)]))
            else:
                gl.append(g)
        batch["grounding_labels"] = mx.stack(gl)

    return batch
