"""
Generate synthetic Phase 3 grounding pair data for pipeline smoke testing.

Output format (JSONL):
    {"text": "...", "label": 1, "source": "node_id"}
    {"text": "...", "label": 0, "source": null}
"""

import argparse
import json
import random
from pathlib import Path

GROUNDED_CLAIMS = [
    ("The verification head outputs a score between 0 and 1.", "ver_head_spec"),
    ("Mamba-2 uses selective scan for state updates.", "mamba2_paper"),
    ("The model accepts structured knowledge nodes as input.", "epg_encoder_spec"),
    ("Grouped-query attention reduces memory by sharing KV heads.", "gqa_paper"),
    ("The essence vector has 7 dimensions.", "essence_config"),
    ("Phase 1 trains on next-token prediction.", "training_pipeline"),
    ("The model uses RoPE for positional encoding.", "attention_impl"),
    ("Attention layers are placed every 8 recurrent layers.", "architecture_spec"),
    ("The EPG encoder produces prefix embeddings.", "epg_encoder_spec"),
    ("DPO uses a frozen reference model.", "alignment_spec"),
    ("The HALT zone triggers retrieval when grounding is below 0.3.", "decoding_spec"),
    ("The model supports tool calling via special tokens.", "tool_calling_spec"),
    ("Thinking blocks use [THINK] and [/THINK] tokens.", "thinking_spec"),
    ("The 7B model uses 64 layers total.", "config_7b"),
    ("SafeTensors format is used for checkpoints.", "checkpoint_spec"),
]

FABRICATED_CLAIMS = [
    "The verification head uses a transformer with 12 layers.",
    "Regent requires a minimum of 128GB RAM.",
    "The model uses BERT-style masked language modeling.",
    "Attention is applied at every layer in the backbone.",
    "The essence vector has 32 dimensions.",
    "Phase 1 trains using reinforcement learning from human feedback.",
    "The model uses absolute positional embeddings.",
    "The EPG encoder uses a convolutional architecture.",
    "DPO trains both the policy and the reference model simultaneously.",
    "The HALT zone triggers when grounding exceeds 0.9.",
    "Tool calling requires an external plugin system.",
    "The model cannot operate without internet connectivity.",
    "Memory grows linearly with conversation length.",
    "The 7B model uses 128 layers total.",
    "The model requires PyTorch for inference.",
]


def generate_pair() -> dict:
    if random.random() > 0.5:
        text, source = random.choice(GROUNDED_CLAIMS)
        return {"text": text, "label": 1, "source": source}
    else:
        text = random.choice(FABRICATED_CLAIMS)
        return {"text": text, "label": 0, "source": None}


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Phase 3 data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w") as f:
        for _ in range(args.count):
            f.write(json.dumps(generate_pair()) + "\n")

    print(f"Generated {args.count} grounding pairs → {args.output}")


if __name__ == "__main__":
    main()
