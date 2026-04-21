"""
Generate synthetic Phase 4 preference pair data for pipeline smoke testing.

Output format (JSONL):
    {"prompt": "...", "chosen": "...", "rejected": "..."}
"""

import argparse
import json
import random
from pathlib import Path

PROMPTS = [
    "What does the verification head do?",
    "How does memory work in Regent?",
    "Explain the HALT zone.",
    "What is the EPG encoder?",
    "How does tool calling work?",
    "What happens when confidence drops?",
    "Why is this architecture different?",
    "How does the thinking block work?",
    "What is essence conditioning?",
    "How are knowledge nodes structured?",
]

CHOSEN = [
    "The verification head reads the same hidden state as the generation head and outputs a confidence score between 0 and 1 per token. It adds less than 0.1% to the total parameter count.",
    "Regent uses a Mamba-2 state-space model that compresses all past context into a fixed-size buffer. Memory does not grow with session length.",
    "When the verification head scores a token below 0.3, generation stops. The system retrieves relevant knowledge from the EPG graph and restarts from the uncertain point.",
    "The EPG encoder converts structured knowledge nodes into dense prefix embeddings. Each node carries confidence, activation, valence, emotional weight, and category.",
    "The model emits a [TOOL_CALL] token followed by a JSON request. Generation pauses. The caller executes the tool and posts the result back via [TOOL_RESULT].",
    "In the CAUTION zone (0.3 to 0.6), temperature drops and the model picks conservative tokens. Below 0.3, it halts and retrieves context.",
    "Regent is a Mamba-2 state-space model with grouped-query attention at selected layers. Transformers use self-attention at every layer. The recurrent backbone gives fixed memory.",
    "The model emits [THINK], reasons internally, then emits [/THINK] before the visible answer. Thinking tokens are not included in the output.",
    "A 7-dimensional behavioral vector is projected and added to hidden states every 8 layers. It shapes tone and behavior without using prompt space.",
    "Each node has key, value, confidence, activation, valence, emotional_weight, and category. They are tokenized and processed by the EPG encoder into prefix embeddings.",
]

REJECTED = [
    "The verification head is a separate model that runs after generation is complete. It requires a second forward pass.",
    "Regent uses a KV cache that grows linearly with conversation length, similar to standard transformers.",
    "When confidence drops, the model continues generating but marks low-confidence tokens for later review by the user.",
    "The EPG encoder flattens knowledge nodes into plain text and injects them into the prompt as a system message.",
    "Tool calling is handled by an external plugin layer that intercepts the model output and matches patterns to registered tools.",
    "When confidence drops, the model increases temperature to explore more diverse completions and find a better answer.",
    "Regent uses the same transformer architecture as GPT but with a verification head added on top.",
    "The thinking block is a prompt engineering technique where the system prompt instructs the model to output reasoning before answering.",
    "Essence conditioning is achieved by prepending a natural language description of the desired behavior to the prompt.",
    "Knowledge nodes are stored as plain text key-value pairs in a JSON file that the model reads as part of the system prompt.",
]


def generate_pair() -> dict:
    idx = random.randint(0, len(PROMPTS) - 1)
    return {
        "prompt": PROMPTS[idx],
        "chosen": CHOSEN[idx],
        "rejected": REJECTED[idx],
    }


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Phase 4 data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w") as f:
        for _ in range(args.count):
            f.write(json.dumps(generate_pair()) + "\n")

    print(f"Generated {args.count} preference pairs → {args.output}")


if __name__ == "__main__":
    main()
