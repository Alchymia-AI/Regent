"""
Generate synthetic Phase 2 conversational data for pipeline smoke testing.

Output format (JSONL):
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
"""

import argparse
import json
import random
from pathlib import Path

QUESTIONS = [
    "What does the verification head do?",
    "How does the memory stay constant?",
    "Explain the three-zone decoding strategy.",
    "What is the EPG encoder?",
    "How does essence conditioning work?",
    "What happens when grounding drops below 0.3?",
    "Why is this not a transformer?",
    "How does the model call tools?",
    "What is the thinking block used for?",
    "How are knowledge nodes encoded?",
    "What is the difference between Regent and Grande Regent?",
    "How does the model handle long sessions?",
    "What is the HALT zone?",
    "How is the behavioral vector injected?",
    "What does a grounding score of 0.8 mean?",
]

ANSWERS = [
    "The verification head reads the same hidden state as the generation head and outputs a confidence score between 0 and 1 for each token.",
    "Memory stays constant because the Mamba-2 state-space model compresses all past context into a fixed-size buffer that does not grow with sequence length.",
    "Three zones: FLOW above 0.6 writes normally, CAUTION between 0.3 and 0.6 lowers temperature, HALT below 0.3 stops and retrieves from the knowledge graph.",
    "The EPG encoder converts structured knowledge nodes into dense prefix embeddings that the model processes as native input.",
    "A 7-dimensional vector is projected through a two-layer MLP and added to hidden states every 8 layers during generation.",
    "The model stops generating, invokes the retrieval callback to fetch relevant EPG nodes, and restarts generation with the augmented context.",
    "Regent uses a Mamba-2 state-space backbone with grouped-query attention at selected layers. Transformers use full self-attention at every layer.",
    "The model emits a [TOOL_CALL] token, generates a JSON tool request, and pauses until the caller provides the result via [TOOL_RESULT].",
    "The thinking block allows the model to reason internally before producing visible output. Thinking tokens are collected but not shown to the user.",
    "Each node has a key, value, confidence, activation, valence, emotional weight, and category. These are tokenized and projected into prefix embeddings.",
    "Regent is 7B to 50B open source. Grande Regent is 70B to 1T commercial with enterprise support.",
    "The Mamba-2 state compresses all past context into a fixed buffer. A session running 8 hours uses the same memory as one running 8 seconds.",
    "HALT means the verification head scored the current token below 0.3. The model stops, retrieves relevant facts, and tries again.",
    "The essence vector is projected to model dimension and added to the hidden state at every 8th layer.",
    "A score of 0.8 means the model is reasonably confident the token is grounded in its knowledge. It falls in the FLOW zone.",
]


def generate_conversation() -> dict:
    idx = random.randint(0, len(QUESTIONS) - 1)
    q = QUESTIONS[idx]
    a = ANSWERS[idx]
    # Sometimes add a follow-up
    if random.random() > 0.6:
        follow_idx = random.randint(0, len(QUESTIONS) - 1)
        return {
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
                {"role": "user", "content": QUESTIONS[follow_idx]},
                {"role": "assistant", "content": ANSWERS[follow_idx]},
            ]
        }
    return {
        "messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Phase 2 data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w") as f:
        for _ in range(args.count):
            f.write(json.dumps(generate_conversation()) + "\n")

    print(f"Generated {args.count} conversations → {args.output}")


if __name__ == "__main__":
    main()
