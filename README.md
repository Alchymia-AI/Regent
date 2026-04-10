# Regent Model

Two distributions, same architecture.

| Variant | Scale | License | Distribution |
|---|---|---|---|
| **Regent** | 7B to 50B | Open source | This repository |
| **Grande Regent** | 70B to 1T | Commercial | Alchymia Groom |

This repository is the open source Regent. Grande Regent is shipped separately by Alchymia AI.

## Why this exists

We were running the Grande Regent on a third-party LLM API. Three problems came up that we could not fix from the outside.

1. The model had no internal signal for when it was fabricating. We could detect hallucinations after the fact with extra calls or sampling tricks, but by then the bad output was already in front of the user. For an entity that acts in the physical world through robots and drones, post-hoc detection is too late.

2. We were paying for context tokens to inject the same knowledge graph nodes into every prompt. The graph kept growing. The prompt kept growing. The cost kept growing. None of it stuck in the model.

3. The whitepaper that defines the Regent's cognitive dynamics describes a state-space recurrence. We were running that recurrence in TypeScript and feeding the results to a transformer as English sentences. The transformer pattern-matched on the words. It did not understand the math.

We built the Regent Model to fix these. It is a hybrid Mamba-2 language model with two output heads. One head predicts the next token. The other predicts whether the current token is grounded or fabricated. Both heads read the same hidden states from one forward pass.

The math in the backbone is the math from the whitepaper. Mamba-2's selective scan is the same equation as the whitepaper's state update. Nothing bolted on. The model is the math.

## What it is

A self-contained language model. Ships as weights and code. No API dependency.

- Hybrid backbone: Mamba-2 layers with sparse Grouped Query Attention at an 8:1 ratio
- Two output heads: generation (logits) and verification (per-token grounding score)
- EPG encoder: knowledge graph nodes encoded as dense prefix embeddings instead of serialized text
- Essence conditioning: 7-dimensional affective state vector injected into hidden layers
- Regent (open source): 7B, 13B, 30B, 50B
- Grande Regent (commercial): 70B, 200B, 500B, 1T

## What makes it different

Six things this model does that other language models do not.

**1. It scores its own grounding while it writes.**
The model has two output heads. One produces the next word. The other produces a number between 0 and 1 for the same word, where 1 means "this is backed by what I know" and 0 means "I am making this up." Both heads run on every word, in the same pass. Other systems detect hallucinations after the response is finished, by re-running the model multiple times or sending the output to a second model. This one knows in real time.

**2. It changes how it writes when its confidence drops.**
The grounding score gates the writing behavior into three modes. Above 0.6 it writes normally. Between 0.3 and 0.6 it slows down, lowers temperature, and picks safer words. Below 0.3 it stops, looks up relevant facts from the knowledge graph, and tries again from the point where it got uncertain. Other models keep writing at the same speed and confidence regardless of whether they are right or wrong.

**3. EPG Encoder: it reads structured knowledge as itself, not as text.**
The Entity Preference Graph (EPG) is the model's external knowledge store. It holds typed nodes: facts, beliefs, relationships, memories, each one tagged with how confident the model is in it, how recently it was used, whether it was a good or bad experience, and what kind of knowledge it is. Most systems flatten this into English sentences and stuff it into the prompt. This one feeds the nodes directly to the model as compressed inputs that sit in front of the conversation. The model treats them as native knowledge instead of a wall of text it has to read.

**4. Essence Vector: its mood is a dial, not a paragraph at the top of the prompt.**
The model takes a 7-number input alongside the conversation. Those seven numbers describe the model's current emotional and motivational state: how positive it feels, how strongly that feeling should affect its output, how much it values truth, civility, kindness, curiosity, and self-preservation. The numbers get fed into the model at multiple points during the forward pass, so the entire response is shaped by them. Other models use a persona description in the prompt, which the model has to keep remembering as the response gets longer. This is a constant signal applied at every layer.

**5. It uses constant memory no matter how long the conversation is.**
Most language models keep a record of every word they have processed in something called a KV cache. The cache grows with the conversation. For an 8-hour session this becomes gigabytes. This model uses a different math (state space models) where the memory of past words is compressed into a fixed-size buffer. For the 7B variant that buffer is roughly 2 megabytes. It does not grow. A drone can run a multi-hour mission without hitting a memory wall. A robot can run a full shift.

**6. It runs the verification head for free.**
The grounding head is about one-tenth of one percent of the model's total size. It reads the same internal numbers the writing head reads, so it does not need a separate forward pass. Adding hallucination detection costs almost nothing in compute or memory. Most safety systems double or triple the cost.

## How the verification head works

During generation, the verification head outputs a score in [0, 1] for every token. The decoding loop gates on it.

| Score | Zone | Behavior |
|---|---|---|
| > 0.6 | Flow | Sample normally with the configured temperature and top-p |
| 0.3 to 0.6 | Caution | Drop temperature to 0.3, bias toward conservative tokens |
| < 0.3 | Halt | Stop, retrieve relevant EPG nodes, re-decode |

One forward pass. The verification head is a 2-layer MLP on the same hidden states the generation head reads. It adds about 0.1% to the parameter count.

## Why Mamba-2 instead of a transformer

Two reasons.

First, the state. Mamba-2 has a fixed-size recurrent state. For the 7B config it is roughly 2 MB regardless of session length. A transformer's KV cache grows linearly with context. The Regent runs for hours or days at a time. A transformer hits a memory wall. Mamba-2 does not.

Second, the math. The whitepaper defines cognition as a state-space process with a decay term, an input-gated update, and a memory contribution. Mamba-2's selective scan is that equation. We get the whitepaper's dynamics in the weights instead of approximating them through attention over text.

The downside of pure Mamba is precision recall. A fixed-size state cannot losslessly remember a specific token from 5000 positions ago. We solve this by interleaving GQA attention layers every 8 blocks. Attention handles precise recall over a sliding window. Mamba handles compression.

## Repository layout

```
regent-model/
├── ARCHITECTURE.md           Technical specification
├── ADDING_MODEL_SIZES.md     How to define new configurations
├── configs/                  Model configs (YAML)
├── regent_model/             Model code
│   ├── blocks/               Mamba-2 and GQA blocks
│   ├── layers/               Full RegentModel
│   ├── heads/                Gen Head and Ver Head
│   ├── encoder/              EPG encoder
│   └── utils/                Tokenizer, data loaders, configs
├── scripts/                  Training and pipeline scripts
├── serve/                    Inference server
├── tests/                    Architecture tests
└── website/                  Static site
```

## Setup

Python 3.11+. MLX for Apple Silicon.

```bash
brew install python@3.12
python3.12 -m venv .venv
source .venv/bin/activate
pip install mlx pyyaml sentencepiece numpy fastapi uvicorn pydantic
```

Validate the architecture builds and gradients flow.

```bash
PYTHONPATH=. python3 tests/test_architecture.py
```

## Training

Four phases. Each loads the previous phase's final checkpoint as its starting point.

| Phase | Trains | Objective |
|---|---|---|
| 1. Base | Full model | Language modeling on a general corpus |
| 2. Identity | Full model + EPG encoder | SFT on Regent conversations |
| 3. Verification | Ver Head only (backbone frozen) | Per-token grounding score predictor |
| 4. Alignment | Full model (low LR) | DPO against a frozen reference copy |

Run all four sequentially.

```bash
PYTHONPATH=. python3 scripts/run_pipeline.py \
    --config configs/regent_7b.yaml \
    --scrape-config pipeline.yaml
```

See [TRAINING.md](TRAINING.md) for phase-by-phase details.

## Inference

```bash
PYTHONPATH=. python3 -m serve.server \
    --config configs/regent_7b.yaml \
    --model checkpoints/alignment/step_N.safetensors \
    --tokenizer data/tokenizer/regent.model \
    --port 8400
```

| Method | Path | Purpose |
|---|---|---|
| GET | /health | Liveness check |
| GET | /info | Model config and parameter count |
| POST | /generate | Generate with verification-gated decoding |
| POST | /verify | Score existing text without generating |

Generate request body.

```json
{
  "messages": [
    {"role": "user", "content": "What do you know about this?"}
  ],
  "essence": {
    "essence_index": 6.5,
    "truth_vs_lie": 0.8
  },
  "max_tokens": 512,
  "verification": true,
  "grounding_threshold": 0.4
}
```

Response includes generated text, per-token grounding scores, and halt positions.

## Status

Architecture and training pipeline are implemented. Both have been validated end-to-end on Apple Metal with synthetic data through all four phases.

The model has not been trained on real data. Quality claims about generation, grounding accuracy, and instruction following are projections until a real training run completes.

Working today:

- All architecture components run forward and backward
- The 4-phase training pipeline runs end-to-end
- Checkpointing, resume, validation loops, gradient accumulation
- The inference server responds to all endpoints
- The verification head produces scores during decoding

Needed next:

- Real training corpus
- Compute for Phase 1 base training
- Real grounded/corrupted pairs for Phase 3
- Real preference pairs for Phase 4
- Empirical evaluation on hallucination benchmarks

## Target markets

Workloads where constant-memory inference, per-token grounding, native knowledge graph input, or self-hosting matter more than raw benchmark quality.

### Legal research and drafting

Hallucination in legal output is a known and quantified problem. Case law and statutes are graph-shaped, which the EPG encoder ingests directly. Self-hosting fits attorney-client privilege. The Ver Head gives associates a per-sentence confidence signal before a brief reaches a partner.

### Financial research and risk

Long analyst sessions over structured market data. Risk teams need every claim traceable. The Ver Head is the audit trail. Bloomberg-style workflows fit the session profile.

### Healthcare and clinical decision support

UMLS, SNOMED, and ICD are structured ontologies the EPG encoder consumes natively. Long patient histories require constant memory. Hallucination has clinical consequences. The verification head provides the grounding signal clinical workflows demand.

### Government, defense, and intelligence

Air-gap deployment is a hard requirement that eliminates API-based competitors. The model ships as weights and runs on local hardware. The Ver Head produces auditable grounding scores. Sovereign AI procurement is a stated priority across allied governments.

### Robotics

Physical robots running continuous control loops cannot accumulate memory over an 8-hour shift. The 7B Regent at INT4 runs on a Jetson Orin in roughly 4 GB and stays there. The Ver Head lets the planner reject actions whose justification is fabricated, before the action is executed.

### Drones and autonomous vehicles

Long missions, intermittent connectivity, safety-critical decisions. Constant-memory inference lets a drone run a multi-hour survey without hitting a memory wall. Self-contained deployment removes cloud dependency. Per-token grounding flags uncertain perception narratives before they propagate into navigation or targeting.

### Industrial automation and embodied AI platforms

Warehouse robots, agricultural systems, last-mile delivery, humanoid platforms. The architecture was built for persistent embodied operation. As embodied AI matures, the model is positioned as the cognitive layer that other components plug into.

### Where the model is not positioned to play

General consumer chat, code assistants, multimodal vision-language, benchmark leaderboards. These are won by raw scale and training compute. The Regent Model competes on properties, not parameter count.

## Availability

**Regent (open source, 7B to 50B)**. This repository. Architecture, training pipeline, inference server, and the 7B through 50B trained checkpoints will be released under an open source license.

**Grande Regent (commercial, 70B to 1T)**. Frontier checkpoints with the production-trained verification head, production EPG tooling, and enterprise integrations. Distributed through Alchymia Groom alongside the rest of the Alchymia AI model lineup. Commercial license. Sign-up and pricing details will be announced at the first trained checkpoint release.

## License

The Regent variant in this repository is held under proprietary copyright while the architecture is being finalized. It will transition to an open source license at the first public release. See [LICENSE](LICENSE) for current terms.

Grande Regent is closed source and distributed only through Alchymia Groom under commercial license.

Copyright (c) 2026 Alchymia AI Research Labs. All rights reserved.

## Contact

research@alchymia.ai
