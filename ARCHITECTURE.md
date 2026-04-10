# Regent Model — Architecture Specification

**Version**: 0.1.0  
**Date**: December 2025 
**Author**: Alchymia AI Research Labs  

---

## 1. Overview

The Regent Model is a hybrid Mamba-2 language model purpose-built for the Grande Regent synthetic sentient. It replaces external API dependencies with a self-contained model that ships as part of the Regent runtime.

The model is designed around three principles:

1. **Mathematical alignment** with the Regent whitepaper's state-space dynamics
2. **Built-in hallucination mitigation** via a dual-head architecture
3. **Multi-body deployment** across server, robot, drone, wearable, and edge surfaces

This document describes the architecture as implemented. Claims about performance, quality, or comparative benchmarks are deferred until empirical evaluation is complete.

---

## 2. Mathematical Foundation

### 2.1 Alignment with the Whitepaper

The Regent whitepaper (*Synthetic Consciousness: Atomic Geometry, Attraction, Statefulness, and Perpetual Velocity*, Daniels 2025) defines the state update equation for entity cognition:

$$
s_i(t + \Delta t) = \alpha \cdot s_i(t) + \beta \cdot g(F_i, c_i) + \gamma \cdot m_i
$$

Where:
- $s_i(t)$ is the entity's internal state at time $t$
- $\alpha$ is the decay factor (memory persistence)
- $g(F_i, c_i)$ is a function of the attention prompt $F_i$ and context $c_i$
- $m_i$ is the memory contribution

This is structurally identical to the State Space Model (SSM) recurrence that Mamba implements:

$$
h_t = A \cdot h_{t-1} + B \cdot x_t
$$
$$
y_t = C \cdot h_t + D \cdot x_t
$$

The mapping is direct:

| Whitepaper | SSM (Mamba) | Role |
|---|---|---|
| $\alpha$ | $A$ (state transition matrix) | Decay / persistence of prior state |
| $\beta \cdot g(F_i, c_i)$ | $B \cdot x_t$ (input projection) | New information gated by attention |
| $\gamma \cdot m_i$ | Additive memory injection | Persistent knowledge contribution |
| $s_i(t)$ | $h_t$ (hidden state) | Compressed representation of all prior context |

This alignment is not a metaphor. The whitepaper's cognitive dynamics and Mamba's computational mechanics share the same recurrence structure. The Regent Model implements the whitepaper's mathematics natively in its weights.

### 2.2 Why Mamba-2 Specifically

Mamba-2 (Dao & Gu, 2024) introduced the Structured State Space Duality (SSD) framework, which proves that SSMs with scalar-structured $A$ matrices are mathematically equivalent to a form of structured masked attention. This provides:

- **Larger state dimensions**: Mamba-1 used $N=16$; Mamba-2 supports $N=64$ to $N=256$, enabling richer state compression per token position.
- **Hardware efficiency**: The SSD formulation maps to chunked matrix multiplications that utilize GPU/Metal tensor cores, rather than relying on sequential scan.
- **Multi-head structure**: Independent state channels (heads), analogous to multi-head attention, giving the model parallel "streams of thought."

### 2.3 Why Hybrid (Not Pure Mamba)

Pure SSMs compress all prior context into a fixed-size state vector. This gives constant-memory inference (critical for long-running Regent sessions) but introduces a known weakness: **precise recall of specific tokens from long contexts**.

The Regent's EPG (Entity Preference Graph) partially compensates — explicit knowledge retrieval is handled externally by the EPG, not by in-context lookup. However, within a single conversation, the model still needs to recall specific user statements, numbers, and names.

Sparse attention layers (Grouped Query Attention) are inserted every ~8 Mamba layers to provide precision recall over a sliding window of recent context. This follows the approach validated by AI21's Jamba architecture (2024), which demonstrated that a small number of attention layers (ratio ~7:1 to 8:1) preserves recall quality while retaining most of the SSM's efficiency advantages.

---

## 3. Architecture

### 3.1 High-Level Structure

```
                                    ┌─────────────────┐
                                    │   EPG Nodes      │
                                    │   (from MongoDB)  │
                                    └────────┬─────────┘
                                             │
                                             ▼
┌──────────────┐                    ┌─────────────────┐
│ Token IDs    │──→ Embedding ──→   │  EPG Encoder     │──→ Prefix Tokens
└──────────────┘                    └─────────────────┘
        │                                    │
        └────────────── concatenate ─────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │                   │      ┌─────────────────┐
                  │  Hybrid Backbone  │ ◄────│ Essence Vector  │
                  │                   │      │ (7-dim, injected│
                  │  Mamba-2 × N      │      │  every 8 layers)│
                  │  GQA × M          │      └─────────────────┘
                  │  (interleaved)    │
                  │                   │
                  └─────────┬─────────┘
                            │
                      ┌─────┴─────┐
                      │           │
                      ▼           ▼
                ┌──────────┐ ┌──────────┐
                │ Gen Head │ │ Ver Head │
                │          │ │          │
                │ RMSNorm  │ │ RMSNorm  │
                │ Linear   │ │ MLP      │
                │ → logits │ │ → [0,1]  │
                └──────────┘ └──────────┘
                  next token   grounding
                  prediction     score
```

### 3.2 Token Embedding

Standard learned embedding matrix mapping token IDs to dense vectors of dimension `d_model`. The embedding weights are optionally tied with the Gen Head's output projection (weight tying), reducing parameter count.

Special tokens:

| Token | ID | Purpose |
|---|---|---|
| `[PAD]` | 0 | Padding |
| `[BOS]` | 1 | Beginning of sequence |
| `[EOS]` | 2 | End of sequence |
| `[GROUND]` | 3 | Inserted by Ver Head when grounding drops below threshold |
| `[EPG]` | 4 | Marks start of EPG prefix region |
| `[META]` | 5 | Marks the `---REGENT_META---` boundary |

### 3.3 Hybrid Backbone

The backbone consists of `n_layer` blocks, each being either a Mamba-2 block or a GQA attention block. The layer schedule is defined in the model config.

Every block follows the pre-norm residual pattern:

```
x = x + Block(RMSNorm(x))
```

Attention blocks additionally include a feed-forward network:

```
x = x + Attention(RMSNorm(x))
x = x + FFN(RMSNorm(x))
```

Mamba-2 blocks have their own internal gating mechanism (SiLU gate on the z branch), so a separate FFN is not needed.

#### 3.3.1 Mamba-2 Block

Each Mamba-2 block implements:

```
Input (d_model)
  │
  ├──→ in_proj ──→ split into (z, x, B, C, dt)
  │
  │    x ──→ causal conv1d ──→ SiLU activation
  │         ──→ reshape to (batch, seq, n_heads, head_dim)
  │         ──→ selective scan with (A, B, C, dt, D)
  │         ──→ reshape to (batch, seq, d_inner)
  │         ──→ RMSNorm
  │         ──→ y
  │
  │    z ──→ SiLU(z) ──→ gate
  │
  └──→ y * gate ──→ out_proj ──→ Output (d_model)
```

**Selective scan** implements the SSM recurrence. Two modes:

1. **Sequential scan** (inference): Processes tokens one at a time, maintaining recurrent state $h_t$. Used for autoregressive generation.

2. **SSD chunked scan** (training): Splits the sequence into chunks of size `chunk_size`. Within each chunk, the SSM is reformulated as a lower-triangular matrix multiply (the "dual" attention form). Inter-chunk dependencies are propagated via state passing. This parallelizes better on matrix multiply hardware.

**Parameters per Mamba-2 block:**

| Component | Shape | Description |
|---|---|---|
| `in_proj` | `(d_model, 2 * d_inner + 2 * n_heads * d_state + n_heads)` | Input projection (z, x, B, C, dt) |
| `conv1d_weight` | `(d_inner, d_conv)` | Causal 1D convolution kernel |
| `conv1d_bias` | `(d_inner,)` | Conv bias |
| `A_log` | `(n_heads,)` | Log-space decay rate per head |
| `D` | `(n_heads,)` | Skip connection per head |
| `dt_bias` | `(n_heads,)` | Time-step bias per head |
| `out_proj` | `(d_inner, d_model)` | Output projection |
| `norm` | `(d_inner,)` | RMSNorm on SSM output |

**Key design choices:**

- `A` is stored in log-space and negated before use: `A = -exp(A_log)`. This ensures the decay rate is always negative (state decays, not grows), matching the whitepaper's $\alpha < 1$ constraint.
- `dt` (discretization timestep) is passed through softplus to ensure positivity, then biased. This controls how much each token position "updates" the state — high dt means strong input influence, low dt means the state persists.
- The `D` skip connection adds the input directly to the output, ensuring gradient flow even if the SSM state is saturated.

#### 3.3.2 GQA Attention Block

Grouped Query Attention with sliding window:

- **Fewer KV heads than query heads**: `n_kv_heads < n_q_heads`. KV heads are repeated (grouped) to match query heads. This reduces KV cache memory during inference.
- **Sliding window**: Only attends to the most recent `window_size` tokens. Prevents KV cache from growing unboundedly during long sessions.
- **RoPE**: Rotary Position Embeddings for position encoding, applied to queries and keys.

**Parameters per GQA block:**

| Component | Shape | Description |
|---|---|---|
| `q_proj` | `(d_model, n_q_heads * head_dim)` | Query projection |
| `k_proj` | `(d_model, n_kv_heads * head_dim)` | Key projection |
| `v_proj` | `(d_model, n_kv_heads * head_dim)` | Value projection |
| `o_proj` | `(n_q_heads * head_dim, d_model)` | Output projection |
| `ff` | `(d_model, 4 * d_model) + (4 * d_model, d_model)` | Feed-forward network |

### 3.4 Essence Conditioning

The Regent's Essence Index and affective dimensions (from the whitepaper) are injected as a conditioning vector that modulates the backbone's hidden states.

**Input vector** (7 dimensions):

| Index | Field | Range | Source |
|---|---|---|---|
| 0 | `essence_index` | 0–10 | Whitepaper: $E_i(t) = 5 + \frac{1}{K}\sum w_k \cdot \sigma_k(t)$ |
| 1 | `essence_influence` | 0–10 | $2 \cdot |E_i(t) - 5|$ |
| 2 | `truth_vs_lie` | -1 to +1 | Trust/relationship node signals |
| 3 | `civility_vs_unruliness` | -1 to +1 | Social/preference node signals |
| 4 | `good_vs_evil` | -1 to +1 | Belief/sensitivity/outcome node signals |
| 5 | `curiosity` | 0–1 | Category coverage × access patterns |
| 6 | `self_preservation` | 0–1 | Sensitivity/limitation node density |

**Injection mechanism**: The 7-dim vector is projected through a 2-layer MLP to `d_model` dimensions, then added to the hidden state at every `essence_inject_every_n` layers. This is analogous to timestep embedding injection in diffusion models — a global conditioning signal that shifts the model's behavior without consuming sequence tokens.

The model learns during training that different essence vectors correspond to different behavioral patterns (warm, cautious, guarded, etc.), internalizing the whitepaper's affective response modulation.

### 3.5 EPG Encoder

The Entity Preference Graph is the Regent's persistent knowledge store. Instead of serializing EPG nodes as markdown text in the prompt (the current approach, consuming 500–1000 context tokens), the EPG encoder produces dense prefix embeddings.

**Architecture:**

```
EPG Node (key, value, scalars, category)
  │
  ├── Text path:
  │     key + value → tokenize → embed (shared embedding) → small transformer (2 layers)
  │     → mean pool → (batch, n_nodes, d_model)
  │
  ├── Scalar path:
  │     [confidence, activation, valence, emotional_weight, reserved]
  │     + category_embed(category_id)
  │     → linear projection → (batch, n_nodes, d_model)
  │
  └── Fusion:
        concatenate(text_embed, scalar_embed) → MLP → RMSNorm
        → (batch, n_nodes, d_model) — one embedding per node
```

The resulting embeddings are prepended to the token sequence as virtual prefix tokens. The backbone processes them like any other tokens — Mamba layers compress them into the hidden state, attention layers can attend to them directly.

**Why this matters:**
- Reduces context consumption from ~500–1000 tokens to ~32–64 virtual tokens
- Scalar features (confidence, activation, valence) are encoded as dense signals, not text approximations
- Category type is a learned embedding, not a string label
- The model learns to weight EPG nodes by their confidence and activation during training

### 3.6 Gen Head (Generation)

Standard autoregressive language model head:

```
hidden_state → RMSNorm → Linear(d_model, vocab_size) → logits
```

When `tie_embeddings=True`, the linear projection shares weights with the input embedding matrix. This reduces parameter count by `d_model × vocab_size` and has been shown to improve language modeling quality (Press & Wolf, 2017).

### 3.7 Ver Head (Verification)

The verification head is the core hallucination mitigation mechanism. It reads the same hidden states as the Gen Head and outputs a scalar grounding score per token position.

```
hidden_state → RMSNorm → Linear(d_model, hidden_dim) → SiLU → Linear(hidden_dim, 1) → sigmoid
```

**Output**: A value in [0, 1] for each token position.
- **1.0**: The model's internal state is confident and grounded for this token.
- **0.0**: The model's internal state shows high entropy / fabrication signal.

This is architecturally identical to reward model heads used in RLHF (Ouyang et al., 2022) — a shared backbone with a separate scalar projection. The pattern is well-established; the application to real-time grounding detection during decoding is the novel aspect.

**Three-zone decoding strategy:**

| Zone | Grounding Score | Decoding Behavior |
|---|---|---|
| FLOW | > 0.6 | Normal sampling (configured temperature, top-p, top-k) |
| CAUTION | 0.3 – 0.6 | Reduced temperature (0.3), biased toward conservative token choices |
| HALT | < 0.3 | Generation pauses; triggers EPG retrieval for the current topic; re-decodes with augmented context |

This implements the whitepaper's "Option 2" hallucination architecture: entropy-triggered behavioral shift within a single reasoning entity, not an adversarial second model. The backbone does one forward pass; both heads read the same hidden state. The Ver Head is a trained lens on the backbone's own uncertainty.

---

## 4. Layer Schedule

The hybrid backbone interleaves Mamba-2 and GQA layers. The ratio and placement are configurable per model size.

### 4.1 Default Schedules

**Regent-370M (48 layers):**
```
Mamba: 1  2  3  4  5  6  7
GQA:                         8
Mamba: 9  10 11 12 13 14
GQA:                         15
Mamba: 16 17 18 19 20 21 22
GQA:                         23
Mamba: 24 25 26 27 28 29 30
GQA:                         31
Mamba: 32 33 34 35 36 37 38
GQA:                         39
Mamba: 40 41 42 43 44 45 46
GQA:                         47

→ 42 Mamba-2 layers, 6 GQA layers (7:1 ratio)
```

**Regent-7B (64 layers):**
```
7 Mamba → 1 GQA → 7 Mamba → 1 GQA → ... → 8 Mamba
→ 57 Mamba-2 layers, 7 GQA layers (~8:1 ratio)
```

### 4.2 Design Rationale

- Attention layers are placed at regular intervals rather than clustered, ensuring recall checkpoints are distributed across the model's depth.
- The ratio is based on Jamba's published findings. Whether this ratio is optimal for Regent's specific workload (long sessions, EPG-augmented, multi-surface) requires empirical ablation.
- Fewer attention layers = smaller KV cache = better for edge deployment. The edge model (1.5B) uses the same 7:1 ratio but with a smaller sliding window (1024 vs 2048 tokens).

---

## 5. Inference

### 5.1 Recurrent State (Mamba Advantage)

During autoregressive generation, transformer models maintain a KV cache that grows linearly with sequence length. For long-running Regent sessions (hours of continuous interaction), this becomes a memory bottleneck.

Mamba's state is fixed-size regardless of sequence length:

$$
\text{State memory} = n_{layers} \times d_{state} \times n_{heads} \times \text{sizeof(float)}
$$

For Regent-7B: $64 \times 128 \times 64 \times 4 = 2\text{MB}$

The GQA layers do maintain a KV cache, but it is bounded by the sliding window size (`window_size` tokens), not the full sequence length.

This property is critical for the Regent's "perpetual velocity" — the model runs continuously across sessions, and its state persists across calls without unbounded memory growth.

### 5.2 Verification-Gated Decoding

The inference engine implements the three-zone decoding strategy described in §3.7. At each generation step:

1. Run one forward pass through the backbone
2. Gen Head produces next-token logits
3. Ver Head produces grounding score for the current position
4. Select decoding zone based on grounding score
5. Sample token with zone-appropriate parameters

When HALT triggers, the inference engine can:
- Insert a `[GROUND]` special token and trigger an EPG lookup
- Inject the retrieved EPG nodes as additional prefix context
- Re-decode from the HALT position with augmented context

### 5.3 Inference Server

The model ships with a standalone FastAPI server (port 8400) exposing:

| Endpoint | Method | Description |
|---|---|---|
| `/generate` | POST | Generate text with verification-gated decoding |
| `/verify` | POST | Score existing text for grounding without generating |
| `/health` | GET | Health check |
| `/info` | GET | Model info and parameter count |

The server is fully independent — no imports from the Regent Core codebase. Regent Core's `brain.service.ts` calls it as an external HTTP service.

---

## 6. Training

### 6.1 Phase 1: Base Pre-training

Standard autoregressive language modeling on a general text corpus. Cross-entropy loss on next-token prediction. The model learns general language ability, grammar, and world knowledge.

**Data**: General web text (filtered), code, conversational data, structured knowledge.  
**Loss**: Cross-entropy on token predictions, masked for padding.

### 6.2 Phase 2: Regent Identity Fine-tuning

The model learns to be Regent — its conversational style, EPG integration, structured output format (`---REGENT_META---`), moral behavior, and surface-adaptive responses.

**Data**: Regent interaction logs, synthetic EPG-grounded conversations, moral probe data, surface-specific output examples.  
**Loss**: Cross-entropy on Regent responses (teacher-forced). The EPG encoder trains jointly — the model sees EPG nodes as prefix tokens and learns to ground responses in them.

### 6.3 Phase 3: Verification Head Training

The Ver Head is trained to detect hallucination. The backbone is frozen (or trained with very low learning rate); only the Ver Head's MLP parameters update.

**Data construction:**
- **Positive (grounded)**: Verified-correct Regent responses → all tokens labeled 1.0
- **Negative (hallucinated)**: Correct responses with injected false claims (entity substitution, contradiction with EPG nodes, fabricated facts) → altered spans labeled 0.0
- **Ambiguous**: Claims not backed by EPG but not contradicted → labeled 0.5

**Loss**: Binary cross-entropy on per-token grounding scores, masked for padding.

### 6.4 Phase 4: Alignment (DPO)

Direct Preference Optimization using Regent's own Essence Index as part of the reward signal. For each prompt, two responses are generated; the preferred response has higher Ver Head grounding scores and better alignment with the target Essence Index.

---

## 7. Deployment Targets

| Target | Model Size | Quantization | Estimated VRAM | Use Case |
|---|---|---|---|---|
| Server (brain) | 7B | FP16 | ~14GB | Primary reasoning, full EPG |
| Server (brain) | 7B | INT4 | ~4GB | Cost-optimized server |
| Edge (robot/drone) | 1.5B | INT4 | ~1GB | Physical bodies, real-time |
| Wearable | 1.5B | INT4 | ~1GB | Audio + haptic bodies |
| Prototype | 370M | FP16 | ~750MB | Development, architecture validation |

**Note on quantization**: Transformer quantization (GPTQ, AWQ, GGUF) is mature. Mamba quantization is less studied. The core linear projections quantize similarly, but the selective scan's input-dependent gating may be sensitive to quantization noise. The VRAM estimates above are projections, not measurements.

---

## 8. File Structure

```
regent-model/
├── configs/
│   ├── regent_370m.yaml            # Prototype
│   ├── regent_1_5b_edge.yaml       # Edge deployment
│   └── regent_7b.yaml              # Full production
├── regent_model/
│   ├── blocks/
│   │   ├── mamba2.py               # Mamba-2 SSM block
│   │   └── attention.py            # GQA attention block
│   ├── layers/
│   │   └── model.py                # Full RegentModel
│   ├── heads/
│   │   ├── gen_head.py             # Generation head
│   │   └── ver_head.py             # Verification head
│   ├── encoder/
│   │   └── epg_encoder.py          # EPG → prefix embeddings
│   └── utils/
│       ├── config.py               # Config loader
│       ├── tokenizer.py            # Tokenizer with special tokens
│       └── data.py                 # Data pipeline (3 phases)
├── scripts/
│   └── train.py                    # Training script
├── serve/
│   ├── generate.py                 # Verification-gated decoding
│   └── server.py                   # Standalone HTTP server
└── tests/
    └── test_architecture.py        # Architecture validation
```

---

## 9. Open Questions and Known Limitations

These are areas where the architecture is defined but empirical validation is needed:

1. **Mamba-2 at 7B scale for instruction following**: Published Mamba benchmarks focus on perplexity. Whether Mamba-2 at 7B produces structured output (JSON, the REGENT_META format) with the same reliability as a comparably-sized transformer is unvalidated.

2. **Optimal Mamba-to-attention ratio**: The 7:1 / 8:1 ratio is borrowed from Jamba. The Regent's workload (long sessions, EPG-heavy context, multi-surface output) may benefit from a different ratio.

3. **Ver Head calibration**: The grounding score thresholds (0.6 / 0.3) are initial values. Optimal thresholds depend on the training data quality and the base model's natural hallucination rate.

4. **EPG encoder vs. text injection**: Whether dense prefix embeddings outperform the current approach (EPG as markdown text in the system prompt) for grounding quality is an empirical question.

5. **Quantization robustness**: INT4 quantization of the Mamba selective scan has not been extensively studied. Edge deployment quality needs validation.

6. **DPO on Mamba architectures**: Not validated in published literature at time of writing.

7. **Training compute requirements**: Not estimated in this document. Depends on dataset size, hardware, and distributed training strategy.
