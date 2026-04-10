# Adding New Model Sizes

This guide explains how to define a new Regent model size, what parameters to set, and how they relate to each other.

---

## 1. Create a Config File

Create a new YAML file in `configs/`. Use an existing config as a template:

```bash
cp configs/regent_370m.yaml configs/regent_YOUR_SIZE.yaml
```

---

## 2. Core Dimensions

These three parameters determine the model's scale:

```yaml
model:
  d_model: ???      # hidden dimension (width)
  n_layer: ???      # total number of layers (depth)
  vocab_size: 32768 # keep constant across sizes (tokenizer-dependent)
```

### How to choose d_model and n_layer

The total parameter count is approximately:

```
params ≈ (3 × d_model × d_inner × n_mamba_layers)
       + (4 × d_model² × n_attn_layers)    # attention + FFN
       + (vocab_size × d_model)              # embeddings
       + (small terms)                       # norms, biases, heads
```

Where `d_inner = expand × d_model` (typically `expand = 2`, so `d_inner = 2 × d_model`).

Simplified: **params ≈ 6 × d_model² × n_layer** (for mostly-Mamba models with expand=2).

Reference scaling table:

| Target Size | d_model | n_layer | Approx Params |
|---|---|---|---|
| 130M | 768 | 24 | ~130M |
| 370M | 1024 | 48 | ~370M |
| 790M | 1536 | 48 | ~790M |
| 1.5B | 2048 | 48 | ~1.5B |
| 2.8B | 2560 | 64 | ~2.8B |
| 7B | 4096 | 64 | ~7B |
| 13B | 5120 | 80 | ~13B |

**Rules of thumb:**
- `d_model` should be divisible by `n_heads` (SSM) and `n_q_heads` (attention)
- Powers of 2 or multiples of 256 work best for hardware efficiency
- Deeper models (more layers) generalize better than wider models at the same parameter count, up to a point
- For a given param budget, start with the d_model from the table above and adjust n_layer

---

## 3. SSM Parameters

```yaml
  ssm:
    expand: 2           # expansion factor: d_inner = expand × d_model
    d_state: ???        # state dimension per head
    d_conv: 4           # causal conv1d kernel width (keep at 4)
    n_heads: ???        # number of SSM heads
```

### expand

Keep at **2**. This is the standard value across all published Mamba models. Changing it scales `d_inner` and thus parameter count linearly.

### d_state

The SSM state dimension per head. This controls how much information each head can compress from prior context into its fixed-size state.

| Model Size | d_state | Rationale |
|---|---|---|
| ≤ 790M | 64 | Sufficient for prototype/edge; keeps state memory small |
| 1.5B–2.8B | 64–128 | Balance between capacity and memory |
| 7B+ | 128 | Full capacity; Mamba-2 SSD makes this tractable |
| 13B+ | 128–256 | Diminishing returns above 128 not established |

**Total state memory** = `n_layer × n_heads × d_state × head_dim × sizeof(float)`. Keep this in mind for edge deployment.

### n_heads

Number of SSM heads. Each head has dimension `head_dim = d_inner / n_heads`.

**Constraint**: `d_inner` must be divisible by `n_heads`.

**Recommended**: Set `head_dim = 128` and derive `n_heads = d_inner / 128`.

| d_model | d_inner (expand=2) | head_dim | n_heads |
|---|---|---|---|
| 768 | 1536 | 128 | 12 |
| 1024 | 2048 | 128 | 16 |
| 2048 | 4096 | 128 | 32 |
| 4096 | 8192 | 128 | 64 |
| 5120 | 10240 | 128 | 80 |

### d_conv

Keep at **4**. This is the standard causal convolution kernel width. Larger values give more local context before the SSM but increase parameters and latency minimally.

---

## 4. Attention Parameters

```yaml
  attention:
    layers: [7, 15, 23, ...]    # which layers are GQA (0-indexed)
    n_q_heads: ???
    n_kv_heads: ???
    head_dim: ???
    window_size: ???
```

### layers

Which layer indices use GQA instead of Mamba-2. Place them at regular intervals.

**Formula**: For a model with `n_layer` layers and a target ratio of `R` Mamba layers per 1 attention layer:

```python
attn_interval = R + 1  # e.g., 7:1 ratio → interval of 8
attn_layers = [i for i in range(attn_interval - 1, n_layer, attn_interval)]
```

**Recommended ratios:**

| Model Size | Ratio (Mamba:Attention) | Rationale |
|---|---|---|
| ≤ 790M | 7:1 | Standard, matches Jamba |
| 1.5B–7B | 7:1 to 8:1 | Slight increase for larger models |
| Edge / Wearable | 7:1 | Keep attention layers few to minimize KV cache |

### n_q_heads, n_kv_heads, head_dim

GQA (Grouped Query Attention) parameters.

- `n_kv_heads < n_q_heads`: fewer KV heads reduces KV cache memory
- Each group of `n_q_heads / n_kv_heads` query heads shares one KV head
- `n_q_heads` must be divisible by `n_kv_heads`

**Recommended:**

| d_model | n_q_heads | n_kv_heads | head_dim | KV cache per layer per token |
|---|---|---|---|---|
| 768 | 8 | 2 | 96 | 384 bytes |
| 1024 | 16 | 4 | 64 | 512 bytes |
| 2048 | 16 | 4 | 128 | 1024 bytes |
| 4096 | 32 | 8 | 128 | 2048 bytes |

**Constraint**: `n_q_heads × head_dim` should equal `d_model` (or close to it) for the output projection dimensions to work cleanly. If not equal, the attention output is projected back to `d_model` via `o_proj`.

### window_size

Sliding window size for attention. Only the most recent `window_size` tokens are kept in the KV cache.

| Deployment | window_size | Rationale |
|---|---|---|
| Server | 2048–4096 | Ample context for conversation |
| Edge | 1024 | Conserve memory on constrained devices |
| Prototype | 512–1024 | Faster iteration |

---

## 5. Head Parameters

```yaml
  gen_head:
    tie_embeddings: true    # share weights with input embedding

  ver_head:
    hidden_dim: ???         # MLP hidden dimension
    enabled: true           # set false to disable hallucination mitigation
```

### gen_head.tie_embeddings

**Keep true** unless you have a reason to decouple. Weight tying saves `d_model × vocab_size` parameters and is standard practice.

### ver_head.hidden_dim

The MLP hidden dimension in the verification head. This is small relative to the model — it's just a lightweight probe on the backbone's hidden states.

**Recommended**: `hidden_dim ≈ d_model / 8` to `d_model / 4`.

| d_model | ver_head.hidden_dim |
|---|---|
| 768 | 96–192 |
| 1024 | 128–256 |
| 2048 | 192–256 |
| 4096 | 256–512 |

---

## 6. EPG Encoder Parameters

```yaml
  epg_encoder:
    max_nodes: ???          # max EPG nodes in prefix
    scalar_features: 5     # keep at 5 (confidence, activation, valence, emotional_weight, reserved)
    n_categories: 15       # keep at 15 (matches EPG NodeCategory enum)
    category_embed_dim: ???
    n_encoder_layers: ???
    encoder_heads: ???
```

### max_nodes

How many EPG nodes are injected as prefix tokens. Each node becomes one virtual token.

| Deployment | max_nodes | Rationale |
|---|---|---|
| Server | 64 | Full knowledge injection |
| Edge | 16–32 | Conserve context space |
| Prototype | 16–32 | Faster iteration |

### category_embed_dim

Learned embedding dimension for the 15 EPG node categories. Scales with d_model.

**Recommended**: `category_embed_dim ≈ d_model / 128` to `d_model / 64`, minimum 4.

| d_model | category_embed_dim |
|---|---|
| 768 | 6–8 |
| 1024 | 8–12 |
| 2048 | 12–16 |
| 4096 | 16–32 |

### n_encoder_layers and encoder_heads

The EPG text encoder is a small transformer. It doesn't need to be large — it's encoding short key+value pairs, not reasoning.

| Model Size | n_encoder_layers | encoder_heads |
|---|---|---|
| ≤ 790M | 1–2 | 2–4 |
| 1.5B–7B | 2–4 | 4–8 |
| 13B+ | 4 | 8 |

---

## 7. Essence Conditioning

```yaml
  essence:
    input_dim: 7            # keep at 7 (matches whitepaper dimensions)
    inject_every_n: ???     # inject conditioning every N layers
```

### inject_every_n

How frequently the essence vector is added to hidden states.

- **Lower values** (4): Stronger conditioning influence; the model's behavior is more tightly coupled to the essence state
- **Higher values** (16): Weaker influence; the model has more freedom to diverge from the conditioning

**Recommended**: 8 for most sizes. Reduce to 4 for smaller models (< 1B) where the conditioning signal needs to be stronger to be effective. Increase to 16 for very large models (13B+) where the backbone has enough capacity to internalize the conditioning.

---

## 8. Training Parameters

```yaml
training:
  max_seq_len: ???
  batch_size: ???
  gradient_accumulation: ???
  lr: ???
  min_lr: ???
  warmup_steps: ???
  max_steps: ???
  weight_decay: 0.1       # standard, rarely needs changing
  grad_clip: 1.0           # standard
  dtype: float16           # or bfloat16 if hardware supports it
```

### Learning rate scaling

Larger models generally use smaller learning rates:

| Model Size | lr | min_lr |
|---|---|---|
| 130M–370M | 3e-4 | 3e-5 |
| 790M–1.5B | 2e-4 | 2e-5 |
| 2.8B–7B | 1.5e-4 | 1.5e-5 |
| 13B+ | 1e-4 | 1e-5 |

### Batch size and accumulation

Effective batch size = `batch_size × gradient_accumulation`.

- Larger effective batch sizes are generally better for language model training
- Constrained by available memory
- Target effective batch size of 16–64 for small models, 64–256 for large models

### max_seq_len

Should match the model's attention window_size for attention layers. Can be longer — Mamba layers handle any length, and attention layers will just use the sliding window.

---

## 9. Validation Checklist

After creating a new config, validate it:

1. **Parameter count**: Run the model with your config and check `model.count_parameters()`. Verify it matches your target size.

2. **Forward pass**: Run `python tests/test_architecture.py` with your config loaded. Verify shapes are correct.

3. **Gradient flow**: Verify gradients are non-zero for all trainable parameters.

4. **Memory fit**: Estimate peak memory usage:
   ```
   peak_memory ≈ 2 × params × sizeof(dtype)  # weights + gradients
                + batch_size × seq_len × d_model × n_layer × sizeof(dtype) × 2  # activations (rough)
   ```
   Ensure this fits in your available GPU/Metal memory.

5. **Divisibility**: Verify:
   - `d_inner` is divisible by `ssm.n_heads`
   - `n_q_heads` is divisible by `n_kv_heads`
   - Attention layer indices are within `[0, n_layer)`

---

## 10. Example: Adding a 2.8B Config

```yaml
# configs/regent_2_8b.yaml
model:
  d_model: 2560
  n_layer: 64
  vocab_size: 32768

  ssm:
    expand: 2            # d_inner = 5120
    d_state: 128
    d_conv: 4
    n_heads: 40          # head_dim = 5120 / 40 = 128

  attention:
    layers: [7, 15, 23, 31, 39, 47, 55, 63]  # 8 attention layers, 56 Mamba (7:1)
    n_q_heads: 20
    n_kv_heads: 4
    head_dim: 128
    window_size: 2048

  gen_head:
    tie_embeddings: true

  ver_head:
    hidden_dim: 192
    enabled: true

  epg_encoder:
    max_nodes: 48
    scalar_features: 5
    n_categories: 15
    category_embed_dim: 12
    n_encoder_layers: 3
    encoder_heads: 6

  essence:
    input_dim: 7
    inject_every_n: 8

  norm_eps: 1.0e-5
  initializer_range: 0.02

training:
  max_seq_len: 2048
  batch_size: 4
  gradient_accumulation: 8
  lr: 2.0e-4
  min_lr: 2.0e-5
  warmup_steps: 1500
  max_steps: 200000
  weight_decay: 0.1
  grad_clip: 1.0
  dtype: float16

  ver_head:
    lr: 8.0e-5
    freeze_backbone: true
    epochs: 8
```

Then validate:

```bash
python -c "
from regent_model.layers.model import RegentModel, RegentConfig
cfg = RegentConfig.from_yaml('configs/regent_2_8b.yaml')
model = RegentModel(cfg)
print(model.count_parameters())
"
```
