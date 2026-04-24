# Code Model

End-to-end recipe for training a Regent code model.

---

## Hardware

| Machine | Phase 1 wall-clock to val PPL 9 |
|---|---|
| A100 40 GB | ~2 h |
| H100 80 GB | ~1 h |

Production-scale (3B+, 100B+ tokens) needs cluster compute.

## Environment

```bash
python3.12 -m venv .venv && source .venv/bin/activate

# Apple Silicon
pip install mlx pyyaml sentencepiece numpy fastapi uvicorn pydantic safetensors datasets

# NVIDIA
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install pyyaml sentencepiece numpy fastapi uvicorn pydantic safetensors transformers datasets
```

---

## Data

All sourced from HuggingFace via `scripts/prepare_code_data.py`.

| Phase | Dataset | ID | Size used | License |
|---|---|---|---|---|
| 1 | CodeParrot Clean | `codeparrot/codeparrot-clean` | 10K–500K docs | MIT-equivalent (GitHub) |
| 2 | Magicoder OSS-Instruct | `ise-uiuc/Magicoder-OSS-Instruct-75K` | 75K | MIT |
| 3 | CodeNet | `ibm/codenet` (with synthetic fallback) | 50K pairs | Apache 2.0 |
| 4 | Code-Feedback | `m-a-p/Code-Feedback` | 20K pairs | Apache 2.0 |

Phase 3 falls back to synthetic correct/buggy pairs if CodeNet is unreachable. Swap for real grounding data before production.

---

## Config

Canonical: [`configs/regent_code_test.yaml`](../configs/regent_code_test.yaml).

| Field | 72M | Scaling rule |
|---|---|---|
| `d_model` | 512 | 2560 @ 3B, 4096 @ 7B |
| `n_layer` | 16 | 32 @ 3B, 64 @ 7B |
| `ssm.d_state` | 64 | 256 @ 3B, 1024 @ 7B |
| `ssm.n_heads` | 8 | keep `head_dim = 128` |
| `attn_layers` | `[7, 15]` | every 8 layers |
| `max_seq_len` | 256 | 4096 in production |
| `batch_size × grad_accum` | 1 × 8 | ≥ 32 effective |
| `dtype` | fp32 | bf16 on NVIDIA; fp32 on Apple Silicon |
| `lr` | 1.0e-4 | scale linearly with effective batch |
| `warmup_steps` | 1000 | 2000+ for larger models |
| `grad_clip` | 0.5 | Mamba is sensitive, keep tight |

### Stability (required)

In `regent_model/blocks/mamba2.py`:

```python
dt = nn.softplus(dt_raw) + self.dt_bias[None, None, :]
dt = mx.clip(dt, 1e-4, 1.0)

A_log_clamped = mx.clip(self.A_log, -8.0, 4.0)
A = -mx.exp(A_log_clamped)
```

Without these, loss goes NaN around step 150–200. Already applied as of `e3da2c9e`.

---

## Pipeline

```bash
# 1. Data
PYTHONPATH=. python3 scripts/prepare_code_data.py --all --max-docs 10000

# 2. Tokenizer
PYTHONPATH=. python3 scripts/train_tokenizer.py \
    --input data/raw/train.txt \
    --vocab-size 32768

# 3. Pack tokens
PYTHONPATH=. python3 scripts/prepare_data.py \
    --input data/raw/train.txt \
    --tokenizer data/tokenizer/regent.model \
    --output data/processed/train.npy \
    --seq-len 256

PYTHONPATH=. python3 scripts/prepare_data.py \
    --input data/raw/val.txt \
    --tokenizer data/tokenizer/regent.model \
    --output data/processed/val.npy \
    --seq-len 256

# 4. Train (unbuffered so logs flush)
PYTHONPATH=. python3 -u scripts/train.py \
    --config configs/regent_code_test.yaml \
    --train-data data/processed/train.npy \
    --val-data data/processed/val.npy \
    --log-interval 10 \
    --save-interval 500 \
    --val-interval 200
```

Or run all 4 phases:

```bash
PYTHONPATH=. python3 scripts/run_pipeline.py --config configs/regent_code_test.yaml
```

---

## Observed trajectory (72M, M3 Ultra, fp32)

| Step | Val PPL |
|---|---|
| 200 | 38.0 |
| 400 | 22.2 |
| 800 | 14.7 |
| 1600 | 10.1 |
| 2800 | 8.29 |
| 5000 | ~6.5 (projected) |
| 10000 | ~5.5 (projected) |
| 20000 | ~5.0 (data-saturated) |

Throughput ~210 tok/s. Memory ~1 GB resident.

72M on 40M tokens is ~555 tokens/param — heavily data-limited. PPL floor is set by data, not steps.

---

## Pitfalls

| Symptom | Fix |
|---|---|
| Loss NaN around step 180 | Apply the Mamba-2 clamps above. |
| Loss stuck at 10+ | Check LR, verify gradient norms non-zero. |
| "Resource limit exceeded" on MLX | Reduce `max_seq_len` or `batch_size`. |
| Nothing in stdout for 30+ min | Run Python with `-u`. |
| Val ≫ train loss | Overfitting — reduce steps or add data. |
| Tokenizer: vocab size too high | Corpus too small for requested vocab — lower `--vocab-size`. |

---

## Scaling

| Target | Config | Data | Compute |
|---|---|---|---|
| 3B | [`configs/regent_3b_code.yaml`](../configs/regent_3b_code.yaml) | 60B (Chinchilla) / 300B (competitive) | 8× A100 for 2-3 weeks, or 1× H100 for ~4 months |
| 7B | [`configs/regent_7b.yaml`](../configs/regent_7b.yaml) | 300B–2T | 8× H100 for 4-8 weeks |

### Production data sources

| Phase | Dataset | Size |
|---|---|---|
| 1 | The Stack v2 | ~3 TB |
| 2 | Magicoder + CommitPackFT + Evol-Instruct-Code | 900K+ |
| 3 | CodeNet + Defects4J + BugsInPy | 14M+ |
| 4 | Code-Feedback + CodeUltraFeedback | 100K+ |
