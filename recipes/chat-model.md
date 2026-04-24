# Chat Model

End-to-end recipe for training a Regent general-purpose chat model. Same 4-phase structure as the code recipe, different data and sequence budget. Pipeline is identical; no chat run has been executed on this repo yet.

---

## Hardware

| Machine | Target |
|---|---|
| A100 40 GB | 1B full pipeline |
| 8× H100 80 GB | 7B full pipeline, 300B tokens, ~3–4 weeks |
| Multi-node cluster | 70B+ |

Chat conversations are longer than code snippets. Budget sequence length and SSM state accordingly.

## Environment

Same as code recipe. See [code-model.md](code-model.md#environment).

---

## Data

| Phase | Dataset | ID | Size typical | License |
|---|---|---|---|---|
| 1 | FineWeb-Edu | `HuggingFaceFW/fineweb-edu` | 50B–1T tokens | ODC-By |
| 1 (alt) | SlimPajama | `cerebras/SlimPajama-627B` | 627B | mixed |
| 1 (alt) | Dolma | `allenai/dolma` | 3T | ODC-By |
| 2 | Tulu 3 SFT | `allenai/tulu-3-sft-mixture` | 1M convs | ODC-By |
| 2 | UltraChat 200K | `HuggingFaceH4/ultrachat_200k` | 200K | MIT |
| 2 | OpenAssistant 2 | `OpenAssistant/oasst2` | 128K | Apache 2.0 |
| 2 | LMSYS-Chat-1M | `lmsys/lmsys-chat-1m` | 1M | custom ToS |
| 3 | TruthfulQA | `truthful_qa` | 817 Q/A | Apache 2.0 |
| 3 | SimpleQA | `lighteval/simpleqa` | 4K | MIT |
| 4 | UltraFeedback | `HuggingFaceH4/ultrafeedback_binarized` | 64K pairs | MIT |
| 4 | HelpSteer 2 | `nvidia/HelpSteer2` | 10K | CC-BY-4.0 |
| 4 | Anthropic HH-RLHF | `Anthropic/hh-rlhf` | 170K | MIT |

### Phase 1 mix (by token fraction)

| Component | Fraction |
|---|---|
| Web (FineWeb-Edu) | 65% |
| Books + papers (SlimPajama subset) | 15% |
| Code (The Stack, Python/JS/Rust) | 15% |
| Math (OpenWebMath, AlgebraicStack) | 5% |

Code at 15% improves reasoning and structure-following. Llama 3 and Qwen 3 push higher.

---

## Config

Use an existing config and adjust the fields below. A dedicated chat config is not yet in the repo; `configs/regent_370m.yaml` and `configs/regent_7b.yaml` are the current targets.

| Field | Chat value | Reason |
|---|---|---|
| `max_seq_len` | 2048 (validation), 4096–8192 (production) | Conversations are longer than code snippets |
| `ssm.d_state` | 256 (small), 1024 (7B+) | Longer context demands larger SSM state |
| `attn_window_size` | 2048+ | Cover a full conversation turn |
| `tokenizer vocab_size` | 32768; 65536 if corpus > 10B tokens | NL-heavy text benefits from larger vocab |
| `dtype` | fp32 on Apple Silicon, bf16 on NVIDIA | |
| `batch × grad_accum` | ≥ 256 effective | Chat benefits from large effective batch |
| `lr` | 3e-4 (pretrain), 2e-5 (SFT) | Standard; scale with batch |

Stability clamps in `mamba2.py` are required and already applied. See [code-model.md](code-model.md#stability-required).

### Role boundaries

Training data uses literal string markers: `<system>`, `<user>`, `<assistant>`. Tokenized as multi-token sequences. The tokenizer already reserves structural specials (`[THINK]`, `[TOOL_CALL]`, etc).

---

## Pipeline

### Phase 1 data prep

Until `scripts/prepare_chat_data.py` exists, prepare Phase 1 manually:

```python
from datasets import load_dataset
from pathlib import Path

out = Path("data/raw/train.txt")
out.parent.mkdir(parents=True, exist_ok=True)
ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                  split="train", streaming=True)
with open(out, "w") as f:
    for i, row in enumerate(ds):
        if i >= 500_000:
            break
        text = row["text"].replace("\n", "\\n")
        if len(text) > 50:
            f.write(text + "\n")
```

### Phase 2–4 data prep

JSONL formats match the code recipe. For Phase 2, swap `ise-uiuc/Magicoder-OSS-Instruct-75K` for `allenai/tulu-3-sft-mixture`. For Phase 4, swap `m-a-p/Code-Feedback` for `HuggingFaceH4/ultrafeedback_binarized`. Adapt `scripts/prepare_code_data.py` accordingly.

### Tokenizer, pack, train

Same commands as code recipe. Use `--seq-len 2048` when packing. Use `configs/regent_370m.yaml` as the starting config with `max_seq_len: 2048` applied.

---

## Expected trajectory

No reference chat run executed on this repo. Rough expectations from standard scaling:

| Config | Data | Val PPL | Wall-clock |
|---|---|---|---|
| 1.5B | 30B tokens | ~8 | ~3 weeks on 1× H100 |
| 7B | 300B tokens | ~4–5 | 3–4 weeks on 8× H100 |
| 70B | 1.5T tokens | ~2.5–3 | months on cluster |

Perplexity is not the full picture. Add MT-Bench, AlpacaEval 2, Arena-Hard, IFEval after Phase 2.

---

## Pitfalls

| Symptom | Fix |
|---|---|
| Repetitive outputs | Add diverse conversational corpora to Phase 2 |
| Ignores role boundaries | Increase Phase 2 data or add role markers to Phase 1 mix |
| Overrefusal | Use HelpSteer 2 or UltraFeedback instead of HH-RLHF, or rebalance HH subset |
| Long-context recall failure | Raise `ssm.d_state` (256+ for validation, 1024 for 7B+) |
| Phase 2 overfits fast | Combine UltraChat + Tulu + OpenAssistant for ≥500K conversations |

Plus the general Mamba pitfalls in [code-model.md](code-model.md#pitfalls).

---

## Scaling

| Target | Config | Data | Compute |
|---|---|---|---|
| 370M | `configs/regent_370m.yaml` w/ `max_seq_len: 2048` | 5–10B tokens + 200K SFT + 64K DPO | 1× H100 for ~2 weeks |
| 1.5B | `configs/regent_1_5b_edge.yaml` | 30B + full SFT/DPO | 1× H100 for 3–4 weeks or 4× A100 for ~2 weeks |
| 7B | `configs/regent_7b.yaml` | 300B + full SFT/DPO | 8× H100 for 3–4 weeks |
| 70B | new config (not in repo) | 1.5T+ | cluster, months — DSTP or Prime Intellect path |

---

## Open tasks

- `scripts/prepare_chat_data.py` mirroring `prepare_code_data.py`
- `configs/regent_chat_*.yaml` with chat-tuned defaults
- Evaluation harness (MT-Bench, AlpacaEval 2, Arena-Hard, IFEval)
- First reference run with an observed trajectory

---

## References

- Tulu 3: https://allenai.org/blog/tulu-3
- FineWeb-Edu: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- UltraFeedback: https://arxiv.org/abs/2310.01377
- HelpSteer 2: https://arxiv.org/abs/2406.08673
