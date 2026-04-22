# Quickstart: Zero to Trained Model

This covers the full path from a fresh clone to a deployed checkpoint. No steps are omitted.

---

## Prerequisites

- Python 3.11 or 3.12
- One of: Apple Silicon Mac (MLX), NVIDIA GPU with CUDA 12.4+, or CPU (slow, only viable for the test config)
- For 7B training: at minimum one A100 80GB or equivalent. Multi-GPU recommended.
- Git, Node.js 18+ (for the UI only)

---

## 1. Clone and install

```bash
git clone https://github.com/Alchymia-AI/Regent
cd regent-model
python3.12 -m venv .venv
source .venv/bin/activate
```

**Apple Silicon**

```bash
pip install mlx pyyaml sentencepiece numpy fastapi uvicorn pydantic safetensors
```

**NVIDIA GPU**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install pyyaml sentencepiece numpy fastapi uvicorn pydantic safetensors transformers
# Optional: Triton SSD kernel for faster long-sequence training
pip install mamba-ssm causal-conv1d
```

---

## 2. Validate the architecture

Before touching any data, confirm the model builds and gradients flow correctly.

```bash
PYTHONPATH=. python3 tests/test_architecture.py
```

All tests should pass. If any fail, there is an environment issue. Fix it before continuing.

---

## 3. Choose your config

| Config | Parameters | Use case |
|---|---|---|
| `configs/regent_test.yaml` | ~10M | CI, architecture validation |
| `configs/regent_370m.yaml` | 370M | Prototype, Apple Silicon, low-VRAM GPU |
| `configs/regent_1_5b_edge.yaml` | 1.5B | Edge deployment target |
| `configs/regent_7b.yaml` | 7B | Production open-source tier |

Start with `regent_370m.yaml` to validate the full pipeline before committing GPU time to 7B.

---

## 4. Prepare training data

The pipeline expects data at `data/raw/train.txt` and `data/raw/val.txt`. One document per line. UTF-8.

**Option A: Use the pipeline scraper**

Edit `pipeline.yaml` to configure your sources. The file has examples for local directories, web URLs, and HuggingFace datasets.

```yaml
# pipeline.yaml
sources:
  - type: local
    path: data/raw/custom/    # drop your .txt files here
    max_docs: 100000

  - type: huggingface
    dataset: wikimedia/wikipedia
    split: train
    column: text
    max_docs: 200000
```

Then run:

```bash
pip install datasets beautifulsoup4   # required for HF datasets and URL scraping
PYTHONPATH=. python3 scripts/scrape_corpus.py --config pipeline.yaml
```

Output: `data/raw/train.txt` and `data/raw/val.txt`

**Option B: Bring your own data**

Place pre-prepared text files directly:

```
data/raw/train.txt   # one document per line
data/raw/val.txt
```

Minimum recommended size for Phase 1: 1GB of text. More is better. Quality matters more than quantity.

**Phase 2 data (domain fine-tuning)**

Phase 2 expects conversational data in the format:

```
<regent_meta>{"session_id": "...", "turn": 0}</regent_meta>
User: <question>
Regent: <answer>
```

Place this at `data/phase2/train.txt`. If you do not have this yet, the pipeline will warn and skip Phase 2. You can run Phase 2 separately later with `--start-stage 5`.

**Phase 3 data (verification head)**

Phase 3 requires labeled grounding pairs. Each line is a JSON object:

```json
{"text": "Aspirin reduces fever.", "label": 1, "source": "pharmacology_node_id"}
{"text": "Aspirin cures diabetes.", "label": 0, "source": null}
```

- `label: 1` = grounded claim (traceable to a knowledge node)
- `label: 0` = fabricated or contradicted claim

Place this at `data/phase3/grounding_pairs.jsonl`. This dataset must be domain-specific. A general corpus will not produce a calibrated verification head. If you do not have this, skip Phase 3 for now.

**Phase 4 data (alignment)**

Phase 4 expects preference pairs:

```json
{"prompt": "...", "chosen": "...", "rejected": "..."}
```

Place at `data/phase4/preferences.jsonl`. Optional. Skip if not available.

---

## 5. Run the full pipeline

Once data is in place:

```bash
PYTHONPATH=. python3 scripts/run_pipeline.py \
    --config configs/regent_7b.yaml \
    --scrape-config pipeline.yaml
```

The pipeline runs 7 stages in sequence:

| Stage | What it does | Output |
|---|---|---|
| 1. Scrape | Collects and cleans corpus | `data/raw/train.txt`, `val.txt` |
| 2. Tokenize | Trains BPE tokenizer on corpus | `data/tokenizer/regent.model` |
| 3. Prepare | Packs tokens into arrays | `data/processed/train.npy`, `val.npy` |
| 4. Phase 1 | Base pre-training | `checkpoints/phase1/` |
| 5. Phase 2 | Domain fine-tuning | `checkpoints/phase2/` |
| 6. Phase 3 | Verification head training | `checkpoints/phase3/` |
| 7. Phase 4 | DPO alignment | `checkpoints/alignment/` |

Each stage is idempotent. If a stage's output already exists, it skips. To re-run a specific stage:

```bash
# Resume from Phase 3 onward
PYTHONPATH=. python3 scripts/run_pipeline.py \
    --config configs/regent_7b.yaml \
    --start-stage 6

# Force re-run a stage even if output exists
PYTHONPATH=. python3 scripts/run_pipeline.py \
    --config configs/regent_7b.yaml \
    --start-stage 4 \
    --force-stage 4
```

---

## 6. Smoke-test the pipeline on synthetic data

Before running on real data, validate that all four training phases run end-to-end on your hardware:

```bash
PYTHONPATH=. python3 scripts/run_pipeline.py \
    --config configs/regent_test.yaml \
    --synthetic
```

This uses a tiny model (~10M parameters) and generated data. Completes in minutes on any hardware. If this fails, fix the environment before proceeding.

---

## 7. Monitor training

Each phase writes checkpoints to `checkpoints/phase{N}/` as it trains. To check loss:

```bash
# Tail the training log
tail -f logs/train.log

# List checkpoints for a phase
ls -lh checkpoints/phase1/
```

Checkpoints are named `step_N.safetensors`. The pipeline automatically resumes from the latest checkpoint if interrupted.

To resume a specific phase from a specific checkpoint:

```bash
PYTHONPATH=. python3 scripts/train.py \
    --config configs/regent_7b.yaml \
    --resume checkpoints/phase1/step_50000.safetensors
```

---

## 8. Run inference against a checkpoint

To test a checkpoint before training completes:

```bash
./start.sh \
    --config configs/regent_7b.yaml \
    --model checkpoints/phase1/step_50000.safetensors
```

Model Studio UI opens at `http://localhost:3000`. The Inference page shows per-token grounding scores in real time.

API directly:

```bash
curl -X POST http://localhost:8400/generate \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "max_tokens": 256,
        "verification": true,
        "grounding_threshold": 0.4
    }'
```

The response includes `grounding_scores` (per token, 0 to 1) and `halt_positions` (token indices where the model stopped and retrieved).

---

## 9. Export the final model

After Phase 4 completes, export the alignment checkpoint to HuggingFace or Docker format:

**HuggingFace**

```bash
python3 -m scripts.export_model \
    --checkpoint checkpoints/alignment/regent.safetensors \
    --config configs/regent_7b.yaml \
    --tokenizer data/tokenizer/regent.model \
    --output export/regent-7b \
    --name "regent-7b" \
    --format hf \
    --dtype float16
```

Load it:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "export/regent-7b", trust_remote_code=True
)
tok = AutoTokenizer.from_pretrained(
    "export/regent-7b", trust_remote_code=True
)
```

**Push to HuggingFace Hub**

```bash
export HF_TOKEN=your_token_here

python3 -m scripts.export_model \
    --checkpoint checkpoints/alignment/regent.safetensors \
    --config configs/regent_7b.yaml \
    --tokenizer data/tokenizer/regent.model \
    --output export/regent-7b \
    --name "regent-7b" \
    --format hf \
    --dtype float16 \
    --hf-repo your-username/regent-7b
```

**Docker**

```bash
python3 -m scripts.export_model \
    --checkpoint checkpoints/alignment/regent.safetensors \
    --config configs/regent_7b.yaml \
    --tokenizer data/tokenizer/regent.model \
    --output export/regent-7b-docker \
    --name "regent-7b" \
    --format vllm \
    --dtype float16
```

Then:

```bash
cd export/regent-7b-docker
docker compose up
```

The server starts on port 8400 with the same API as the development server.

---

## 10. INT4 quantization for edge deployment

To reduce the 7B model's weight footprint, quantize to INT4 after export:

```bash
pip install auto-gptq optimum

python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

tokenizer = AutoTokenizer.from_pretrained('export/regent-7b', trust_remote_code=True)
gptq_config = GPTQConfig(bits=4, dataset='c4', tokenizer=tokenizer)

model = AutoModelForCausalLM.from_pretrained(
    'export/regent-7b',
    quantization_config=gptq_config,
    trust_remote_code=True,
    device_map='auto',
)
model.save_pretrained('export/regent-7b-int4')
tokenizer.save_pretrained('export/regent-7b-int4')
"
```

Expected weight size: ~4 GB for the 7B model at INT4. Requires ~5 GB additional for the 1 GB fixed state plus overhead. Runs on GPUs with 16 GB+ VRAM.

---

## Data checklist before starting

| Data | Path | Required for |
|---|---|---|
| General text corpus | `data/raw/train.txt` | Phase 1 |
| Validation split | `data/raw/val.txt` | Phase 1 |
| Domain conversations | `data/phase2/train.txt` | Phase 2 |
| Grounding pairs | `data/phase3/grounding_pairs.jsonl` | Phase 3 |
| Preference pairs | `data/phase4/preferences.jsonl` | Phase 4 |

Phase 1 is the prerequisite for everything. Phases 2, 3, and 4 can be skipped initially and run later with `--start-stage`.

---

## Compute estimates

| Config | Phase 1 GPU hours | Hardware |
|---|---|---|
| 370M | 100 to 500 | Single A100 or M2 Ultra |
| 1.5B | 500 to 2000 | Single A100 or multi-GPU |
| 7B | 10000 to 50000 | Multi-GPU (8x A100 recommended) |

These are estimates. Actual time depends on corpus size, batch size, and gradient accumulation settings. Phases 2, 3, and 4 are significantly shorter than Phase 1.

---

## Tool calling

Regent supports tool calling natively. The model emits a `[TOOL_CALL]` token when it decides to invoke a tool, collects the JSON argument block, and stops with `stop_reason: "tool_call"`. The caller executes the tool, posts the result back as a `tool_result` message, and resumes generation in the same session.

**Define tools in the request**

```json
POST /generate
{
  "messages": [
    {"role": "user", "content": "What is the weather in Lagos today?"}
  ],
  "tools": [
    {
      "name": "get_weather",
      "description": "Returns current weather for a given city.",
      "parameters": {
        "type": "object",
        "properties": {
          "city": {"type": "string"}
        },
        "required": ["city"]
      }
    }
  ],
  "session_id": "abc123"
}
```

**Response when the model calls a tool**

```json
{
  "text": "",
  "stop_reason": "tool_call",
  "tool_calls": [
    {"name": "get_weather", "arguments": {"city": "Lagos"}}
  ],
  "session_id": "abc123"
}
```

**Resume with the tool result**

Post back to the same session with a `tool_result` message. The session cache carries the model state forward — no re-encoding of the full conversation.

```json
POST /generate
{
  "messages": [
    {"role": "tool_result", "content": "{\"temperature\": 31, \"condition\": \"sunny\"}"}
  ],
  "session_id": "abc123"
}
```

**Response with the final answer**

```json
{
  "text": "The weather in Lagos today is 31°C and sunny.",
  "stop_reason": "eos",
  "tool_calls": [],
  "session_id": "abc123"
}
```

**Tool message roles**

| Role | Direction | Purpose |
|---|---|---|
| `tool_call` | Model to caller | Model-emitted tool invocation (injected into prompt on replay) |
| `tool_result` | Caller to model | Result of tool execution, injected before model continues |

**Training note**

The model learns to call tools from training data. To produce a model that reliably calls tools, include tool call examples in your Phase 2 fine-tuning data using the same format: `[TOOL_CALL]{"name": "...", "arguments": {...}}[TOOL_END]`. Without this, the model will not generate `[TOOL_CALL]` tokens regardless of what tools are passed in the request.

---

## Common issues

**`data/raw/train.txt not found`**
Either run the scraper with `--scrape-config pipeline.yaml` or place your own data at `data/raw/train.txt`.

**CUDA out of memory**
Reduce `batch_size` in the config YAML, or increase `gradient_accumulation` by the same factor to preserve the effective batch size.

**Tokenizer not found during training**
The tokenizer trains in Stage 2. If you skipped to `--start-stage 4`, make sure `data/tokenizer/regent.model` exists. If not, run Stage 2 first: `--start-stage 2 --force-stage 2`.

**Checkpoint not found on resume**
The pipeline resumes from the latest checkpoint in the phase directory. If the directory is empty, training starts from scratch for that phase.

**`trust_remote_code` warning on HuggingFace load**
This is expected. Regent uses a custom model class. Always load with `trust_remote_code=True`.
