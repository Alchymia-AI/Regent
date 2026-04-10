#!/usr/bin/env python3
"""
Export a trained Regent checkpoint to deployable formats.

HuggingFace package (--format hf or both):
    <output_dir>/
        config.json                 # PretrainedConfig-compatible fields
        configuration_regent.py    # subclass — upload alongside weights for trust_remote_code
        model.safetensors           # weight file (copied verbatim)
        tokenizer_config.json
        special_tokens_map.json
        generation_config.json
        README.md                   # auto-generated model card

vLLM / Docker package (--format vllm or both):
    <output_dir>/
        Dockerfile
        docker-compose.yml
        start.sh                    # convenience wrapper

Usage
-----
    python -m scripts.export_model \\
        --checkpoint checkpoints/alignment/regent.safetensors \\
        --config     configs/regent_7b.yaml \\
        --tokenizer  tokenizer/regent.model \\
        --output     export/regent-7b \\
        --name       "my-regent-7b" \\
        --format     both

Hub push (optional — requires huggingface_hub installed and HF_TOKEN env var):
    python -m scripts.export_model ... --hf-repo username/regent-7b
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(msg, flush=True)


def _require_file(path: str, label: str) -> Path:
    p = Path(path)
    if not p.exists():
        print(f"ERROR: {label} not found: {path}", file=sys.stderr)
        sys.exit(1)
    return p


def _write(dest: Path, name: str, content: str) -> None:
    (dest / name).write_text(content, encoding="utf-8")
    _log(f"  wrote {name}")


# ---------------------------------------------------------------------------
# HuggingFace export
# ---------------------------------------------------------------------------

def _convert_weights(src: Path, dest: Path, dtype: str) -> None:
    """
    Load safetensors from src, convert to dtype, save to dest.
    Falls back to plain copy if safetensors or torch is not installed.
    """
    if dtype == "float32":
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        return
    try:
        from safetensors.torch import load_file, save_file
        import torch
        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
        _log(f"  converting weights → {dtype} …")
        tensors = load_file(str(src), device="cpu")
        tensors = {k: v.to(dtype_map[dtype]) for k, v in tensors.items()}
        save_file(tensors, str(dest))
        _log(f"  saved {dest.stat().st_size / 1e6:.1f} MB ({dtype})")
    except ImportError:
        _log("  WARN: safetensors/torch not found — copying weights as-is (float32)")
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)


def export_hf(
    checkpoint: Path,
    config,   # RegentConfig instance
    tokenizer_path: Path | None,
    out: Path,
    name: str,
    description: str,
    license_id: str,
    tags: list[str],
    dtype: str = "float32",     # float32 | float16 | bfloat16
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    _log(f"\n[HF] Writing package → {out}  (dtype={dtype})")

    has_tokenizer = tokenizer_path is not None

    # --- config.json ---
    auto_map = {
        "AutoConfig":           "configuration_regent.RegentConfig",
        "AutoModelForCausalLM": "modeling_regent.RegentForCausalLM",
    }
    if has_tokenizer:
        auto_map["AutoTokenizer"] = "tokenization_regent.RegentTokenizer"

    cfg_dict = {
        "model_type": "regent",
        "architectures": ["RegentForCausalLM"],
        "torch_dtype": dtype,
        "auto_map": auto_map,
        # Core dims
        "d_model":      config.d_model,
        "n_layer":      config.n_layer,
        "vocab_size":   config.vocab_size,
        # Mamba-2 SSM
        "ssm_expand":       config.ssm_expand,
        "ssm_d_state":      config.ssm_d_state,
        "ssm_d_conv":       config.ssm_d_conv,
        "ssm_n_heads":      config.ssm_n_heads,
        "ssm_chunk_size":   config.ssm_chunk_size,
        # Attention
        "attn_layers":          list(config.attn_layers),
        "attn_n_q_heads":       config.attn_n_q_heads,
        "attn_n_kv_heads":      config.attn_n_kv_heads,
        "attn_head_dim":        config.attn_head_dim,
        "attn_window_size":     config.attn_window_size,
        # Ver head
        "ver_enabled":          config.ver_enabled,
        "ver_hidden_dim":       config.ver_hidden_dim,
        # EPG encoder
        "epg_max_nodes":            config.epg_max_nodes,
        "epg_scalar_features":      config.epg_scalar_features,
        "epg_n_categories":         config.epg_n_categories,
        "epg_category_embed_dim":   config.epg_category_embed_dim,
        "epg_n_encoder_layers":     config.epg_n_encoder_layers,
        "epg_encoder_heads":        config.epg_encoder_heads,
        # Essence
        "essence_input_dim":        config.essence_input_dim,
        "essence_inject_every_n":   config.essence_inject_every_n,
        # Misc
        "tie_embeddings":           config.tie_embeddings,
        "norm_eps":                 config.norm_eps,
    }
    _write(out, "config.json", json.dumps(cfg_dict, indent=2))

    # --- modeling_regent.py (PyTorch model — trust_remote_code) ---
    _modeling_src = Path(__file__).parent / "modeling_regent.py"
    if _modeling_src.exists():
        shutil.copy2(_modeling_src, out / "modeling_regent.py")
        _log("  copied modeling_regent.py")
    else:
        _log("  WARN: scripts/modeling_regent.py not found — skipping")

    # --- configuration_regent.py ---
    _write(out, "configuration_regent.py", _REGENT_CONFIG_PY.format(
        d_model=config.d_model,
        n_layer=config.n_layer,
        vocab_size=config.vocab_size,
    ))

    # --- generation_config.json ---
    gen_cfg = {
        "bos_token_id": 1,
        "eos_token_id": 2,
        "max_new_tokens": 512,
        "temperature": 0.85,
        "top_p": 0.9,
    }
    _write(out, "generation_config.json", json.dumps(gen_cfg, indent=2))

    # --- tokenizer files ---
    if has_tokenizer:
        # Copy SPM model file
        tok_dest = out / "regent.model"
        if tokenizer_path.resolve() != tok_dest.resolve():
            shutil.copy2(tokenizer_path, tok_dest)
        _log(f"  copied regent.model")
        # Copy tokenization_regent.py for trust_remote_code
        _tok_src = Path(__file__).parent / "tokenization_regent.py"
        if _tok_src.exists():
            shutil.copy2(_tok_src, out / "tokenization_regent.py")
            _log("  copied tokenization_regent.py")
        else:
            _log("  WARN: scripts/tokenization_regent.py not found — skipping")

    tok_cfg = {
        "tokenizer_class":              "RegentTokenizer" if has_tokenizer else "PreTrainedTokenizer",
        "auto_map":                     {"AutoTokenizer": "tokenization_regent.RegentTokenizer"} if has_tokenizer else {},
        "vocab_file":                   "regent.model" if has_tokenizer else "",
        "bos_token":                    "[BOS]",
        "eos_token":                    "[EOS]",
        "unk_token":                    "[UNK]",
        "pad_token":                    "[PAD]",
        "add_bos_token":                True,
        "add_eos_token":                False,
        "model_max_length":             131072,
        "clean_up_tokenization_spaces": False,
    }
    _write(out, "tokenizer_config.json", json.dumps(tok_cfg, indent=2))
    special = {
        "bos_token": {"content": "[BOS]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False},
        "eos_token": {"content": "[EOS]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False},
        "unk_token": {"content": "[UNK]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False},
        "pad_token": {"content": "[PAD]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False},
    }
    _write(out, "special_tokens_map.json", json.dumps(special, indent=2))

    # --- weights (with optional dtype conversion) ---
    ckpt_dest = out / "model.safetensors"
    _convert_weights(checkpoint, ckpt_dest, dtype)

    # --- model card ---
    param_millions = _estimate_params_m(config)
    _write(out, "README.md", _model_card(
        name=name,
        description=description,
        license_id=license_id,
        tags=tags,
        param_millions=param_millions,
        config=config,
    ))

    _log(
        f"[HF] Done.\n"
        f"  PyTorch:  AutoModelForCausalLM.from_pretrained('{out}', trust_remote_code=True)\n"
        f"  MLX:      model.load_weights(list(mx.load('{out}/model.safetensors').items()))"
    )


# ---------------------------------------------------------------------------
# vLLM / Docker export
# ---------------------------------------------------------------------------

def export_vllm(
    checkpoint: Path,
    config,
    out: Path,
    name: str,
    server_port: int = 8400,
    api_port:    int = 8000,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    _log(f"\n[vLLM/Docker] Writing package → {out}")

    root_rel = os.path.relpath(
        Path(__file__).resolve().parent.parent,
        out.resolve(),
    )

    _write(out, "Dockerfile", _dockerfile(name))
    _write(out, "docker-compose.yml", _docker_compose(name, server_port, api_port))
    _write(out, "start.sh", _start_sh(checkpoint, config, server_port))
    (out / "start.sh").chmod(0o755)

    _log(f"[Docker] Build & run:\n"
         f"    docker compose -f {out}/docker-compose.yml up --build")


# ---------------------------------------------------------------------------
# HuggingFace Hub push
# ---------------------------------------------------------------------------

def push_to_hub(out: Path, repo_id: str, token: str | None) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        _log("ERROR: pip install huggingface_hub  to enable hub push")
        return

    _log(f"\n[Hub] Uploading {out} → {repo_id} …")
    api = HfApi()
    api.upload_folder(
        folder_path=str(out),
        repo_id=repo_id,
        repo_type="model",
        token=token or os.environ.get("HF_TOKEN"),
    )
    _log(f"[Hub] Done: https://huggingface.co/{repo_id}")


# ---------------------------------------------------------------------------
# Param count estimation (no model instantiation needed)
# ---------------------------------------------------------------------------

def _estimate_params_m(config) -> float:
    d = config.d_model
    v = config.vocab_size
    n = config.n_layer
    expand = config.ssm_expand
    d_inner = d * expand
    n_attn = len(config.attn_layers)
    n_ssm  = n - n_attn

    embed  = d * v
    ssm_block = (
        d * d_inner * 3 +           # in_proj (z, x, dt parts)
        d_inner * config.ssm_d_conv + # conv1d weight
        d_inner * config.ssm_d_conv + # conv1d bias
        config.ssm_n_heads +          # A_log, D, dt_bias
        config.ssm_n_heads * 2 +
        d_inner * d                   # out_proj
    )
    q_dim  = config.attn_n_q_heads  * config.attn_head_dim
    kv_dim = config.attn_n_kv_heads * config.attn_head_dim
    attn_block = d * (q_dim + kv_dim * 2 + d)

    total = embed + n_ssm * ssm_block + n_attn * attn_block
    return round(total / 1e6, 1)


# ---------------------------------------------------------------------------
# Inline templates
# ---------------------------------------------------------------------------

_REGENT_CONFIG_PY = '''\
"""
Regent PretrainedConfig — upload alongside model.safetensors for trust_remote_code.
"""
from transformers import PretrainedConfig


class RegentConfig(PretrainedConfig):
    model_type = "regent"

    def __init__(
        self,
        d_model={d_model},
        n_layer={n_layer},
        vocab_size={vocab_size},
        ssm_expand=2,
        ssm_d_state=64,
        ssm_d_conv=4,
        ssm_n_heads=16,
        ssm_chunk_size=64,
        attn_layers=(),
        attn_n_q_heads=16,
        attn_n_kv_heads=4,
        attn_head_dim=64,
        attn_window_size=1024,
        ver_enabled=True,
        ver_hidden_dim=128,
        epg_max_nodes=32,
        epg_scalar_features=5,
        epg_n_categories=15,
        epg_category_embed_dim=8,
        epg_n_encoder_layers=2,
        epg_encoder_heads=4,
        essence_input_dim=7,
        essence_inject_every_n=8,
        tie_embeddings=True,
        norm_eps=1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model             = d_model
        self.n_layer             = n_layer
        self.vocab_size          = vocab_size
        self.ssm_expand          = ssm_expand
        self.ssm_d_state         = ssm_d_state
        self.ssm_d_conv          = ssm_d_conv
        self.ssm_n_heads         = ssm_n_heads
        self.ssm_chunk_size      = ssm_chunk_size
        self.attn_layers         = list(attn_layers)
        self.attn_n_q_heads      = attn_n_q_heads
        self.attn_n_kv_heads     = attn_n_kv_heads
        self.attn_head_dim       = attn_head_dim
        self.attn_window_size    = attn_window_size
        self.ver_enabled         = ver_enabled
        self.ver_hidden_dim      = ver_hidden_dim
        self.epg_max_nodes       = epg_max_nodes
        self.epg_scalar_features = epg_scalar_features
        self.epg_n_categories    = epg_n_categories
        self.epg_category_embed_dim = epg_category_embed_dim
        self.epg_n_encoder_layers   = epg_n_encoder_layers
        self.epg_encoder_heads   = epg_encoder_heads
        self.essence_input_dim   = essence_input_dim
        self.essence_inject_every_n = essence_inject_every_n
        self.tie_embeddings      = tie_embeddings
        self.norm_eps            = norm_eps
'''


def _model_card(
    name: str,
    description: str,
    license_id: str,
    tags: list[str],
    param_millions: float,
    config,
) -> str:
    tag_block = "\n".join(f"- {t}" for t in tags)
    attn_str  = ", ".join(str(x) for x in config.attn_layers) or "none"
    return f"""\
---
license: {license_id}
tags:
{tag_block}
---

# {name}

{description or "Regent hybrid Mamba-2 / GQA language model with EPG grounding and Essence conditioning."}

## Architecture

| Param | Value |
|---|---|
| Total params | ~{param_millions}M |
| d_model | {config.d_model} |
| n_layer | {config.n_layer} |
| vocab_size | {config.vocab_size} |
| SSM expand | {config.ssm_expand}× |
| SSM d_state | {config.ssm_d_state} |
| Attn layers | {attn_str} |
| Ver head | {"enabled" if config.ver_enabled else "disabled"} |
| EPG max nodes | {config.epg_max_nodes} |
| Essence dim | {config.essence_input_dim} |

Regent is a hybrid SSM/Attention model combining Mamba-2 recurrent blocks with sparse GQA
attention layers. It includes:

- **EPG Encoder** — Entity Preference Graph nodes injected as dense prefix embeddings
- **Essence Conditioning** — 7-dim behavioural vector injected additively every N layers
- **Verification Head** — per-token grounding confidence [0, 1]; HALT-and-retrieve on low confidence

## Usage — HuggingFace Transformers (PyTorch)

```python
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "path/to/export",
    trust_remote_code=True,
    torch_dtype=torch.float32,
)
# Basic generation (no EPG/Essence):
inputs = torch.tensor([[1, 2, 3]])   # your token IDs
out    = model.generate(inputs, max_new_tokens=200)
```

## Usage — MLX (Apple Silicon, native)

```python
import json, mlx.core as mx
from regent_model.layers.model import RegentModel, RegentConfig

cfg   = RegentConfig(**json.load(open("config.json")))
model = RegentModel(cfg)
model.load_weights(list(mx.load("model.safetensors").items()))
mx.eval(model.parameters())
```

> Weight keys are identical in both frameworks — the same `model.safetensors` file
> is read by both PyTorch (`safetensors.torch`) and MLX (`mlx.core.load`).

## Licence

{license_id}
"""


def _dockerfile(name: str) -> str:
    return f"""\
# Regent model server — Docker image
# Build:  docker build -t {name} .
# Run:    docker run -p 8400:8400 {name}

FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential git curl && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

EXPOSE 8400

CMD ["python", "-m", "serve.server", \\
     "--config",    "configs/regent_7b.yaml", \\
     "--model",     "model.safetensors", \\
     "--host",      "0.0.0.0", \\
     "--port",      "8400"]
"""


def _docker_compose(name: str, server_port: int, api_port: int) -> str:
    return f"""\
version: "3.9"
services:
  regent:
    build: .
    image: {name}:latest
    ports:
      - "{server_port}:{server_port}"
    environment:
      - PYTHONUNBUFFERED=1
    volumes:
      - ./checkpoints:/app/checkpoints
      - ./data:/app/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-sf", "http://localhost:{server_port}/health"]
      interval: 15s
      timeout: 5s
      retries: 3
"""


def _start_sh(checkpoint: Path, config, port: int) -> str:
    return f"""\
#!/usr/bin/env bash
# Auto-generated start script for Regent model server
set -e
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

exec python -m serve.server \\
    --config   "$REPO_ROOT/configs/regent_7b.yaml" \\
    --model    "{checkpoint.resolve()}" \\
    --host     0.0.0.0 \\
    --port     {port}
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Regent checkpoint to HuggingFace / vLLM-Docker format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--checkpoint", required=True, help="Path to .safetensors weight file")
    parser.add_argument("--config",     required=True, help="Path to model YAML config")
    parser.add_argument("--tokenizer",  default=None,  help="Path to .model SentencePiece tokenizer")
    parser.add_argument("--output",     default="export", help="Output directory")
    parser.add_argument("--name",       default="regent", help="Model name (used in filenames, model card)")
    parser.add_argument("--description", default="", help="Short description for model card")
    parser.add_argument("--license",    default="apache-2.0", help="SPDX license identifier")
    parser.add_argument("--tags",       default="mamba,ssm,regent", help="Comma-separated HF tags")
    parser.add_argument("--format",     default="hf",
                        choices=["hf", "vllm", "both"],
                        help="Export format (default: hf)")
    parser.add_argument("--dtype",      default="float32",
                        choices=["float32", "float16", "bfloat16"],
                        help="Weight dtype for HF export (default: float32)")
    parser.add_argument("--hf-repo",    default=None, help="HuggingFace repo id to push to, e.g. user/model-name")
    parser.add_argument("--hf-token",   default=None, help="HuggingFace API token (or set HF_TOKEN env var)")
    parser.add_argument("--port",       type=int, default=8400, help="Server port for Docker/vLLM config")

    args = parser.parse_args()

    checkpoint = _require_file(args.checkpoint, "checkpoint")
    config_path = _require_file(args.config, "config YAML")
    tokenizer   = Path(args.tokenizer) if args.tokenizer else None
    if tokenizer and not tokenizer.exists():
        print(f"WARN: tokenizer not found at {tokenizer} — skipping", file=sys.stderr)
        tokenizer = None

    # Load RegentConfig from repo
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from regent_model.layers.model import RegentConfig
    config = RegentConfig.from_yaml(str(config_path))

    out = Path(args.output)
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    fmt = args.format
    if fmt in ("hf", "both"):
        export_hf(
            checkpoint=checkpoint,
            config=config,
            tokenizer_path=tokenizer,
            out=out,
            name=args.name,
            description=args.description,
            license_id=args.license,
            tags=tags,
            dtype=args.dtype,
        )

    if fmt in ("vllm", "both"):
        export_vllm(
            checkpoint=checkpoint,
            config=config,
            out=out,
            name=args.name,
            server_port=args.port,
        )

    if args.hf_repo:
        push_to_hub(out, args.hf_repo, args.hf_token)

    _log("\nExport complete.")


if __name__ == "__main__":
    main()
