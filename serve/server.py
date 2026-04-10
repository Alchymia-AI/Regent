"""
Standalone HTTP inference server for the Regent model.

Exposes a REST API that any client (including Regent Core's brain.service.ts)
can call. No coupling to the Regent codebase.

Endpoints:
    POST /generate          — generate text with verification-gated decoding
    POST /verify            — score existing text for grounding
    GET  /health            — health check
    GET  /info              — model info and parameter count

Usage:
    python -m serve.server --model checkpoints/regent-370m.safetensors --config configs/regent_370m.yaml
"""

import argparse
import json
import time

import mlx.core as mx
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from regent_model.layers.model import RegentModel, RegentConfig
from serve.generate import generate, GenerateConfig, GenerateResult


app = FastAPI(title="Regent Model Server", version="0.1.0")

# Global state (set during startup)
_model: RegentModel | None = None
_tokenizer = None
_config: RegentConfig | None = None


# --- Request/Response models ---

class EPGNode(BaseModel):
    key: str
    value: str
    confidence: float = 0.5
    activation: float = 0.5
    valence: float = 0.0
    emotional_weight: float = 0.5
    category: str = "domain"


class EssenceState(BaseModel):
    essence_index: float = 5.0
    essence_influence: float = 0.0
    truth_vs_lie: float = 0.0
    civility_vs_unruliness: float = 0.0
    good_vs_evil: float = 0.0
    curiosity: float = 0.5
    self_preservation: float = 0.3


class GenerateRequest(BaseModel):
    messages: list[dict]
    epg_nodes: list[EPGNode] | None = None
    essence: EssenceState | None = None
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    verification: bool = True
    grounding_threshold: float = 0.4


class GenerateResponse(BaseModel):
    text: str
    grounding_scores: list[float]
    halted_positions: list[int]
    total_tokens: int
    inference_time_ms: float


class VerifyRequest(BaseModel):
    text: str
    epg_nodes: list[EPGNode] | None = None
    essence: EssenceState | None = None


class VerifyResponse(BaseModel):
    grounding_scores: list[float]
    mean_grounding: float
    flagged_spans: list[dict]


# --- Endpoints ---

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/info")
async def info():
    if _model is None:
        return {"error": "model not loaded"}

    params = _model.count_parameters()
    return {
        "model": {
            "d_model": _config.d_model,
            "n_layer": _config.n_layer,
            "vocab_size": _config.vocab_size,
            "ssm_d_state": _config.ssm_d_state,
            "ssm_n_heads": _config.ssm_n_heads,
            "attn_layers": list(_config.attn_layers),
            "ver_enabled": _config.ver_enabled,
        },
        "parameters": params,
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate_endpoint(req: GenerateRequest):
    if _model is None or _tokenizer is None:
        return GenerateResponse(
            text="Error: model not loaded",
            grounding_scores=[],
            halted_positions=[],
            total_tokens=0,
            inference_time_ms=0,
        )

    start = time.time()

    # Tokenize messages into a flat prompt
    prompt = ""
    for msg in req.messages:
        prompt += f"<{msg['role']}>{msg['content']}"
    token_ids = _tokenizer.encode(prompt, add_bos=True)
    input_ids = mx.array([token_ids], dtype=mx.int32)

    # Encode essence
    essence = None
    if req.essence:
        e = req.essence
        essence = mx.array([[
            e.essence_index, e.essence_influence, e.truth_vs_lie,
            e.civility_vs_unruliness, e.good_vs_evil,
            e.curiosity, e.self_preservation,
        ]])

    # Encode EPG nodes (simplified — full encoding needs tokenizer)
    epg_node_tokens = None
    epg_scalars = None
    epg_categories = None
    # TODO: implement full EPG encoding for server
    # For now, EPG context is passed in the prompt text

    gen_cfg = GenerateConfig(
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        caution_threshold=req.grounding_threshold,
    )

    result = generate(
        model=_model,
        input_ids=input_ids,
        tokenizer=_tokenizer,
        config=gen_cfg,
        essence=essence,
        epg_node_tokens=epg_node_tokens,
        epg_scalars=epg_scalars,
        epg_categories=epg_categories,
    )

    elapsed = (time.time() - start) * 1000

    return GenerateResponse(
        text=result.text,
        grounding_scores=result.grounding_scores,
        halted_positions=result.halted_positions,
        total_tokens=result.total_tokens,
        inference_time_ms=round(elapsed, 1),
    )


@app.post("/verify", response_model=VerifyResponse)
async def verify_endpoint(req: VerifyRequest):
    """Score existing text for grounding without generating new text."""
    if _model is None or _tokenizer is None:
        return VerifyResponse(grounding_scores=[], mean_grounding=0.0, flagged_spans=[])

    token_ids = _tokenizer.encode(req.text, add_bos=True)
    input_ids = mx.array([token_ids], dtype=mx.int32)

    essence = None
    if req.essence:
        e = req.essence
        essence = mx.array([[
            e.essence_index, e.essence_influence, e.truth_vs_lie,
            e.civility_vs_unruliness, e.good_vs_evil,
            e.curiosity, e.self_preservation,
        ]])

    output = _model(input_ids=input_ids, essence=essence, use_chunked=False)

    scores = []
    if output.get("grounding") is not None:
        scores = output["grounding"][0].tolist()

    mean_g = sum(scores) / len(scores) if scores else 0.0

    # Find spans where grounding drops below threshold
    flagged = []
    in_flag = False
    start_idx = 0
    threshold = 0.4
    for i, s in enumerate(scores):
        if s < threshold and not in_flag:
            in_flag = True
            start_idx = i
        elif s >= threshold and in_flag:
            in_flag = False
            span_tokens = token_ids[start_idx:i]
            flagged.append({
                "start": start_idx,
                "end": i,
                "text": _tokenizer.decode(span_tokens),
                "min_score": min(scores[start_idx:i]),
            })
    if in_flag:
        span_tokens = token_ids[start_idx:]
        flagged.append({
            "start": start_idx,
            "end": len(scores),
            "text": _tokenizer.decode(span_tokens),
            "min_score": min(scores[start_idx:]),
        })

    return VerifyResponse(
        grounding_scores=scores,
        mean_grounding=round(mean_g, 3),
        flagged_spans=flagged,
    )


def load_model(config_path: str, weights_path: str | None = None, tokenizer_path: str | None = None):
    """Load model, config, and tokenizer into global state."""
    global _model, _config, _tokenizer

    _config = RegentConfig.from_yaml(config_path)
    _model = RegentModel(_config)

    if weights_path:
        weights = mx.load(weights_path)
        _model.load_weights(list(weights.items()))

    if tokenizer_path:
        from regent_model.utils.tokenizer import RegentTokenizer
        _tokenizer = RegentTokenizer(tokenizer_path)
        print(f"Tokenizer loaded: vocab={_tokenizer.vocab_size}")

    params = _model.count_parameters()
    print(f"Model loaded: {params['total_millions']}M parameters")


def main():
    parser = argparse.ArgumentParser(description="Regent Model Server")
    parser.add_argument("--config", required=True, help="Model config YAML")
    parser.add_argument("--model", default=None, help="Model weights (safetensors)")
    parser.add_argument("--tokenizer", default=None, help="SentencePiece tokenizer (.model)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8400)
    args = parser.parse_args()

    load_model(args.config, args.model, args.tokenizer)
    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
