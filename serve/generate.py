"""
Generation engine with Ver Head-gated decoding.

Three-zone strategy:
    FLOW    (grounding > 0.6): sample normally
    CAUTION (0.3 < grounding <= 0.6): lower temperature, conservative tokens
    HALT    (grounding <= 0.3): stop, invoke retrieve_epg_fn, restart with augmented EPG

Key fixes over the prototype:
    - Cache is maintained across token steps so Mamba state accumulates correctly.
    - HALT triggers an EPG retrieval callback and restarts generation if new context
      is available (limited to max_halt_retries to prevent loops).
    - token_texts is populated per-step for per-token UI alignment.
    - Returns (GenerateResult, final_cache) so callers can persist session state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

import mlx.core as mx

from regent_model.layers.model import RegentModel


@dataclass
class ToolCall:
    name: str
    arguments: dict


@dataclass
class GenerateConfig:
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1

    # Verification zone thresholds
    flow_threshold: float = 0.6
    caution_threshold: float = 0.3
    caution_temperature: float = 0.3

    # Max restarts triggered by HALT before falling back to caution sampling
    max_halt_retries: int = 1

    # Tool calling — special token IDs
    tool_call_id: int = 6
    tool_end_id: int = 8

    # Thinking — special token IDs
    think_start_id: int = 10
    think_end_id: int = 11


@dataclass
class GenerateResult:
    text: str
    tokens: list[int]
    token_texts: list[str]   # each token decoded individually — aligns with grounding_scores
    grounding_scores: list[float]
    halted_positions: list[int]
    total_tokens: int
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "eos"  # "eos" | "tool_call" | "max_tokens"


# retrieve_epg_fn signature:
#   input: (prompt_ids, generated_token_ids_so_far)
#   output: (epg_node_tokens, epg_scalars, epg_categories) or None if no new context
RetrieveEPGFn = Callable[
    [mx.array, list[int]],
    tuple[mx.array, mx.array, mx.array] | None,
]


def _get_grounding(output: dict) -> float:
    """Extract the last grounding score from model output, defaulting to 1.0."""
    g = output.get("grounding")
    return g[:, -1].item() if g is not None else 1.0


def _eval_cache(cache: list[dict]) -> None:
    """Force evaluation of all deferred arrays in the layer cache."""
    arrays = [
        v
        for layer_cache in cache
        for v in layer_cache.values()
        if isinstance(v, mx.array)
    ]
    if arrays:
        mx.eval(*arrays)


def sample_token(
    logits: mx.array,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
) -> mx.array:
    """Sample a single token from logits with temperature, top-k, and nucleus filtering."""
    if temperature <= 0:
        return mx.argmax(logits, axis=-1)

    logits = logits / temperature

    if top_k > 0:
        sorted_vals = mx.sort(logits, axis=-1)
        threshold = sorted_vals[:, -top_k : -top_k + 1]
        logits = mx.where(logits < threshold, mx.array(float("-inf")), logits)

    if top_p < 1.0:
        sorted_indices = mx.argsort(logits, axis=-1)
        sorted_logits = mx.take_along_axis(logits, sorted_indices, axis=-1)
        cumulative = mx.cumsum(mx.softmax(sorted_logits, axis=-1), axis=-1)
        # Mask tokens whose cumulative probability exceeds top_p (keep the nucleus)
        cutoff = cumulative - mx.softmax(sorted_logits, axis=-1)
        mask = cutoff > top_p
        sorted_logits = mx.where(mask, mx.array(float("-inf")), sorted_logits)
        # Unsort back to original vocabulary order
        unsort_indices = mx.argsort(sorted_indices, axis=-1)
        logits = mx.take_along_axis(sorted_logits, unsort_indices, axis=-1)

    return mx.random.categorical(logits)


def generate(
    model: RegentModel,
    input_ids: mx.array,
    tokenizer,
    config: GenerateConfig | None = None,
    essence: mx.array | None = None,
    epg_node_tokens: mx.array | None = None,
    epg_scalars: mx.array | None = None,
    epg_categories: mx.array | None = None,
    retrieve_epg_fn: RetrieveEPGFn | None = None,
    eos_id: int = 2,
    initial_cache: list[dict] | None = None,
    _halt_depth: int = 0,
) -> tuple[GenerateResult, list[dict]]:
    """
    Generate tokens with verification-gated decoding.

    Args:
        model:            the Regent model
        input_ids:        (1, seq_len) — tokenized prompt (or new tokens if initial_cache set)
        tokenizer:        for per-token decoding
        config:           generation hyperparameters
        essence:          (1, 7) — essence state vector
        epg_*:            EPG context tensors (optional)
        retrieve_epg_fn:  callback invoked on HALT — receives (prompt_ids, generated_so_far)
                          and returns augmented EPG tensors or None
        eos_id:           end-of-sequence token ID
        initial_cache:    restored layer cache from a previous session step
        _halt_depth:      internal recursion counter for restart limiting

    Returns:
        (GenerateResult, final_cache) — result + updated layer cache for session storage
    """
    cfg = config or GenerateConfig()

    # Initialise layer cache. Use the restored session cache if provided,
    # otherwise start fresh. HALT restarts always start fresh.
    if initial_cache is not None and _halt_depth == 0:
        cache = initial_cache
    else:
        cache = model.init_cache()

    # --- Initial forward pass on the full prompt (or new tokens + session state) ---
    output = model(
        input_ids=input_ids,
        essence=essence,
        epg_node_tokens=epg_node_tokens,
        epg_scalars=epg_scalars,
        epg_categories=epg_categories,
        cache=cache,
        use_chunked=False,
    )
    cache = output.get("cache") or cache
    _eval_cache(cache)

    generated_tokens: list[int] = []
    token_texts: list[str] = []
    grounding_scores: list[float] = []
    halted_positions: list[int] = []
    thinking_tokens: list[int] = []

    next_logits = output["logits"][:, -1:, :]   # (1, 1, vocab)
    next_grounding = _get_grounding(output)

    # --- Token-by-token generation loop ---
    for step in range(cfg.max_tokens):
        g_score = next_grounding

        if g_score > cfg.flow_threshold:
            temp = cfg.temperature

        elif g_score > cfg.caution_threshold:
            temp = cfg.caution_temperature

        else:
            # HALT zone
            halted_positions.append(step)

            if retrieve_epg_fn is not None and _halt_depth < cfg.max_halt_retries:
                augmented = retrieve_epg_fn(input_ids, generated_tokens)
                if augmented is not None:
                    new_node_tokens, new_scalars, new_categories = augmented
                    # Restart generation from the beginning with augmented EPG.
                    # We do not carry forward any partial result.
                    return generate(
                        model=model,
                        input_ids=input_ids,
                        tokenizer=tokenizer,
                        config=cfg,
                        essence=essence,
                        epg_node_tokens=new_node_tokens,
                        epg_scalars=new_scalars,
                        epg_categories=new_categories,
                        retrieve_epg_fn=retrieve_epg_fn,
                        eos_id=eos_id,
                        initial_cache=None,         # fresh start on restart
                        _halt_depth=_halt_depth + 1,
                    )

            # No retrieval available or retries exhausted — continue with caution
            temp = cfg.caution_temperature

        token = sample_token(
            next_logits[:, 0, :],
            temperature=temp,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
        )
        mx.eval(token)

        token_id = int(token.item())
        if token_id == eos_id:
            break

        # Thinking: model emits [THINK].
        # Advance the model on the [THINK] token, then collect reasoning tokens
        # until [/THINK] or EOS. Thinking tokens are not part of the visible output.
        if token_id == cfg.think_start_id:
            output = model(input_ids=token.reshape(1, 1), essence=essence, cache=cache, use_chunked=False)
            cache = output.get("cache") or cache
            _eval_cache(cache)
            next_logits = output["logits"]

            for _ in range(cfg.max_tokens):
                think_tok = sample_token(
                    next_logits[:, 0, :],
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    top_k=cfg.top_k,
                )
                mx.eval(think_tok)
                think_id = int(think_tok.item())
                if think_id == cfg.think_end_id or think_id == eos_id:
                    break
                thinking_tokens.append(think_id)
                output = model(input_ids=think_tok.reshape(1, 1), essence=essence, cache=cache, use_chunked=False)
                cache = output.get("cache") or cache
                _eval_cache(cache)
                next_logits = output["logits"]

            # Step the model on [/THINK] (or last token if EOS) to prime logits
            # for the next visible token
            if think_id == cfg.think_end_id:
                output = model(input_ids=think_tok.reshape(1, 1), essence=essence, cache=cache, use_chunked=False)
                cache = output.get("cache") or cache
                _eval_cache(cache)

            next_logits = output["logits"]
            next_grounding = (
                output["grounding"][:, -1].item()
                if output.get("grounding") is not None
                else 1.0
            )
            continue

        # Tool call: model emits [TOOL_CALL].
        # Advance the model one step on the [TOOL_CALL] token, then collect
        # JSON content tokens until [TOOL_END] or EOS, then parse and return.
        if token_id == cfg.tool_call_id:
            # Step the model forward on [TOOL_CALL] to prime logits for JSON content
            output = model(input_ids=token.reshape(1, 1), essence=essence, cache=cache, use_chunked=False)
            cache = output.get("cache") or cache
            _eval_cache(cache)
            next_logits = output["logits"]

            tool_tokens: list[int] = []
            for _ in range(512):  # max tool call JSON length in tokens
                tool_tok = sample_token(
                    next_logits[:, 0, :],
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    top_k=cfg.top_k,
                )
                mx.eval(tool_tok)
                tid = int(tool_tok.item())
                if tid == cfg.tool_end_id or tid == eos_id:
                    break
                tool_tokens.append(tid)
                output = model(input_ids=tool_tok.reshape(1, 1), essence=essence, cache=cache, use_chunked=False)
                cache = output.get("cache") or cache
                _eval_cache(cache)
                next_logits = output["logits"]

            raw_json = tokenizer.decode(tool_tokens).strip()
            try:
                parsed = json.loads(raw_json)
                tool_call = ToolCall(
                    name=parsed.get("name", ""),
                    arguments=parsed.get("arguments", {}),
                )
            except (json.JSONDecodeError, AttributeError):
                tool_call = ToolCall(name="", arguments={"raw": raw_json})

            text = tokenizer.decode(generated_tokens)
            result = GenerateResult(
                text=text,
                tokens=generated_tokens,
                token_texts=token_texts,
                grounding_scores=grounding_scores,
                halted_positions=halted_positions,
                total_tokens=len(generated_tokens),
                thinking=tokenizer.decode(thinking_tokens),
                tool_calls=[tool_call],
                stop_reason="tool_call",
            )
            return result, cache

        # Normal token: record grounding score aligned with this token position
        grounding_scores.append(g_score)
        generated_tokens.append(token_id)
        token_texts.append(tokenizer.decode([token_id]))

        # Single-token step: pass accumulated cache so Mamba state carries forward
        output = model(
            input_ids=token.reshape(1, 1),
            essence=essence,
            cache=cache,
            use_chunked=False,
        )
        cache = output.get("cache") or cache
        # Evaluate cache lazily-computed arrays to prevent unbounded deferred graph growth
        _eval_cache(cache)

        next_logits = output["logits"]          # (1, 1, vocab)
        next_grounding = (
            output["grounding"][:, -1].item()
            if output.get("grounding") is not None
            else 1.0
        )

    text = tokenizer.decode(generated_tokens)
    stop_reason = "max_tokens" if len(generated_tokens) >= cfg.max_tokens else "eos"

    result = GenerateResult(
        text=text,
        tokens=generated_tokens,
        token_texts=token_texts,
        grounding_scores=grounding_scores,
        halted_positions=halted_positions,
        total_tokens=len(generated_tokens),
        thinking=tokenizer.decode(thinking_tokens),
        stop_reason=stop_reason,
    )
    return result, cache
